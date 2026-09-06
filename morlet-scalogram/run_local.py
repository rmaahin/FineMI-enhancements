"""
Local batch driver for the FINE feature-extraction comparison.

Runs any subset of the 28 task pairs in either arm, sequentially, and SKIPS pairs that
are already saved - so it is safe to Ctrl+C and restart at any point.

Examples
--------
  python run_local.py --selftest                     # front-end + model checks, no training
  python run_local.py --mode cwt --pairs HOC/SPS --smoke
  python run_local.py --mode raw --pairs all         # the baseline arm, 28 pairs
  python run_local.py --mode cwt --pairs all         # the Morlet arm, 28 pairs
  python run_local.py --mode both --pairs all --time-probe
  python run_local.py --mode cwt --pairs all --redo  # ignore existing results

Results are skipped only if produced by the CURRENT pipeline (fine_mi.PIPELINE_VERSION
and cue alignment must match, and predictions.csv must exist); stale directories are
recomputed and the reason is printed.

Cue-alignment diagnostic - reproduce the ORIGINAL notebook's window (starts 500 ms
before the cue, as in the paper) and compare against the cue-aligned result:

  FINE_CUE_SAMPLE=0 python run_local.py --mode raw --pairs HOC/SPS \\
      --results-root diag_cue0

Progress is appended to <results-root>/run_log.txt.
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np

import fine_mi as F


def parse_pairs(spec):
    """'all' | 'HOC/SPS,SAA/SFE' | '0-5,6-7' -> [(a, b), ...]"""
    if spec.strip().lower() == 'all':
        return list(F.ALL_PAIRS)
    name_to_id = {v: k for k, v in F.JOINT_NAMES.items()}
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        sep = '/' if '/' in tok else '-'
        a, b = tok.split(sep)
        a = name_to_id[a.strip().upper()] if a.strip().upper() in name_to_id else int(a)
        b = name_to_id[b.strip().upper()] if b.strip().upper() in name_to_id else int(b)
        if a > b:
            a, b = b, a
        if (a, b) not in F.ALL_PAIRS:
            raise ValueError(f"not a valid pair: {tok}")
        out.append((a, b))
    return out


def log(msg, path):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def selftest(device, args):
    """Front-end and model checks. Mirrors the offline NumPy validation."""
    print("=" * 68)
    print("SELF-TESTS")
    print("=" * 68)
    print(f"kernel K={F.CWT_K} ({F.CWT_K / F.SAMPLING_RATE * 1000:.0f} ms), pad={F.CWT_PAD}")
    print(f"sigma_t {F.SIGMA_T.min()*1000:.1f}-{F.SIGMA_T.max()*1000:.1f} ms | "
          f"3-sigma COI {3*F.SIGMA_T.min()*1000:.0f}-{3*F.SIGMA_T.max()*1000:.0f} ms")
    print(f"context [{F.CTX_START}:{F.CTX_END}] = {F.CTX_LEN} -> "
          f"(C, {F.N_FREQS}, {F.T_OUT}) @ {F.SAMPLING_RATE / F.TIME_DECIM:.1f} Hz")

    # 1. kernel geometry
    assert abs(F.PSI_R_NP.mean(axis=1)).max() < 1e-12, "kernels are not zero-mean"
    assert np.allclose(np.abs(F.PSI_R_NP + 1j * F.PSI_I_NP).sum(axis=1), 1.0), \
        "kernels are not L1-normalised"
    print(f"[ok] {F.N_FREQS} kernels zero-mean and L1-normalised")

    # 2. frequency selectivity
    t = np.arange(F.CTX_LEN) / F.SAMPLING_RATE
    for f0 in (10.0, 20.0):
        sig = np.tile(np.sin(2 * np.pi * f0 * t), (2, 2, 1))
        peak = F.FREQS[int(np.argmax(F.morlet_scalogram(sig, device).mean(axis=(0, 1, 3))))]
        assert abs(np.log10(peak / f0)) < np.log10(F.FREQS[1] / F.FREQS[0]) / 2, \
            f"{f0} Hz peaked at {peak}"
        print(f"[ok] {f0:5.1f} Hz sine -> peak row {peak:6.2f} Hz")

    subs, n_channels = F.load_pair(0, 5, dataset_root=args.dataset_root,
                                   max_subjects=1, verbose=False)
    raw = subs[sorted(subs)[0]]['X'][:8].astype(np.float64)
    mu = raw[:, :, F.WIN_SLICE].mean(axis=(0, 2))[None, :, None]
    sd = raw[:, :, F.WIN_SLICE].std(axis=(0, 2))[None, :, None]
    probe = (raw - mu) / np.where(sd == 0, 1.0, sd)

    # 3. locality/causality. A perturbation at one time sample may only move output
    #    columns within +/- CWT_PAD of it (the wavelet half-support). Stated as a RATIO:
    #    float32 FFT round-off, amplified by log(), leaves ~1e-4 relative noise
    #    everywhere, so an absolute threshold would be meaningless. Independent of
    #    CUE_SAMPLE, unlike a "pre-cue context" test.
    assert F.WIN_SLICE.stop == F.CTX_LEN, "context must end at the decision window"
    a = F.morlet_scalogram(probe, device)
    t0 = F.CTX_LEN // 2
    tamper = probe.copy()
    tamper[:, :, t0] += 10.0
    per_col = np.abs(a - F.morlet_scalogram(tamper, device)).max(axis=(0, 1, 2))
    j0 = t0 - F.WIN_SLICE.start
    lo = (j0 - F.CWT_PAD) // F.TIME_DECIM
    hi = (j0 + F.CWT_PAD) // F.TIME_DECIM + 1
    inside = per_col[lo:hi + 1].max()
    outside = max(per_col[:lo].max(), per_col[hi + 1:].max())
    assert outside / inside < 1e-2, f"leakage {outside/inside:.2e} of the in-support change"
    nz = np.nonzero(per_col > 0.01 * inside)[0]
    assert nz.min() >= lo and nz.max() <= hi,         f"influence {nz.min()}..{nz.max()} escapes support {lo}..{hi}"
    print(f"[ok] causal + local: sample t={t0} moves cols {nz.min()}..{nz.max()} "
          f"(support {lo}..{hi}); off-support leakage {outside/inside:.1e} = round-off")

    # 4. chunk invariance
    assert np.abs(F.morlet_scalogram(probe, device, chunk=3)
                  - F.morlet_scalogram(probe, device, chunk=8)).max() == 0.0
    print("[ok] result independent of CWT chunk size")

    # 4. scale invariance. log() turns an input gain into a CONSTANT offset, which the
    #    per-(c,f) z-score then removes. Assert on the offset being constant - that is
    #    the actual property. The post-normalisation max is checked loosely because
    #    float32 log() loses relative precision at the smallest magnitudes (~1e-5).
    m1 = a
    m2 = F.morlet_scalogram(probe * 1000.0, device)
    shift = m2 - m1
    assert abs(float(shift.mean()) - np.log(1000.0)) < 1e-3, "gain is not a pure offset"
    assert float(shift.std()) < 1e-3, f"offset not constant (sd={shift.std():.2e})"
    n1, _, _ = F.scalogram_normalize(m1, m1, m1)
    n2, _, _ = F.scalogram_normalize(m2, m1, m1)
    diff = np.abs(n1 - n2)
    assert np.percentile(diff, 99.99) < 1e-3 and diff.max() < 1e-2,         f"scale invariance broken: p99.99={np.percentile(diff,99.99):.2e} max={diff.max():.2e}"
    print(f"[ok] scale invariance: 1000x gain -> constant log offset "
          f"{shift.mean():.4f}+-{shift.std():.1e}; post-z-score p99.99="
          f"{np.percentile(diff,99.99):.1e}, max={diff.max():.1e}")

    # 6. shapes, forward and backward under determinism
    import torch
    import torch.nn as nn
    for mode in ('raw', 'cwt'):
        feat, _, _ = F.apply_frontend(mode, probe, probe, probe, device)
        expect = ((n_channels, F.WINDOW_SAMPLES) if mode == 'raw'
                  else (n_channels, F.N_FREQS, F.T_OUT))
        assert feat.shape[1:] == expect, f"{feat.shape[1:]} != {expect}"
        m = F.build_model(mode, n_channels).to(device)
        m.apply(F.init_weights)
        out = m(torch.FloatTensor(feat[:4]).to(device))
        assert out.shape == (4, F.N_CLASSES)
        nn.CrossEntropyLoss()(out, torch.zeros(4, dtype=torch.long, device=device)).backward()
        nparam = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[ok] {mode}: {feat.shape} -> {tuple(out.shape)} | {nparam:,} params | "
              f"{feat[0].nbytes / 1e6:.2f} MB/trial | fwd+bwd deterministic")
        del m, out
        if torch.cuda.is_available():
            print(f"     peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    print("\nALL SELF-TESTS PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['raw', 'cwt', 'both'], default='cwt')
    ap.add_argument('--pairs', default='all',
                    help="'all', or e.g. 'HOC/SPS,SAA/SFE' or '0-5,6-7'")
    ap.add_argument('--epochs', type=int, default=F.N_EPOCHS)
    ap.add_argument('--smoke', action='store_true',
                    help='2 subjects, 5 epochs, results are not saved')
    ap.add_argument('--redo', action='store_true', help='recompute pairs already saved')
    ap.add_argument('--selftest', action='store_true', help='run checks and exit')
    ap.add_argument('--time-probe', action='store_true',
                    help='stop after the first pair and report the projected sweep time')
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--results-root', default=None)
    ap.add_argument('--shard', default=None, metavar='K/N',
                    help="run only shard K of N (1-based), e.g. --shard 2/4. Pairs are "
                         "dealt round-robin so every shard gets a mix of cheap and "
                         "expensive pairs. Shards write to the same results root.")
    ap.add_argument('--cwt-chunk', type=int, default=None,
                    help=f'CWT batch size (default {F.CWT_CHUNK}); lower it if VRAM is tight')
    ap.add_argument('--allow-nondeterministic', action='store_true',
                    help='downgrade determinism errors to warnings (NOT for final runs)')
    args = ap.parse_args()

    if args.cwt_chunk is not None:
        F.CWT_CHUNK = args.cwt_chunk

    results_root = args.results_root or F.RESULTS_ROOT
    log_path = os.path.join(results_root, 'run_log.txt')
    if args.shard:
        k = args.shard.split('/')[0]
        log_path = os.path.join(results_root, f'run_log_shard{k}.txt')

    device, gpu = F.setup_determinism(strict=not args.allow_nondeterministic)
    print(F.describe_environment(device, gpu))
    print(f"dataset: {args.dataset_root or F.DATASET_ROOT}")
    print(f"results: {results_root}\n")

    if args.selftest:
        selftest(device, args)
        return 0

    modes = ['raw', 'cwt'] if args.mode == 'both' else [args.mode]
    pairs = parse_pairs(args.pairs)
    shard_tag = ''
    if args.shard:
        k, n = (int(v) for v in args.shard.split('/'))
        if not 1 <= k <= n:
            raise ValueError(f"bad --shard {args.shard}")
        pairs = pairs[k - 1::n]          # round-robin, so cost is spread evenly
        shard_tag = f" shard={k}/{n}"
        print(f"shard {k}/{n}: {len(pairs)} pairs -> "
              + ', '.join(F.pair_name(a, b) for a, b in pairs))
    n_epochs = 5 if args.smoke else args.epochs
    max_subjects = 2 if args.smoke else None

    jobs = [(m, a, b) for m in modes for (a, b) in pairs]
    todo = [j for j in jobs
            if args.redo or args.smoke
            or not F.is_done(j[1], j[2], j[0], results_root=results_root, verbose=True)]
    skipped = len(jobs) - len(todo)

    log(f"START mode={args.mode}{shard_tag} pairs={len(pairs)} jobs={len(jobs)} "
        f"todo={len(todo)} already_done={skipped} epochs={n_epochs} "
        f"smoke={args.smoke} gpu={gpu}", log_path)

    if not todo:
        log("nothing to do - every requested pair is already saved (use --redo to force)",
            log_path)
        return 0

    durations, failures = [], []
    sweep_start = time.time()

    for i, (mode, a, b) in enumerate(todo, 1):
        name = F.pair_name(a, b)
        eta = ""
        if durations:
            mean_d = sum(durations) / len(durations)
            eta = f" | ETA {(mean_d * (len(todo) - i + 1)) / 3600:.1f} h"
        log(f"[{i}/{len(todo)}] {mode:3s} {name}{eta}", log_path)

        t0 = time.time()
        try:
            results = F.run_pair(a, b, mode, device, n_epochs=n_epochs,
                                 max_subjects=max_subjects,
                                 dataset_root=args.dataset_root, verbose=False,
                                 progress=lambda m: print("      " + m, flush=True))
        except Exception:
            failures.append((mode, name))
            log(f"      FAILED {mode} {name}\n{traceback.format_exc()}", log_path)
            continue
        dt = time.time() - t0
        durations.append(dt)

        if args.smoke:
            accs = [r['time_window_results'][F.WINDOW_MS]['test_accuracy'] for r in results]
            log(f"      smoke ok: {len(results)} subjects, "
                f"mean {np.mean(accs):.2f}% in {dt / 60:.1f} min (not saved)", log_path)
        else:
            _, wide = F.save_results(results, a, b, mode, gpu,
                                     results_root=results_root, verbose=False)
            log(f"      done in {dt / 60:.1f} min | mean "
                f"{wide.mean().iloc[0]:.2f}% | sd {wide.std(ddof=1).iloc[0]:.2f}", log_path)

        if args.time_probe:
            per_pair = dt / 60
            log(f"\nTIME PROBE: {per_pair:.1f} min/pair for mode '{mode}'", log_path)
            log(f"  28 pairs, this mode : {per_pair * 28 / 60:.1f} h", log_path)
            log(f"  56 jobs, both modes : ~{per_pair * 56 / 60:.1f} h "
                f"(upper bound; 'raw' is cheaper)", log_path)
            return 0

    total = (time.time() - sweep_start) / 3600
    log(f"SWEEP COMPLETE: {len(durations)}/{len(todo)} jobs in {total:.2f} h", log_path)
    if failures:
        log(f"FAILURES ({len(failures)}): " + ", ".join(f"{m}:{n}" for m, n in failures),
            log_path)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
