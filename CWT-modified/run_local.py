"""
Batch driver for the CWT-modified FINE comparison (0.5-3 Hz, decimated to 50 Hz).

Runs any subset of the 28 task pairs in either arm, sequentially, and SKIPS pairs that
are already saved - so it is safe to Ctrl+C / resubmit and restart at any point.

Examples
--------
  python run_local.py --selftest                     # front-end + model checks, no training
  python run_local.py --mode cwt --pairs HOC/SPS --smoke
  python run_local.py --mode raw --pairs all         # decimated baseline arm, 28 pairs
  python run_local.py --mode cwt --pairs all         # Morlet arm, 28 pairs
  python run_local.py --mode both --pairs all --time-probe
  python run_local.py --mode cwt --pairs all --redo  # ignore existing results
  python run_local.py --mode both --pairs "HOC_SPS WAA_SAA" --shard 1/3

Results are skipped only if produced by the CURRENT pipeline (fine_mi.PIPELINE_VERSION,
band, decimation, cue alignment and checkpoint rule must match, and predictions.csv
must exist); stale directories are recomputed and the reason is printed.

Env knobs (see fine_mi.py): FINE_DATASET_ROOT, FINE_RESULTS_ROOT, FINE_CUE_SAMPLE,
FINE_BEST_CKPT, FINE_MTC_KERNELS_2D, FINE_CWT_CHUNK.

Progress is appended to <results-root>/run_log.txt (or run_log_shardK.txt).
"""
import argparse
import os
import re
import sys
import time
import traceback

import numpy as np

import fine_mi as F


def parse_pairs(spec):
    """'all' | 'HOC/SPS,SAA/SFE' | 'HOC_SPS WAA_SAA' | '0-5,6-7' -> [(a, b), ...]"""
    if spec.strip().lower() == 'all':
        return list(F.ALL_PAIRS)
    name_to_id = {v: k for k, v in F.JOINT_NAMES.items()}
    out = []
    for tok in re.split(r'[,\s]+', spec.strip()):
        if not tok:
            continue
        sep = '/' if '/' in tok else ('_' if '_' in tok else '-')
        a, b = tok.split(sep)
        a = name_to_id[a.strip().upper()] if a.strip().upper() in name_to_id else int(a)
        b = name_to_id[b.strip().upper()] if b.strip().upper() in name_to_id else int(b)
        if a > b:
            a, b = b, a
        if (a, b) not in F.ALL_PAIRS:
            raise ValueError(f"not a valid pair: {tok}")
        if (a, b) not in out:
            out.append((a, b))
    return out


def log(msg, path):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def selftest(device, args):
    """Front-end, decimation and model checks. No training."""
    print("=" * 68)
    print("SELF-TESTS")
    print("=" * 68)
    fs = F.SAMPLING_RATE
    print(f"band {F.BAND_TAG} | decim {F.DECIM}: {F.ORIG_SAMPLING_RATE} -> {fs} Hz | "
          f"window {F.WINDOW_SAMPLES} samples = {F.WINDOW_MS} ms")
    print(f"kernel K={F.CWT_K} ({F.CWT_K / fs:.1f} s), pad={F.CWT_PAD}")
    print(f"freqs {F.FREQS[0]:.2f}-{F.FREQS[-1]:.2f} Hz x{F.N_FREQS} | "
          f"sigma_t {F.SIGMA_T.min():.2f}-{F.SIGMA_T.max():.2f} s | "
          f"per-row half-support {F.ROW_HALF_SUPPORT.min()}-{F.ROW_HALF_SUPPORT.max()} samples")
    print(f"context [{F.CTX_START}:{F.CTX_END}] = {F.CTX_LEN} -> "
          f"(C, {F.N_FREQS}, {F.T_OUT}) @ {fs} Hz")

    # 0. receptive-field arithmetic from the paper: 42 samples = 168 ms @250, 840 ms @50
    rf = ((31 + 2) + 1) + 4 * 2          # MTC 31 -> spatial 33 -> pool 34 -> DS-conv 42
    print(f"[ok] FINE-1D receptive field {rf} samples = "
          f"{rf / F.ORIG_SAMPLING_RATE * 1000:.0f} ms @ {F.ORIG_SAMPLING_RATE} Hz, "
          f"{rf / fs * 1000:.0f} ms @ {fs} Hz")

    # 1. kernel geometry
    assert abs(F.PSI_R_NP.mean(axis=1)).max() < 1e-12, "kernels are not zero-mean"
    assert np.allclose(np.abs(F.PSI_R_NP + 1j * F.PSI_I_NP).sum(axis=1), 1.0), \
        "kernels are not L1-normalised"
    print(f"[ok] {F.N_FREQS} kernels zero-mean and L1-normalised")

    # 2a. frequency selectivity of the kernel bank itself (exact, no edge effects):
    #     psi_c(t) = g(t) e^{+i 2pi f_c t}, so its response to a tone at f0 is
    #     |sum psi_c(t) e^{-i 2pi f0 t}| = |sum g(t) e^{i 2pi (f_c - f0) t}|, which must
    #     peak at the row nearest f0 (the + sign would probe f_c + f0 instead).
    step = np.log10(F.FREQS[1] / F.FREQS[0])
    tk = np.arange(F.CWT_K) / fs
    psi = F.PSI_R_NP + 1j * F.PSI_I_NP
    for f0 in (0.7, 1.0, 2.0, 3.0):
        resp = np.abs(psi @ np.exp(-2j * np.pi * f0 * tk))
        peak = F.FREQS[int(np.argmax(resp))]
        assert abs(np.log10(peak / f0)) <= step / 2 * 1.01, f"{f0} Hz peaked at {peak}"
        print(f"[ok] kernel bank: {f0:4.1f} Hz -> peak row {peak:5.2f} Hz "
              f"(response {resp.max():.3f})")

    # 2b. end-to-end on a 4 s sine. Every column is within the wavelet's edge region at
    #     these frequencies (half-support 93-239 samples on a 200-sample window), so
    #     this is only a loose check: LINEAR magnitude, central half of the window,
    #     peak within +-2 rows (~17 %).
    t = np.arange(F.CTX_LEN) / fs
    c0, c1 = F.T_OUT // 4, 3 * F.T_OUT // 4
    for f0 in (1.0, 2.0):
        sig = np.tile(np.sin(2 * np.pi * f0 * t), (2, 2, 1))
        mag = np.exp(F.morlet_scalogram(sig, device))[..., c0:c1].mean(axis=(0, 1, 3))
        peak = F.FREQS[int(np.argmax(mag))]
        assert abs(np.log10(peak / f0)) <= 2 * step * 1.01, \
            f"{f0} Hz sine peaked at {peak:.2f} Hz end-to-end"
        print(f"[ok] {f0:4.1f} Hz sine (4 s, edge-affected) -> peak row {peak:5.2f} Hz")

    subs, n_channels = F.load_pair(0, 5, dataset_root=args.dataset_root,
                                   max_subjects=1, verbose=False)
    raw = subs[sorted(subs)[0]]['X'][:8].astype(np.float64)
    assert raw.shape[2] == F.CTX_LEN and F.CTX_LEN >= F.WINDOW_SAMPLES
    print(f"[ok] decimated context loads as {raw.shape} (C={n_channels}, T={F.CTX_LEN})")
    mu = raw[:, :, F.WIN_SLICE].mean(axis=(0, 2))[None, :, None]
    sd = raw[:, :, F.WIN_SLICE].std(axis=(0, 2))[None, :, None]
    probe = (raw - mu) / np.where(sd == 0, 1.0, sd)

    # 3. locality/causality on the TOP frequency row (the only row whose support fits
    #    inside a 4 s window). A perturbation at one sample may only move output columns
    #    within +/- its half-support. Stated as a RATIO: float32 FFT round-off,
    #    amplified by log(), leaves ~1e-4 relative noise everywhere.
    assert F.WIN_SLICE.stop == F.CTX_LEN, "context must end at the decision window"
    a = F.morlet_scalogram(probe, device)
    t0 = F.CTX_LEN // 2
    tamper = probe.copy()
    tamper[:, :, t0] += 10.0
    b = F.morlet_scalogram(tamper, device)
    hs = int(F.ROW_HALF_SUPPORT[-1])
    per_col = np.abs(a - b)[:, :, -1, :].max(axis=(0, 1))
    j0 = t0 - F.WIN_SLICE.start
    lo, hi = max(0, j0 - hs), min(F.T_OUT - 1, j0 + hs)
    assert lo > 0 and hi < F.T_OUT - 1, "top row support must leave columns to test"
    inside = per_col[lo:hi + 1].max()
    outside = max(per_col[:lo].max(), per_col[hi + 1:].max())
    assert outside / inside < 1e-2, f"leakage {outside/inside:.2e} of the in-support change"
    nz = np.nonzero(per_col > 0.01 * inside)[0]
    assert nz.min() >= lo and nz.max() <= hi, \
        f"influence {nz.min()}..{nz.max()} escapes support {lo}..{hi}"
    print(f"[ok] causal + local ({F.FREQS[-1]:.1f} Hz row): sample t={t0} moves cols "
          f"{nz.min()}..{nz.max()} (support {lo}..{hi}); off-support leakage "
          f"{outside/inside:.1e} = round-off")

    # 4. chunk invariance
    assert np.abs(F.morlet_scalogram(probe, device, chunk=3)
                  - F.morlet_scalogram(probe, device, chunk=8)).max() == 0.0
    print("[ok] result independent of CWT chunk size")

    # 5. scale invariance. log() turns an input gain into a CONSTANT offset, which the
    #    per-(c,f) z-score then removes.
    m1 = a
    m2 = F.morlet_scalogram(probe * 1000.0, device)
    shift = m2 - m1
    assert abs(float(shift.mean()) - np.log(1000.0)) < 1e-3, "gain is not a pure offset"
    assert float(shift.std()) < 1e-3, f"offset not constant (sd={shift.std():.2e})"
    n1, _, _ = F.scalogram_normalize(m1, m1, m1)
    n2, _, _ = F.scalogram_normalize(m2, m1, m1)
    diff = np.abs(n1 - n2)
    assert np.percentile(diff, 99.99) < 1e-3 and diff.max() < 1e-2, \
        f"scale invariance broken: p99.99={np.percentile(diff,99.99):.2e} max={diff.max():.2e}"
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
                    help="'all', or e.g. 'HOC/SPS,SAA/SFE', 'HOC_SPS WAA_SAA' or '0-5,6-7'")
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
                    help="run only shard K of N (1-based), e.g. --shard 2/3. Pairs are "
                         "dealt round-robin. Shards write to the same results root.")
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
        f"smoke={args.smoke} decim={F.DECIM} best_ckpt={F.BEST_CKPT} gpu={gpu}", log_path)

    if not todo:
        log("nothing to do - every requested pair is already saved (use --redo to force)",
            log_path)
        return 0

    durations, failures = [], []
    probe = {}
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
            probe[mode] = dt / 60.0
            if len(probe) < len(modes):
                log(f"      probe: {mode} = {probe[mode]:.1f} min/pair; "
                    f"continuing to time the other arm", log_path)
                continue
            log("", log_path)
            log(f"TIME PROBE ({', '.join(f'{m} {v:.1f} min/pair' for m, v in probe.items())})",
                log_path)
            total_h = sum(v * 28 for v in probe.values()) / 60.0
            for m, v in probe.items():
                log(f"  28 pairs, {m:3s} : {v * 28 / 60:.1f} h", log_path)
            log(f"  ALL {len(probe) * 28} jobs   : {total_h:.1f} h on one GPU", log_path)
            for n in (2, 3):
                log(f"    with {n} shards: {total_h / n:.1f} h", log_path)
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
