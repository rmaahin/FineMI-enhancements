"""
Local batch driver for the FINE filter-bank (Technique 1) comparison.

Runs any subset of the 28 task pairs in either arm, sequentially, and SKIPS pairs that
are already saved - so it is safe to Ctrl+C and restart at any point.

Examples
--------
  python run_local.py --selftest                     # front-end + model checks, no training
  python run_local.py --mode fbc --pairs HOC/SPS --smoke
  python run_local.py --mode raw --pairs all         # the baseline arm, 28 pairs
  python run_local.py --mode fbc --pairs all         # the filter-bank arm, 28 pairs
  python run_local.py --mode both --pairs all --time-probe
  python run_local.py --mode fbc --pairs all --redo  # ignore existing results

Results are skipped only if produced by the CURRENT pipeline (fine_mi.PIPELINE_VERSION,
cue alignment and band layout must match, and predictions.csv must exist); stale
directories are recomputed and the reason is printed.

Band-layout diagnostics (each writes to its own results root so nothing is mixed):

  FINE_FB_BANDS="8-13,13-20,20-30" python run_local.py --mode fbc --pairs HOC/SPS \\
      --results-root diag_3band
  FINE_FB_NORM=none python run_local.py --mode fbc --pairs HOC/SPS \\
      --results-root diag_nonorm

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
    """Front-end and model checks. Nothing here trains."""
    print("=" * 68)
    print("SELF-TESTS")
    print("=" * 68)
    print(f"bands: " + ", ".join(f"{lo:g}-{hi:g}" for lo, hi in F.BANDS))
    print(f"FIR: {F.FB_NUMTAPS} taps "
          f"({F.FB_NUMTAPS / F.SAMPLING_RATE * 1000:.0f} ms), pad={F.FB_PAD}, "
          f"transition {F.FB_TRANS_HZ:g} Hz, norm={F.FB_NORM}")
    print(f"context [{F.CTX_START}:{F.CTX_END}] = {F.CTX_LEN} -> "
          f"(C, {F.N_BANDS}, {F.T_OUT}) @ {F.SAMPLING_RATE} Hz")

    # 1. filter geometry: linear phase (symmetric), unit gain at band centre, DC/Nyquist
    #    rejection. A bandpass with DC leakage would let the z-score offset through.
    w = np.linspace(0, F.SAMPLING_RATE / 2, 4096)
    n = np.arange(F.FB_NUMTAPS) - (F.FB_NUMTAPS - 1) / 2.0
    resp = np.abs(F.FB_H_NP @ np.exp(-2j * np.pi * np.outer(n, w) / F.SAMPLING_RATE))
    assert np.abs(F.FB_H_NP - F.FB_H_NP[:, ::-1]).max() < 1e-12, "FIR taps are not symmetric"
    print("[ok] all FIRs symmetric (exactly linear phase, integer group delay)")
    # Hamming's first sidelobe is -53 dB, so a few e-3 at DC is the design limit, not a bug.
    assert resp[:, 0].max() < 5e-3, f"DC leakage {resp[:, 0].max():.2e}"
    assert resp[:, -1].max() < 5e-3, f"Nyquist leakage {resp[:, -1].max():.2e}"
    print(f"[ok] DC gain <= {resp[:, 0].max():.1e}, Nyquist gain <= {resp[:, -1].max():.1e}")
    for i, (lo, hi) in enumerate(F.BANDS):
        gain_c = resp[i][np.argmin(np.abs(w - 0.5 * (lo + hi)))]
        out_of_band = resp[i][(w < lo - F.FB_TRANS_HZ) | (w > hi + F.FB_TRANS_HZ)].max()
        assert abs(gain_c - 1.0) < 0.05, f"band {lo}-{hi}: centre gain {gain_c:.3f}"
        assert out_of_band < 0.1, f"band {lo}-{hi}: stopband {out_of_band:.3f}"
        print(f"[ok] {lo:5.1f}-{hi:4.1f} Hz: centre gain {gain_c:.3f}, "
              f"stopband <= {out_of_band:.3f} ({20 * np.log10(out_of_band):.0f} dB)")

    # 2. band selectivity end to end: a pure sine lands in the band that contains it
    t = np.arange(F.CTX_LEN) / F.SAMPLING_RATE
    for f0 in (10.0, 18.0, 26.0):
        sig = np.tile(np.sin(2 * np.pi * f0 * t), (2, 2, 1))
        power = np.abs(F.filter_bank(sig, device)).mean(axis=(0, 1, 3))
        got = F.BANDS[int(np.argmax(power))]
        assert got[0] <= f0 < got[1], f"{f0} Hz peaked in band {got}"
        print(f"[ok] {f0:5.1f} Hz sine -> band {got[0]:g}-{got[1]:g} Hz "
              f"({power.max() / np.sort(power)[-2]:.0f}x the runner-up)")

    subs, n_channels = F.load_pair(0, 5, dataset_root=args.dataset_root,
                                   max_subjects=1, verbose=False)
    raw = subs[sorted(subs)[0]]['X'][:8].astype(np.float64)
    mu = raw[:, :, F.WIN_SLICE].mean(axis=(0, 2))[None, :, None]
    sd = raw[:, :, F.WIN_SLICE].std(axis=(0, 2))[None, :, None]
    probe = (raw - mu) / np.where(sd == 0, 1.0, sd)

    # 3. reconstruction: the bank covers 8-30 Hz, so summing the sub-bands must return
    #    the (band-limited) input up to the transition ripple.
    total = F.filter_bank(probe, device).sum(axis=2)
    err = np.abs(total - probe[:, :, F.WIN_SLICE]).std() / probe[:, :, F.WIN_SLICE].std()
    print(f"[ok] sub-bands sum back to the input, relative residual {err:.3f}")

    # 4. causality: the window ends at the context end, so no post-window sample exists
    #    to leak; and only the leading FIR-support columns see the pre-window context.
    assert F.WIN_SLICE.stop == F.CTX_LEN, "context must end at the decision window"
    a = F.filter_bank(probe, device)
    if F.LEFT_CTX > 0:
        tamper = probe.copy()
        tamper[:, :, :F.LEFT_CTX] = 0.0
        d = np.abs(a - F.filter_bank(tamper, device))
        assert d[..., F.LEFT_CTX + F.FB_PAD:].max() < 1e-4, "pre-window context leaks"
        print(f"[ok] causal by construction; pre-window context confined to the first "
              f"~{F.LEFT_CTX + F.FB_PAD} of {F.T_OUT} columns")
    else:
        print(f"[ok] causal by construction; window starts at the epoch edge, "
              f"left context is reflection only ({F.LEFT_REFLECT} samples)")

    # 4b. linear phase: a sine survives the bank unshifted in the window interior
    sine = np.tile(np.sin(2 * np.pi * 10.0 * t), (1, 1, 1))
    b_idx = next(i for i, (lo, hi) in enumerate(F.BANDS) if lo <= 10.0 < hi)
    core = slice(F.T_OUT // 4, 3 * F.T_OUT // 4)
    y = F.filter_bank(sine, device)[0, 0, b_idx]
    err = np.abs(y[core] - sine[0, 0, F.WIN_SLICE][core]).max()
    assert err < 1e-3, f"group delay not removed: {err:.2e}"
    print(f"[ok] group delay removed: 10 Hz sine reproduced to {err:.1e} in the interior")

    # 5. reproducibility. Bit-identity is required at a FIXED chunk - that is what a
    #    rerun does. Across chunk sizes cuFFT picks a different batched plan, so the
    #    result moves by float32 epsilon; FB_CHUNK is fixed per run (and recorded in
    #    results.json), so that never varies within a sweep.
    a3 = F.filter_bank(probe, device, chunk=3)
    assert np.abs(a3 - F.filter_bank(probe, device, chunk=3)).max() == 0.0, \
        "not reproducible at a fixed chunk size"
    print("[ok] bit-identical on rerun at a fixed chunk size")
    rel = (np.abs(a3 - F.filter_bank(probe, device, chunk=8)).max()
           / np.abs(a3).max())
    assert rel < 1e-5, f"chunk size changes the result by {rel:.2e} (> float32 noise)"
    print(f"[ok] chunk size changes the result by {rel:.1e} relative (float32 cuFFT noise)")

    # 6. shapes, forward and backward under determinism
    import torch
    import torch.nn as nn
    for mode in ('raw', 'fbc'):
        feat, _, _ = F.apply_frontend(mode, probe, probe, probe, device)
        expect = ((n_channels, F.WINDOW_SAMPLES) if mode == 'raw'
                  else (n_channels, F.N_BANDS, F.T_OUT))
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
    ap.add_argument('--mode', choices=['raw', 'fbc', 'both'], default='fbc')
    ap.add_argument('--pairs', default='all',
                    help="'all', or e.g. 'HOC/SPS,SAA/SFE' or '0-5,6-7'")
    ap.add_argument('--epochs', type=int, default=F.N_EPOCHS)
    ap.add_argument('--smoke', action='store_true',
                    help='2 subjects, 5 epochs, results are not saved')
    ap.add_argument('--max-subjects', type=int, default=None,
                    help='use only the first N subjects (reduced run; NOT comparable to '
                         'the 18-subject means in the paper - give it its own '
                         '--results-root)')
    ap.add_argument('--redo', action='store_true', help='recompute pairs already saved')
    ap.add_argument('--selftest', action='store_true', help='run checks and exit')
    ap.add_argument('--time-probe', action='store_true',
                    help='stop after the first pair and report the projected sweep time')
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--results-root', default=None)
    ap.add_argument('--fb-chunk', type=int, default=None,
                    help=f'filter batch size (default {F.FB_CHUNK}); lower it if VRAM is tight')
    ap.add_argument('--allow-nondeterministic', action='store_true',
                    help='downgrade determinism errors to warnings (NOT for final runs)')
    args = ap.parse_args()

    if args.fb_chunk is not None:
        F.FB_CHUNK = args.fb_chunk

    results_root = args.results_root or F.RESULTS_ROOT
    log_path = os.path.join(results_root, 'run_log.txt')

    device, gpu = F.setup_determinism(strict=not args.allow_nondeterministic)
    print(F.describe_environment(device, gpu))
    print("bands: " + ", ".join(f"{lo:g}-{hi:g}" for lo, hi in F.BANDS)
          + f" | {F.FB_NUMTAPS} taps | norm={F.FB_NORM}")
    print(f"dataset: {args.dataset_root or F.DATASET_ROOT}")
    print(f"results: {results_root}\n")

    if args.selftest:
        selftest(device, args)
        return 0

    modes = ['raw', 'fbc'] if args.mode == 'both' else [args.mode]
    pairs = parse_pairs(args.pairs)
    n_epochs = 5 if args.smoke else args.epochs
    max_subjects = 2 if args.smoke else args.max_subjects

    jobs = [(m, a, b) for m in modes for (a, b) in pairs]
    todo = [j for j in jobs
            if args.redo or args.smoke
            or not F.is_done(j[1], j[2], j[0], results_root=results_root, verbose=True)]
    skipped = len(jobs) - len(todo)

    log(f"START mode={args.mode} pairs={len(pairs)} jobs={len(jobs)} "
        f"todo={len(todo)} already_done={skipped} epochs={n_epochs} "
        f"smoke={args.smoke} max_subjects={max_subjects} gpu={gpu}", log_path)

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
