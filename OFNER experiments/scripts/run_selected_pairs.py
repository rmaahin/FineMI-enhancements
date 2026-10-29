import src.determinism  # noqa: F401 — must be first (sets CUBLAS env before torch loads)
# Train LEAN (conformer_b_slim) and EEG Conformer (conformer_a) on selected OFNER pairs,
# on one band: 0.5_3hz (decimated to 50 Hz) or 8_30hz (kept at 250 Hz).
#
# Run from this project folder as a module (puts it on sys.path so `import src.*`
# resolves). run_pipeline.sh calls this once per GPU; by hand:
#
#   python -m scripts.run_selected_pairs --band 0.5_3hz --pairs EF_EE SUP_PRO HC_HO --mode smoke
#   python -m scripts.run_selected_pairs --band 8_30hz --pairs all          # both models
#   python -m scripts.run_selected_pairs --band 0.5_3hz --pairs all --models conformer_a
#
# Pairs: class names (EF_EE, EF/EE), class ids (0_1), pair indices (0-14), or all.
# Classes: 0 EF, 1 EE, 2 SUP, 3 PRO, 4 HC, 5 HO.
#
# Output (inside this folder), for <model> in conformer_b_slim, conformer_a:
#   results/<band>/<mode>/<model>/<PAIR>/   per_subject.csv, per_fold.csv,
#                                           wide_subject_x_window.csv, results.{json,pkl}
#   results/<band>/<mode>/<model>/config.json
#   results/<band>/<mode>/logs/run_<timestamp>.log   copy of everything printed
#
# Completed pairs are skipped, so an interrupted run can simply be restarted with the
# same command; --overwrite re-trains them.

import argparse
import os
import sys
import time
import traceback

from src.config import (BAND_PROFILES, CLASS_NAMES, DEFAULT_BAND, MODELS, MODE_OVERRIDES,
                        RESULTS_ROOT_DEFAULT, dataset_root_default, make_config, mode_root,
                        normalize_band, pair_slug, parse_pairs)


class _Tee:
    """Mirror writes to the console and a log file."""

    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def __getattr__(self, name):   # isatty, encoding, fileno, ... come from the console stream
        return getattr(self.stream, name)


def main():
    p = argparse.ArgumentParser(
        description="Train LEAN and EEG Conformer on selected OFNER pairs, one band.")
    p.add_argument('--pairs', required=True, nargs='+', metavar='PAIR',
                   help="pairs to run, e.g. EF_EE SUP_PRO HC_HO (or 0_1, pair index 0-14, or all)")
    p.add_argument('--band', default=DEFAULT_BAND,
                   help=f"one of {', '.join(BAND_PROFILES)} (default {DEFAULT_BAND})")
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--mode', choices=list(MODE_OVERRIDES), default='full',
                   help="smoke = 2 subjects/2 folds/2 epochs/all 3 windows; "
                        "full = all subjects, 5 folds, 50 epochs (default)")
    p.add_argument('--dataset-root', default=None,
                   help="folder with subject*_eeg_epochs_<band>_*.npz "
                        "(default: $SCRATCH/ofner/ofner_mi_<band> or /workspace/ofner/ofner_mi_<band>)")
    p.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT,
                   help=f"results folder (default: {RESULTS_ROOT_DEFAULT})")
    p.add_argument('--overwrite', action='store_true',
                   help="re-train pairs that already have results.json")
    p.add_argument('--no-log-file', action='store_true',
                   help="don't write results/<band>/<mode>/logs/run_*.log")
    p.add_argument('--list-pairs', action='store_true',
                   help="print the normalized, de-duplicated pair names and exit (no training)")
    p.add_argument('--pending', action='store_true',
                   help="with --list-pairs: only list pairs without results.json for --band "
                        "and --mode (ignored with --overwrite)")
    args = p.parse_args()

    try:
        pairs = parse_pairs(args.pairs)
        band = normalize_band(args.band)
    except ValueError as e:
        p.error(str(e))
    models = list(dict.fromkeys(args.models))
    root = mode_root(args.results_root, band, args.mode)
    dataset_root = args.dataset_root or dataset_root_default(band)

    if args.list_pairs:
        if args.pending and not args.overwrite:
            from src.utils import is_pair_done
            pairs = [(a, b) for a, b in pairs
                     if not all(is_pair_done(os.path.join(root, m, pair_slug(a, b)), band)
                                for m in models)]
        print(' '.join(pair_slug(a, b) for a, b in pairs))
        return 0

    slugs = [pair_slug(a, b) for a, b in pairs]
    if not os.path.isdir(dataset_root):
        p.error(f"--dataset-root does not exist: {dataset_root}")

    if args.no_log_file:
        log_path = '(stdout only)'
    else:
        log_dir = os.path.join(root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        # pid keeps names unique when several workers start in the same second
        log_path = os.path.join(
            log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}.log")
        log_file = open(log_path, 'w', encoding='utf-8')
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"run_selected_pairs: band={band} mode={args.mode}", flush=True)
    print(f"  pairs  = {slugs}", flush=True)
    print(f"  models = {models}", flush=True)
    print(f"  dataset_root = {dataset_root}", flush=True)
    print(f"  results_root = {root}", flush=True)
    print(f"  log = {log_path}", flush=True)

    from src.train import train_pair   # after the Tee so import-time output is logged too

    for m in models:
        make_config(m, args.mode, band, dataset_root, args.results_root).to_json(
            os.path.join(root, m, 'config.json'))

    failed = []
    t0 = time.time()
    total = len(pairs) * len(models)
    run_i = 0
    for a, b in pairs:
        for m in models:
            run_i += 1
            slug = pair_slug(a, b)
            print(f"\n##### [{run_i}/{total}] {slug} ({CLASS_NAMES[a]} vs {CLASS_NAMES[b]}) "
                  f"model={m} band={band} #####", flush=True)
            src.determinism.apply_all(seed=42)
            cfg = make_config(m, args.mode, band, dataset_root, args.results_root)
            try:
                train_pair(a, b, cfg, overwrite=args.overwrite)
            except Exception:
                traceback.print_exc()
                sys.stderr.flush()
                print(f"FAILED {slug} [{m}]", flush=True)
                failed.append(f"{slug} [{m}]")

    print(f"\nTraining finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    if failed:
        print(f"\nDone with failures: {failed}", flush=True)
        return 1
    print("\nDone.", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
