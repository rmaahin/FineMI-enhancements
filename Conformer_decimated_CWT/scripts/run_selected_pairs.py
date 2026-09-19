import src.determinism  # noqa: F401 — must be first (sets CUBLAS env before torch loads)
# Run selected joint pairs through Conformer A, Conformer B and Conformer B + CWT on
# the FineMI 0.5-3 Hz data decimated to 50 Hz (D=5), then build the comparison table.
#
# Run from Conformer_decimated_CWT/ as a module (puts this folder on sys.path so
# `import src.*` resolves):
#
#   cd "FineMI-enhancements/Conformer_decimated_CWT"
#   python -m scripts.run_selected_pairs --pairs WAA_SAA HOC_WFE EPS_SPS
#   python -m scripts.run_selected_pairs --pairs 2_6 0_1 3_5 --mode smoke
#   python -m scripts.run_selected_pairs --pairs WAA_SAA --models conformer_b_cwt
#   python -m scripts.run_selected_pairs --pairs all            # full 28-pair sweep
#
# Pairs: joint names (WAA_SAA, WAA/SAA), class ids (2_6), pair indices (0-27), or all.
# Joints: 0 HOC, 1 WFE, 2 WAA, 3 EPS, 4 EFE, 5 SPS, 6 SAA, 7 SFE.
# On TACC use submit_ls6.slurm, which splits the pairs across the node's 3 GPUs.
#
# Output (all inside this folder):
#   results/<mode>/<model>/<PAIR>/   per_subject.csv, per_fold.csv,
#                                    wide_subject_x_window.csv, results.{json,pkl}
#   results/<mode>/<model>/config.json
#   results/<mode>/comparison/       comparison_table.md + CSVs (see src/compare.py)
#   results/<mode>/logs/run_<timestamp>.log   copy of everything printed
#
# Completed (pair, model) runs are skipped, so an interrupted run can simply be
# restarted with the same command; --overwrite re-trains them. Every fold
# re-seeds torch/numpy/loaders, so results do not depend on which models or
# pairs share a process.

import argparse
import os
import sys
import time
import traceback

from src.config import (DATASET_ROOT_DEFAULT, RESULTS_ROOT_DEFAULT, JOINT_NAMES, MODELS,
                        MODE_OVERRIDES, make_config, mode_root, pair_slug, parse_pairs)


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
        description="Train Conformer A / B / B + CWT on selected pairs (0.5-3 Hz, decimated to 50 Hz).")
    p.add_argument('--pairs', required=True, nargs='+', metavar='PAIR',
                   help="pairs to run, e.g. WAA_SAA HOC_WFE EPS_SPS (or 2_6, pair index 0-27, "
                        "or all)")
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS),
                   help=f"architectures to run (default: all of {' '.join(MODELS)})")
    p.add_argument('--mode', choices=list(MODE_OVERRIDES), default='full',
                   help="smoke = 2 subjects/2 folds/2 epochs/all 4 windows; "
                        "full = all subjects, 5 folds, 50 epochs (default)")
    p.add_argument('--dataset-root', default=DATASET_ROOT_DEFAULT,
                   help=f"folder with subject*_eeg_epochs_0.5_3hz_*.npz (default: {DATASET_ROOT_DEFAULT})")
    p.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT,
                   help=f"results folder (default: {RESULTS_ROOT_DEFAULT})")
    p.add_argument('--overwrite', action='store_true',
                   help="re-train (pair, model) runs that already have results.json")
    p.add_argument('--no-table', action='store_true',
                   help="skip building the comparison table at the end")
    p.add_argument('--no-log-file', action='store_true',
                   help="don't write results/<mode>/logs/run_*.log (the SLURM job redirects "
                        "each worker's output to its own log instead)")
    p.add_argument('--list-pairs', action='store_true',
                   help="print the normalized, de-duplicated pair names and exit (no training)")
    p.add_argument('--pending', action='store_true',
                   help="with --list-pairs: only list pairs where at least one of --models has no "
                        "results.json yet for --mode (ignored with --overwrite)")
    args = p.parse_args()

    try:
        pairs = parse_pairs(args.pairs)
    except ValueError as e:
        p.error(str(e))
    models = list(dict.fromkeys(args.models))
    root = mode_root(args.results_root, args.mode)

    if args.list_pairs:
        if args.pending and not args.overwrite:
            from src.utils import is_pair_done
            pairs = [(a, b) for a, b in pairs
                     if not all(is_pair_done(os.path.join(root, m, pair_slug(a, b)))
                                for m in models)]
        print(' '.join(pair_slug(a, b) for a, b in pairs))
        return 0

    slugs = [pair_slug(a, b) for a, b in pairs]
    if not os.path.isdir(args.dataset_root):
        p.error(f"--dataset-root does not exist: {args.dataset_root}")

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

    print(f"run_selected_pairs: mode={args.mode}", flush=True)
    print(f"  pairs  = {slugs}", flush=True)
    print(f"  models = {models}", flush=True)
    print(f"  dataset_root = {args.dataset_root}", flush=True)
    print(f"  results_root = {root}", flush=True)
    print(f"  log = {log_path}", flush=True)

    from src.train import train_pair   # after the Tee so import-time output is logged too

    for m in models:
        make_config(m, args.mode, args.dataset_root, args.results_root).to_json(
            os.path.join(root, m, 'config.json'))

    failed = []
    t0 = time.time()
    total = len(pairs) * len(models)
    run_i = 0
    # pair outer, model inner: the pair's data is loaded once and reused by every model
    for a, b in pairs:
        for m in models:
            run_i += 1
            slug = pair_slug(a, b)
            print(f"\n##### [{run_i}/{total}] {slug} ({JOINT_NAMES[a]} vs {JOINT_NAMES[b]}) "
                  f"model={m} #####", flush=True)
            src.determinism.apply_all(seed=42)
            cfg = make_config(m, args.mode, args.dataset_root, args.results_root)
            try:
                train_pair(a, b, cfg, overwrite=args.overwrite)
            except Exception:
                traceback.print_exc()
                sys.stderr.flush()
                print(f"FAILED {slug} [{m}]", flush=True)
                failed.append(f"{slug} [{m}]")

    print(f"\nTraining finished in {(time.time() - t0) / 60:.1f} min", flush=True)

    if not args.no_table:
        from src.compare import build_comparison
        try:
            build_comparison(root, pairs, models=models)
        except Exception:
            traceback.print_exc()
            failed.append('comparison table')

    if failed:
        print(f"\nDone with failures: {failed}", flush=True)
        return 1
    print("\nDone.", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
