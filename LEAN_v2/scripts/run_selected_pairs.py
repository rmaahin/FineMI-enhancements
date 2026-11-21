import src.determinism  # noqa: F401 — must be first (sets CUBLAS env before torch loads)
# Train LEAN_v2 presets on selected joint pairs. run_vast.sh calls this once per worker;
# by hand, from this folder:
#
#   python -m scripts.run_selected_pairs --mode screen --pairs screen --models ladder
#   python -m scripts.run_selected_pairs --mode full --pairs all --models lean_v2_r7 eegnet_50_recipe
#
# --models takes preset names and/or groups (ladder, loo, training, baselines, all; see
# src/config.py). --pairs takes joint names (WAA_SAA), class ids (2_6), pair indices
# (0-27), 'all', or 'screen' (the 6 SCREEN_PAIRS).
#
# Output: results/<mode>/<model>/<PAIR>/ (per_subject.csv, per_fold.csv,
# wide_subject_x_window.csv, results.{json,pkl}) and results/<mode>/<model>/config.json.
# Finished pairs are skipped; --overwrite re-trains them.

import argparse
import os
import sys
import time
import traceback

from src.config import (DATASET_ROOT_DEFAULT, RESULTS_ROOT_DEFAULT, JOINT_NAMES,
                        MODE_OVERRIDES, BAND_LABEL, SCREEN_PAIRS, MODEL_GROUPS,
                        expand_models, make_config, mode_root, pair_slug, parse_pairs)


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

    def __getattr__(self, name):
        return getattr(self.stream, name)


def resolve_pairs(tokens):
    out = []
    for t in tokens:
        out += list(SCREEN_PAIRS) if t.strip().lower() == 'screen' else [t]
    return parse_pairs(out)


def main():
    p = argparse.ArgumentParser(description=f"Train LEAN_v2 presets on selected pairs ({BAND_LABEL}).")
    p.add_argument('--pairs', nargs='+', metavar='PAIR', default=None,
                   help="pairs, 'all' or 'screen' (default: the mode's pairs)")
    p.add_argument('--models', nargs='+', default=['ladder'],
                   help=f"presets and/or groups: {', '.join(MODEL_GROUPS)} (default ladder)")
    p.add_argument('--mode', choices=list(MODE_OVERRIDES), default='screen')
    p.add_argument('--dataset-root', default=DATASET_ROOT_DEFAULT)
    p.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--no-log-file', action='store_true')
    p.add_argument('--list-pairs', action='store_true',
                   help="print the normalized pair names and exit")
    p.add_argument('--pending', action='store_true',
                   help="with --list-pairs: only pairs some model has not finished")
    p.add_argument('--list-models', action='store_true',
                   help="print the expanded model names and exit")
    args = p.parse_args()

    try:
        models = expand_models(args.models)
    except ValueError as e:
        p.error(str(e))
    if args.list_models:
        print(' '.join(models))
        return 0

    pair_tokens = args.pairs or list(MODE_OVERRIDES[args.mode]['pairs'])
    try:
        pairs = resolve_pairs(pair_tokens)
    except ValueError as e:
        p.error(str(e))
    root = mode_root(args.results_root, args.mode)

    if args.list_pairs:
        if args.pending and not args.overwrite:
            from src.utils import is_pair_done
            pairs = [(a, b) for a, b in pairs
                     if not all(is_pair_done(os.path.join(root, m, pair_slug(a, b)), m)
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

    from src.train import train_pair

    for m in models:
        make_config(m, args.mode, args.dataset_root, args.results_root).to_json(
            os.path.join(root, m, 'config.json'))

    failed = []
    t0 = time.time()
    total = len(pairs) * len(models)
    run_i = 0
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
    if failed:
        print(f"\nDone with failures: {failed}", flush=True)
        return 1
    print("\nDone.", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
