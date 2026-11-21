# Verify a finished run's outputs (no training, no torch). run_vast.sh runs it after
# training; a nonzero exit marks the run as failed.
#
#   python -m scripts.check_results --mode screen --models ladder
#   python -m scripts.check_results --mode full --pairs all --models lean_v2_r7
#
# For every pair and model: results.json exists with this pipeline's stamp and the
# preset's current hash, the run's mode / folds / epochs / windows, and exactly the
# parameter count src/param_budget.py predicts at every window; the CSVs have every
# window, the expected number of subjects, no missing cells, accuracies in [0, 100] and
# subjects x windows x folds rows.

import argparse
import glob
import os
import sys

import pandas as pd

from src.config import (BAND_TAG, DATASET_ROOT_DEFAULT, MODE_OVERRIDES, RESULTS_ROOT_DEFAULT,
                        expand_models, make_config, mode_root, pair_slug)
from src.param_budget import expected_params
from src.utils import is_pair_done, read_results_json
from scripts.run_selected_pairs import resolve_pairs


def main():
    ap = argparse.ArgumentParser(description="Check the outputs of a smoke, screen or full run.")
    ap.add_argument('--mode', choices=list(MODE_OVERRIDES), required=True)
    ap.add_argument('--pairs', nargs='+', default=None)
    ap.add_argument('--models', nargs='+', default=['ladder'])
    ap.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    ap.add_argument('--dataset-root', default=DATASET_ROOT_DEFAULT)
    ap.add_argument('--expect-subjects', type=int, default=None)
    args = ap.parse_args()

    try:
        models = expand_models(args.models)
        pairs = resolve_pairs(args.pairs or list(MODE_OVERRIDES[args.mode]['pairs']))
    except ValueError as e:
        ap.error(str(e))
    root = mode_root(args.results_root, args.mode)

    n_subj = args.expect_subjects
    if n_subj is None:
        n_subj = MODE_OVERRIDES[args.mode]['max_subjects']
    if n_subj is None:
        files = glob.glob(os.path.join(args.dataset_root, f"subject*_eeg_epochs_{BAND_TAG}_*.npz"))
        n_subj = len(files)
        if n_subj == 0:
            ap.error(f"no subject files in {args.dataset_root}; pass --expect-subjects")

    problems = []
    n_ok = 0
    for m in models:
        cfg = make_config(m, args.mode, args.dataset_root, args.results_root)
        windows = [int(w) for w in cfg.test_time_windows_ms]
        want_params = {str(w): expected_params(cfg, s)
                       for w, s in zip(windows, cfg.window_samples())}
        for a, b in pairs:
            slug = pair_slug(a, b)
            d = os.path.join(root, m, slug)
            tag = f"{slug} [{m}]"
            if not is_pair_done(d, m):
                meta = read_results_json(d)
                why = "missing" if meta is None else (
                    f"stamp pipeline_version={meta.get('pipeline_version')} "
                    f"preset_hash={meta.get('preset_hash')} (current {cfg.preset_hash})")
                problems.append(f"{tag}: not done ({why})")
                continue
            meta = read_results_json(d)
            for key, want in (('mode', args.mode), ('model', m), ('band', BAND_TAG),
                              ('n_folds', cfg.n_folds), ('n_epochs', cfg.n_epochs),
                              ('sampling_rate', cfg.sampling_rate),
                              ('n_params', want_params),
                              ('window_samples', cfg.window_samples())):
                if meta.get(key) != want:
                    problems.append(f"{tag}: results.json {key}={meta.get(key)!r}, expected {want!r}")
            if os.path.basename(os.path.normpath(meta.get('dataset_root', ''))) != \
                    os.path.basename(os.path.normpath(args.dataset_root)):
                problems.append(f"{tag}: trained on {meta.get('dataset_root')!r}, "
                                f"not on {args.dataset_root!r}")
            try:
                wide = pd.read_csv(os.path.join(d, 'wide_subject_x_window.csv'), index_col=0)
                folds = pd.read_csv(os.path.join(d, 'per_fold.csv'))
            except (OSError, ValueError) as e:
                problems.append(f"{tag}: cannot read CSVs ({e})")
                continue
            got_windows = [int(float(c)) for c in wide.columns]
            if got_windows != windows:
                problems.append(f"{tag}: windows {got_windows}, expected {windows}")
            if len(wide) != n_subj:
                problems.append(f"{tag}: {len(wide)} subjects, expected {n_subj}")
            if wide.isna().any().any():
                problems.append(f"{tag}: missing subject/window cells")
            elif ((wide < 0) | (wide > 100)).any().any():
                problems.append(f"{tag}: accuracy outside [0, 100]")
            want_rows = n_subj * len(windows) * cfg.n_folds
            if len(folds) != want_rows:
                problems.append(f"{tag}: per_fold.csv has {len(folds)} rows, expected {want_rows}")
            n_ok += 1

    total = len(pairs) * len(models)
    print(f"check_results ({args.mode}): {n_ok}/{total} pair runs present, "
          f"expecting {n_subj} subjects each", flush=True)
    if problems:
        for pr in problems:
            print(f"  PROBLEM: {pr}", flush=True)
        print(f"CHECK FAILED: {len(problems)} problem(s)", flush=True)
        return 1
    print("CHECK PASSED: every run is complete", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
