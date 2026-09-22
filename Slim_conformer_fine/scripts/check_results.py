# Verify a finished run's outputs (no training, no torch). local_test.py and
# submit_ls6.slurm run it after training; a nonzero exit marks the run as failed.
#
#   python -m scripts.check_results --mode smoke --pairs WAA_SAA HOC_WFE EPS_SPS
#   python -m scripts.check_results --mode full --pairs all --dataset-root <data folder>
#
# For every pair: results.json exists with this pipeline's stamp, the run's mode /
# folds / epochs, and the expected 25,752 parameters; wide_subject_x_window.csv has
# every window and the expected number of subjects with no missing cells and
# accuracies in [0, 100]; per_fold.csv has subjects x windows x folds rows.

import argparse
import glob
import os
import sys

import pandas as pd

from src.config import (DATASET_ROOT_DEFAULT, DEFAULT_PAIRS, EXPECTED_PARAMS, MODELS,
                        MODE_OVERRIDES, RESULTS_ROOT_DEFAULT, make_config, mode_root,
                        pair_slug, parse_pairs)
from src.utils import is_pair_done, read_results_json


def main():
    ap = argparse.ArgumentParser(description="Check the outputs of a smoke or full run.")
    ap.add_argument('--mode', choices=list(MODE_OVERRIDES), required=True)
    ap.add_argument('--pairs', nargs='+', default=list(DEFAULT_PAIRS))
    ap.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    ap.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    ap.add_argument('--dataset-root', default=DATASET_ROOT_DEFAULT,
                    help="used to count subject files when the mode has no subject cap")
    ap.add_argument('--expect-subjects', type=int, default=None,
                    help="default: the mode's subject cap (smoke 2), else the number of "
                         "subject files in --dataset-root")
    args = ap.parse_args()

    try:
        pairs = parse_pairs(args.pairs)
    except ValueError as e:
        ap.error(str(e))
    models = list(dict.fromkeys(args.models))
    root = mode_root(args.results_root, args.mode)

    n_subj = args.expect_subjects
    if n_subj is None:
        n_subj = MODE_OVERRIDES[args.mode]['max_subjects']
    if n_subj is None:
        files = glob.glob(os.path.join(args.dataset_root, "subject*_eeg_epochs_0.5_3hz_*.npz"))
        n_subj = len(files)
        if n_subj == 0:
            ap.error(f"no subject files in {args.dataset_root}; pass --expect-subjects")

    problems = []
    n_ok = 0
    for a, b in pairs:
        slug = pair_slug(a, b)
        for m in models:
            cfg = make_config(m, args.mode, args.dataset_root, args.results_root)
            windows = [int(w) for w in cfg.test_time_windows_ms]
            d = os.path.join(root, m, slug)
            tag = f"{slug} [{m}]"
            if not is_pair_done(d):
                meta = read_results_json(d)
                why = "missing" if meta is None else (
                    f"stamp pipeline_version={meta.get('pipeline_version')} "
                    f"decim={meta.get('decim')}")
                problems.append(f"{tag}: not done ({why})")
                continue
            meta = read_results_json(d)
            for key, want in (('mode', args.mode), ('model', m), ('n_folds', cfg.n_folds),
                              ('n_epochs', cfg.n_epochs), ('sampling_rate', cfg.sampling_rate),
                              ('n_params', EXPECTED_PARAMS[m])):
                if meta.get(key) != want:
                    problems.append(f"{tag}: results.json {key}={meta.get(key)!r}, expected {want!r}")
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
