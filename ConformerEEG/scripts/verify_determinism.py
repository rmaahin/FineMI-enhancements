import src.determinism  # noqa: F401 — must be first
# Reproducibility check (SPEC §10.3): run WAA_SAA (classes 2 vs 6) twice, each in
# a fresh Python process (the notebook's "restart the runtime" procedure), into
# <results_root>/<mode>/_determinism/run{1,2}/, then compare the two
# wide_subject_x_window.csv files.
#
#   python scripts/verify_determinism.py --model {conformer_a,conformer_b} \
#       --mode {smoke,sanity,full} --scratch $SCRATCH --repo-root <path>
#
# Prints OK and exits 0 if max |diff| == 0.0; prints the max abs diff and exits 1
# otherwise. Run from ConformerEEG/ with it on PYTHONPATH:  export PYTHONPATH=$PWD

import argparse
import os
import shutil
import subprocess
import sys

from src.config import PAIRS, make_config

REPO_ROOT_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK_PAIR = (2, 6)   # WAA vs SAA


def run_once(cfg, run_dir):
    """One training pass of CHECK_PAIR with results written under run_dir."""
    from src.train import train_pair
    cfg.results_root = run_dir
    train_pair(*CHECK_PAIR, cfg)


def compare(csv_a, csv_b):
    """Return (max_abs_diff, problem) for two wide CSVs; problem is a str or None."""
    import numpy as np
    import pandas as pd
    a = pd.read_csv(csv_a, index_col=0)
    b = pd.read_csv(csv_b, index_col=0)
    if not a.index.equals(b.index) or not a.columns.equals(b.columns):
        return float('nan'), (f"shape/labels differ: {a.shape} {list(a.columns)} vs "
                              f"{b.shape} {list(b.columns)}")
    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    if (np.isnan(av) != np.isnan(bv)).any():
        return float('nan'), "missing cells differ between runs"
    diff = np.abs(av - bv)
    max_diff = float(np.nanmax(diff)) if (~np.isnan(diff)).any() else 0.0
    return max_diff, None


def main():
    p = argparse.ArgumentParser(description="Run WAA_SAA twice and check bit-identical results.")
    p.add_argument('--model', required=True, choices=['conformer_a', 'conformer_b'])
    p.add_argument('--mode', choices=['smoke', 'sanity', 'full'], default='smoke')
    p.add_argument('--scratch', default=os.environ.get('SCRATCH'),
                   help="TACC $SCRATCH (default: $SCRATCH)")
    p.add_argument('--repo-root', default=REPO_ROOT_DEFAULT,
                   help="repo root (default: parent of scripts/)")
    p.add_argument('--run-dir', default=None, help=argparse.SUPPRESS)   # internal: child pass
    args = p.parse_args()

    if not args.scratch:
        p.error("--scratch not given and $SCRATCH is not set")

    src.determinism.apply_all(seed=42)
    cfg = make_config(args.model, args.mode, args.scratch, args.repo_root)

    # Child process: do one pass and exit
    if args.run_dir is not None:
        run_once(cfg, args.run_dir)
        return 0

    pair_idx = PAIRS.index(CHECK_PAIR)
    slug = f"{cfg.joint_names[CHECK_PAIR[0]]}_{cfg.joint_names[CHECK_PAIR[1]]}"
    base = os.path.join(cfg.results_root, '_determinism')
    run_dirs = [os.path.join(base, 'run1'), os.path.join(base, 'run2')]

    print(f"verify_determinism: model={cfg.model} mode={cfg._mode} "
          f"pair_idx={pair_idx} ({slug})", flush=True)
    print(f"  output: {base}", flush=True)

    # Child needs the repo root importable regardless of how this parent was launched
    env = dict(os.environ)
    env['PYTHONPATH'] = args.repo_root + (os.pathsep + env['PYTHONPATH']
                                          if env.get('PYTHONPATH') else '')

    for i, run_dir in enumerate(run_dirs, start=1):
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir)   # stale output from a previous check would be skipped
        print(f"\n===== run {i}/2 -> {run_dir} =====", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__),
               '--model', args.model, '--mode', args.mode,
               '--scratch', args.scratch, '--repo-root', args.repo_root,
               '--run-dir', run_dir]
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            print(f"FAILED: run {i} exited with code {rc}", flush=True)
            return 1

    csvs = [os.path.join(d, slug, 'wide_subject_x_window.csv') for d in run_dirs]
    max_diff, problem = compare(*csvs)
    if problem is None and max_diff == 0.0:
        print("\nmax |diff| = 0.0 pp", flush=True)
        print("OK", flush=True)
        return 0

    if problem:
        print(f"\nMISMATCH: {problem}", flush=True)
    print(f"max |diff| = {max_diff} pp", flush=True)
    print("Check that both runs report the same gpu= name (results.json).", flush=True)
    return 1


if __name__ == '__main__':
    sys.exit(main())
