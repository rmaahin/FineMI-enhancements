import src.determinism  # noqa: F401 — must be first
# Run one joint pair, one model, on the current GPU (SPEC §10.1).
#
#   python scripts/run_pair.py --model {conformer_a,conformer_b} --pair-idx N \
#       --mode {smoke,sanity,full} --scratch $SCRATCH --repo-root <path>
#
# Run from ConformerEEG/ with it on PYTHONPATH:  export PYTHONPATH=$PWD

import argparse
import os
import sys

from src.config import PAIRS, make_config, write_config_if_missing
from src.train import train_pair

REPO_ROOT_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="Run one joint pair on the current GPU.")
    p.add_argument('--model', required=True, choices=['conformer_a', 'conformer_b'])
    p.add_argument('--pair-idx', required=True, type=int,
                   help=f"pair index in [0, {len(PAIRS) - 1}]")
    p.add_argument('--mode', choices=['smoke', 'sanity', 'full'], default='full')
    p.add_argument('--scratch', default=os.environ.get('SCRATCH'),
                   help="TACC $SCRATCH (default: $SCRATCH)")
    p.add_argument('--repo-root', default=REPO_ROOT_DEFAULT,
                   help="repo root (default: parent of scripts/)")
    args = p.parse_args()

    if not args.scratch:
        p.error("--scratch not given and $SCRATCH is not set")
    if not 0 <= args.pair_idx < len(PAIRS):
        p.error(f"--pair-idx must be in [0, {len(PAIRS) - 1}], got {args.pair_idx}")

    src.determinism.apply_all(seed=42)

    cfg = make_config(args.model, args.mode, args.scratch, args.repo_root)
    class_a, class_b = PAIRS[args.pair_idx]
    config_path = write_config_if_missing(cfg, args.repo_root)

    print(f"run_pair: model={cfg.model} mode={cfg._mode} pair_idx={args.pair_idx} "
          f"({cfg.joint_names[class_a]}_{cfg.joint_names[class_b]})", flush=True)
    print(f"  dataset_root={cfg.dataset_root}", flush=True)
    print(f"  results_root={cfg.results_root}", flush=True)
    print(f"  config={config_path}", flush=True)

    train_pair(class_a, class_b, cfg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
