import src.determinism  # noqa: F401 — must be first
# Run pair indices [start, end) sequentially, one model, on the current GPU
# (SPEC §10.2). This is what submit_ls6.slurm launches once per GPU.
#
#   python scripts/run_pairs_range.py --model {conformer_a,conformer_b} \
#       --start N --end M --mode {smoke,sanity,full} \
#       --scratch $SCRATCH --repo-root <path>
#
# Pairs whose results.json already exists are skipped (resume). A pair that
# raises is logged and the rest of the range still runs; the exit code is 1 if
# any pair failed. Run from ConformerEEG/ with it on PYTHONPATH:
#   export PYTHONPATH=$PWD

import argparse
import os
import sys
import traceback

from src.config import PAIRS, make_config, write_config_if_missing
from src.train import train_pair

REPO_ROOT_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="Run a range of joint pairs on the current GPU.")
    p.add_argument('--model', required=True, choices=['conformer_a', 'conformer_b'])
    p.add_argument('--start', required=True, type=int, help="first pair index (inclusive)")
    p.add_argument('--end', required=True, type=int, help="last pair index (exclusive)")
    p.add_argument('--mode', choices=['smoke', 'sanity', 'full'], default='full')
    p.add_argument('--scratch', default=os.environ.get('SCRATCH'),
                   help="TACC $SCRATCH (default: $SCRATCH)")
    p.add_argument('--repo-root', default=REPO_ROOT_DEFAULT,
                   help="repo root (default: parent of scripts/)")
    args = p.parse_args()

    if not args.scratch:
        p.error("--scratch not given and $SCRATCH is not set")
    if not 0 <= args.start <= args.end <= len(PAIRS):
        p.error(f"need 0 <= --start <= --end <= {len(PAIRS)}, got {args.start}, {args.end}")

    src.determinism.apply_all(seed=42)

    cfg = make_config(args.model, args.mode, args.scratch, args.repo_root)
    config_path = write_config_if_missing(cfg, args.repo_root)

    pair_indices = list(range(args.start, args.end))
    if cfg._max_pairs_per_range is not None:
        pair_indices = pair_indices[:cfg._max_pairs_per_range]

    print(f"run_pairs_range: model={cfg.model} mode={cfg._mode} "
          f"range=[{args.start}, {args.end}) -> pairs {pair_indices}", flush=True)
    print(f"  dataset_root={cfg.dataset_root}", flush=True)
    print(f"  results_root={cfg.results_root}", flush=True)
    print(f"  config={config_path}", flush=True)

    failed = []
    for pair_idx in pair_indices:
        class_a, class_b = PAIRS[pair_idx]
        print(f"\n##### pair_idx={pair_idx} "
              f"({cfg.joint_names[class_a]}_{cfg.joint_names[class_b]}) #####", flush=True)
        try:
            train_pair(class_a, class_b, cfg)
        except Exception:
            traceback.print_exc()
            sys.stderr.flush()
            print(f"FAILED pair_idx={pair_idx}", flush=True)
            failed.append(pair_idx)

    if failed:
        print(f"\nDone with failures: pair indices {failed}", flush=True)
        return 1
    print(f"\nDone: pairs {pair_indices}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
