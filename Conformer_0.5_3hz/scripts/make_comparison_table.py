# Rebuild the architecture x pair comparison table from saved results (no training,
# no torch). run_selected_pairs already does this at the end; use this to refresh
# the table after partial runs or to compare a different subset.
#
# Run from Conformer_0.5_3hz/:
#
#   python -m scripts.make_comparison_table                      # every completed pair
#   python -m scripts.make_comparison_table --pairs WAA_SAA HOC_WFE EPS_SPS
#   python -m scripts.make_comparison_table --mode sanity
#
# Writes results/<mode>/comparison/ (see src/compare.py).

import argparse
import os
import sys

from src.compare import build_comparison, discover_pairs
from src.config import MODELS, MODE_OVERRIDES, RESULTS_ROOT_DEFAULT, mode_root, parse_pair


def main():
    p = argparse.ArgumentParser(description="Build the FINE / Conformer A / B comparison table.")
    p.add_argument('--pairs', nargs='+', metavar='PAIR', default=None,
                   help="pairs to include (default: every pair with completed results)")
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--mode', choices=list(MODE_OVERRIDES), default='full')
    p.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    args = p.parse_args()

    root = mode_root(args.results_root, args.mode)
    if not os.path.isdir(root):
        p.error(f"no results for mode {args.mode!r}: {root} does not exist")

    models = list(dict.fromkeys(args.models))
    if args.pairs:
        pairs = []
        for token in args.pairs:
            try:
                pair = parse_pair(token)
            except ValueError as e:
                p.error(str(e))
            if pair not in pairs:
                pairs.append(pair)
    else:
        pairs = discover_pairs(root, models)

    table = build_comparison(root, pairs, models=models)
    return 0 if table is not None else 1


if __name__ == '__main__':
    sys.exit(main())
