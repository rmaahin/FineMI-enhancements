"""
Compare the SAME arm across two results roots, pair by pair and subject by subject.

Typical use - decimation A/B on the CWT arm:

    python compare_runs.py --a results_decim1 --b results --mode cwt \
        --label-a "vanilla 250 Hz" --label-b "decimated 50 Hz"

For each pair present in both roots it reports mean accuracy per run, the mean
per-subject delta (B - A), how many subjects improved, Cohen's dz and a paired
Wilcoxon p-value, then an overall line pooling all (pair, subject) cells. Written to
<b>/compare_<mode>_<basename(a)>_vs_<basename(b)>.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load_wide(root, mode):
    """{pair: Series(subject -> accuracy)} for every finished pair under <root>/<mode>/."""
    out = {}
    d = os.path.join(root, mode)
    if not os.path.isdir(d):
        return out
    for slug in sorted(os.listdir(d)):
        p = os.path.join(d, slug, "wide_subject_x_window.csv")
        if os.path.isfile(p):
            out[slug.replace("_", "/", 1)] = pd.read_csv(p, index_col=0).iloc[:, 0]
    return out


def paired(a, b):
    d = (b - a).values
    sd = np.std(d, ddof=1) if len(d) > 1 else np.nan
    try:
        p = float(wilcoxon(b.values, a.values).pvalue)
    except ValueError:          # all-zero differences
        p = 1.0
    return dict(n=len(d), delta=float(d.mean()),
                n_better=int((d > 0).sum()), n_worse=int((d < 0).sum()),
                cohen_dz=float(d.mean() / sd) if sd and sd > 0 else np.nan, p=p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', required=True, help='results root A (baseline)')
    ap.add_argument('--b', required=True, help='results root B (treatment)')
    ap.add_argument('--mode', default='cwt', choices=['raw', 'cwt'])
    ap.add_argument('--label-a', default=None)
    ap.add_argument('--label-b', default=None)
    args = ap.parse_args()
    la = args.label_a or os.path.basename(os.path.normpath(args.a))
    lb = args.label_b or os.path.basename(os.path.normpath(args.b))

    A, B = load_wide(args.a, args.mode), load_wide(args.b, args.mode)
    common = sorted(set(A) & set(B))
    print(f"{args.mode}: {len(A)} pairs in A ({la}), {len(B)} in B ({lb}), "
          f"{len(common)} in common")
    if not common:
        return
    for missing, name in ((set(A) - set(B), lb), (set(B) - set(A), la)):
        if missing:
            print(f"  not yet in {name}: {', '.join(sorted(missing))}")

    rows, all_a, all_b = [], [], []
    for pair in common:
        sa, sb = A[pair], B[pair]
        subs = sa.index.intersection(sb.index)
        sa, sb = sa.loc[subs], sb.loc[subs]
        rec = dict(pair=pair, acc_a=round(sa.mean(), 2), sd_a=round(sa.std(ddof=1), 2),
                   acc_b=round(sb.mean(), 2), sd_b=round(sb.std(ddof=1), 2))
        rec.update(paired(sa, sb))
        rows.append(rec)
        all_a.append(sa)
        all_b.append(sb)

    tab = pd.DataFrame(rows)
    print(f"\n=== per pair: accuracy % (mean over subjects), delta = {lb} - {la} ===")
    print(tab.round(3).to_string(index=False))

    pooled = paired(pd.concat(all_a), pd.concat(all_b))
    print(f"\n=== pooled over {pooled['n']} (pair, subject) cells ===")
    print(f"{la}: {pd.concat(all_a).mean():.2f}%   {lb}: {pd.concat(all_b).mean():.2f}%   "
          f"delta {pooled['delta']:+.2f} pp | better {pooled['n_better']} / "
          f"worse {pooled['n_worse']} | dz={pooled['cohen_dz']:.3f} | "
          f"Wilcoxon p={pooled['p']:.4g}")
    if len(tab) >= 2:
        print(f"pair-level: {lb} ahead on {int((tab['delta'] > 0).sum())}/{len(tab)} pairs, "
              f"mean delta {tab['delta'].mean():+.2f} pp")

    out = os.path.join(args.b, f"compare_{args.mode}_"
                       f"{os.path.basename(os.path.normpath(args.a))}_vs_"
                       f"{os.path.basename(os.path.normpath(args.b))}.csv")
    tab.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
