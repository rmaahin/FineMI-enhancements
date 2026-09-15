# Paired statistics between architectures, across subjects (no training, no torch).
#
# Run from Conformer_0.5_3hz/ (submit_ls6.slurm runs it after the comparison table):
#
#   python -m scripts.paired_stats                  # every pair all models completed
#   python -m scripts.paired_stats --pairs all --mode full
#
# Contrasts: Conformer A - FINE, Conformer B - FINE, Conformer A - Conformer B
# (only those whose models are selected), for every window and the window average:
#   scope=pair    one test per pair (18 subjects)
#   scope=pooled  each subject's accuracy averaged over all complete pairs first (18 subjects)
# Tests: paired t-test and Wilcoxon signed-rank; Cohen's dz; subjects better/worse/tied.
# p_holm = Holm-corrected paired t-test p within the scope (all contrasts x windows x pairs).
#
# Writes results/<mode>/comparison/paired_stats.csv and paired_stats.md.

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.compare import AVG_COL, discover_pairs
from src.config import (MODELS, MODEL_LABELS, MODE_OVERRIDES, RESULTS_ROOT_DEFAULT,
                        mode_root, pair_slug, parse_pairs)
from src.utils import is_pair_done

CONTRASTS = [('conformer_a', 'fine'), ('conformer_b', 'fine'), ('conformer_a', 'conformer_b')]
SHORT = {'fine': 'FINE', 'conformer_a': 'A', 'conformer_b': 'B'}


def read_wide(pair_dir):
    """subjects x [windows..., avg] with str column names."""
    wide = pd.read_csv(os.path.join(pair_dir, 'wide_subject_x_window.csv'), index_col=0)
    wide.columns = [str(c) for c in wide.columns]
    wide[AVG_COL] = wide.mean(axis=1)
    return wide


def holm(pvals):
    """Holm step-down adjusted p-values (NaNs are left out and stay NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out


def paired_test(x, y):
    d = (x - y).dropna()
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else np.nan
    t, p_t = stats.ttest_rel(x.loc[d.index], y.loc[d.index]) if n > 1 else (np.nan, np.nan)
    try:
        p_w = stats.wilcoxon(d).pvalue
    except ValueError:   # all differences zero, or too few samples
        p_w = np.nan
    return dict(n_subjects=n, mean_x=x.loc[d.index].mean(), mean_y=y.loc[d.index].mean(),
                diff=d.mean(), sd_diff=sd, t=t, p_ttest=p_t, p_wilcoxon=p_w,
                cohen_dz=d.mean() / sd if sd and sd > 0 else np.nan,
                n_x_better=int((d > 1e-9).sum()), n_y_better=int((d < -1e-9).sum()),
                n_ties=int((d.abs() <= 1e-9).sum()))


def fmt_p(p):
    if pd.isna(p):
        return '-'
    return '<.001' if p < 0.001 else f"{p:.3f}".lstrip('0')


def main():
    ap = argparse.ArgumentParser(description="Paired tests between FINE / Conformer A / B.")
    ap.add_argument('--pairs', nargs='+', metavar='PAIR', default=None,
                    help="pairs to include, or all (default: every pair with completed results)")
    ap.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    ap.add_argument('--mode', choices=list(MODE_OVERRIDES), default='full')
    ap.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    args = ap.parse_args()

    root = mode_root(args.results_root, args.mode)
    models = list(dict.fromkeys(args.models))
    contrasts = [(x, y) for x, y in CONTRASTS if x in models and y in models]
    if not contrasts:
        ap.error("need at least two of fine / conformer_a / conformer_b in --models")
    try:
        pairs = parse_pairs(args.pairs) if args.pairs else discover_pairs(root, models)
    except ValueError as e:
        ap.error(str(e))

    slugs = [pair_slug(a, b) for a, b in pairs]
    complete = [s for s in slugs if all(is_pair_done(os.path.join(root, m, s)) for m in models)]
    skipped = [s for s in slugs if s not in complete]
    if skipped:
        print(f"paired_stats: skipping pairs not complete for all models: {', '.join(skipped)}")
    if not complete:
        print(f"paired_stats: no pair is complete for all of {models} under {root}")
        return 1

    wides = {(s, m): read_wide(os.path.join(root, m, s)) for s in complete for m in models}
    columns = list(wides[(complete[0], models[0])].columns)   # windows..., avg

    rows = []
    for s in complete:
        for x, y in contrasts:
            for c in columns:
                rows.append(dict(scope='pair', pair=s, contrast=f"{SHORT[x]} - {SHORT[y]}",
                                 window=c, **paired_test(wides[(s, x)][c], wides[(s, y)][c])))
    pooled_w = {m: pd.concat([wides[(s, m)] for s in complete]).groupby(level=0).mean()
                for m in models}
    for x, y in contrasts:
        for c in columns:
            rows.append(dict(scope='pooled', pair=f"ALL({len(complete)})",
                             contrast=f"{SHORT[x]} - {SHORT[y]}", window=c,
                             **paired_test(pooled_w[x][c], pooled_w[y][c])))

    df = pd.DataFrame(rows)
    df['p_holm'] = np.nan
    for scope in ('pair', 'pooled'):
        mask = df['scope'] == scope
        df.loc[mask, 'p_holm'] = holm(df.loc[mask, 'p_ttest'])

    out_dir = os.path.join(root, 'comparison')
    os.makedirs(out_dir, exist_ok=True)
    df.round(6).to_csv(os.path.join(out_dir, 'paired_stats.csv'), index=False)

    # ---- Markdown summary --------------------------------------------------
    win_labels = [f"{c} ms" if c != AVG_COL else 'Avg' for c in columns]
    lines = [
        f"# Paired tests between architectures ({args.mode} mode, {len(complete)} pairs)",
        "",
        "Differences in percentage points (first minus second architecture), across 18 subjects. "
        "Paired t-test p, Holm-corrected within scope; W = subjects better/worse.",
        "",
        f"## Pooled over {len(complete)} pairs",
        "",
        "Each subject's accuracy averaged over all pairs first.",
        "",
        "| Contrast | " + " | ".join(win_labels) + " |",
        "|" + "|".join(['---'] + ['---:'] * len(columns)) + "|",
    ]
    pooled = df[df['scope'] == 'pooled'].set_index(['contrast', 'window'])
    for x, y in contrasts:
        name = f"{SHORT[x]} - {SHORT[y]}"
        cells = []
        for c in columns:
            r = pooled.loc[(name, c)]
            txt = (f"{r['diff']:+.2f} (p={fmt_p(r['p_ttest'])}, holm={fmt_p(r['p_holm'])}, "
                   f"W {r['n_x_better']}-{r['n_y_better']})")
            cells.append(f"**{txt}**" if r['p_holm'] < 0.05 else txt)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    per_pair = df[df['scope'] == 'pair']
    lines += [
        "",
        f"## Per pair: how many of {len(complete)} pairs favour each side",
        "",
        "Counts of pairs where the difference is positive / negative; in brackets, "
        "how many of those are significant after Holm correction over all per-pair tests.",
        "",
        "| Contrast | " + " | ".join(win_labels) + " |",
        "|" + "|".join(['---'] + ['---:'] * len(columns)) + "|",
    ]
    for x, y in contrasts:
        name = f"{SHORT[x]} - {SHORT[y]}"
        cells = []
        for c in columns:
            sub = per_pair[(per_pair['contrast'] == name) & (per_pair['window'] == c)]
            pos, neg = sub[sub['diff'] > 0], sub[sub['diff'] < 0]
            cells.append(f"{len(pos)} [{int((pos['p_holm'] < 0.05).sum())}] / "
                         f"{len(neg)} [{int((neg['p_holm'] < 0.05).sum())}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Labels: " + ", ".join(f"{SHORT[m]} = {MODEL_LABELS[m]}" for m in models)]
    if skipped:
        lines += ["", "Excluded (not complete for all models): " + ", ".join(skipped)]

    with open(os.path.join(out_dir, 'paired_stats.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"paired_stats written to {out_dir}")
    print(pooled[['diff', 'p_ttest', 'p_holm', 'n_x_better', 'n_y_better']]
          .round(4).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
