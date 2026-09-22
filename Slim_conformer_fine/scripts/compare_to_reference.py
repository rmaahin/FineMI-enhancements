# Compare Slim Conformer B with Conformer A and Conformer B from Conformer_decimated_CWT
# (no training, no torch). Same data, decimation, folds, seeds and training, so every
# subject pairs up across the three models.
#
#   python -m scripts.compare_to_reference                      # 3 pilot pairs, full mode
#   python -m scripts.compare_to_reference --pairs all
#   python -m scripts.compare_to_reference --mode smoke --reference-mode smoke
#
# For each contrast (Slim - A, Slim - B) and each window plus the window average:
#   scope=pair    paired t-test across subjects, one per pair
#   scope=pooled  each subject's accuracy averaged over the selected pairs first
# Reports the mean difference, its 95% confidence interval, t-test p, Holm-corrected p
# (within each scope) and how many subjects are better / worse.
#
# "On par" is judged as non-inferiority on the pooled window average: Slim is on par
# with a reference model if the lower end of the 95% interval for (Slim - reference)
# is above -MARGIN points (default 1.0, change with --margin). With only a few pairs
# the interval is wide, so "not shown" usually means "needs more pairs", not "worse".
#
# Writes results/<mode>/comparison/vs_reference.csv and vs_reference.md.

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (DEFAULT_PAIRS, EXPECTED_PARAMS, MODE_OVERRIDES, MODELS,
                        REFERENCE_LABELS, REFERENCE_MODELS, REFERENCE_PARAMS,
                        REFERENCE_ROOT_DEFAULT, RESULTS_ROOT_DEFAULT, mode_root, pair_slug,
                        parse_pairs)

SLIM = MODELS[0]
AVG = 'avg'


def read_wide(pair_dir):
    """subjects x [windows..., avg] with str column names, or None if missing."""
    path = os.path.join(pair_dir, 'wide_subject_x_window.csv')
    if not os.path.isfile(path):
        return None
    wide = pd.read_csv(path, index_col=0)
    wide.columns = [str(int(float(c))) for c in wide.columns]
    wide[AVG] = wide.mean(axis=1)
    return wide


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out


def paired(x, y):
    common = x.index.intersection(y.index)
    d = (x.loc[common] - y.loc[common]).dropna()
    n = len(d)
    res = dict(n_subjects=n, mean_slim=x.loc[d.index].mean(), mean_ref=y.loc[d.index].mean(),
               diff=d.mean(), ci_lo=np.nan, ci_hi=np.nan, p_ttest=np.nan,
               n_slim_better=int((d > 1e-9).sum()), n_ref_better=int((d < -1e-9).sum()))
    if n > 1:
        sd = d.std(ddof=1)
        if sd > 0:
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            res.update(ci_lo=d.mean() - half, ci_hi=d.mean() + half,
                       p_ttest=float(stats.ttest_rel(x.loc[d.index], y.loc[d.index]).pvalue))
    return res


def fmt_p(p):
    if pd.isna(p):
        return '-'
    return '<.001' if p < 0.001 else f"{p:.3f}".lstrip('0')


def main():
    ap = argparse.ArgumentParser(description="Compare Slim Conformer B with Conformer A and B.")
    ap.add_argument('--pairs', nargs='+', default=list(DEFAULT_PAIRS))
    ap.add_argument('--mode', choices=list(MODE_OVERRIDES), default='full',
                    help="which Slim results to read (results/<mode>/)")
    ap.add_argument('--reference-mode', choices=list(MODE_OVERRIDES), default=None,
                    help="which reference results to read (default: same as --mode)")
    ap.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    ap.add_argument('--reference-root', default=REFERENCE_ROOT_DEFAULT)
    ap.add_argument('--margin', type=float, default=1.0,
                    help="non-inferiority margin in points for the 'on par' verdict (default 1.0)")
    args = ap.parse_args()

    try:
        pairs = parse_pairs(args.pairs)
    except ValueError as e:
        ap.error(str(e))
    ref_mode = args.reference_mode or args.mode
    root = mode_root(args.results_root, args.mode)
    ref_root = mode_root(args.reference_root, ref_mode)

    slugs = [pair_slug(a, b) for a, b in pairs]
    wides, skipped = {}, []
    for s in slugs:
        w = {SLIM: read_wide(os.path.join(root, SLIM, s))}
        for m in REFERENCE_MODELS:
            w[m] = read_wide(os.path.join(ref_root, m, s))
        missing = [m for m, v in w.items() if v is None]
        if missing:
            skipped.append(f"{s} ({', '.join(missing)})")
        else:
            wides[s] = w
    if skipped:
        print(f"compare_to_reference: skipping pairs with missing results: {'; '.join(skipped)}")
    if not wides:
        print(f"compare_to_reference: nothing to compare (slim: {root}, reference: {ref_root})")
        return 1
    complete = list(wides)
    columns = list(wides[complete[0]][SLIM].columns)

    rows = []
    for s in complete:
        for m in REFERENCE_MODELS:
            for c in columns:
                rows.append(dict(scope='pair', pair=s, contrast=f"Slim - {REFERENCE_LABELS[m]}",
                                 window=c, **paired(wides[s][SLIM][c], wides[s][m][c])))
    pooled = {m: pd.concat([wides[s][m] for s in complete]).groupby(level=0).mean()
              for m in [SLIM] + list(REFERENCE_MODELS)}
    for m in REFERENCE_MODELS:
        for c in columns:
            rows.append(dict(scope='pooled', pair=f"ALL({len(complete)})",
                             contrast=f"Slim - {REFERENCE_LABELS[m]}", window=c,
                             **paired(pooled[SLIM][c], pooled[m][c])))
    df = pd.DataFrame(rows)
    df['p_holm'] = np.nan
    for scope in ('pair', 'pooled'):
        mask = df['scope'] == scope
        df.loc[mask, 'p_holm'] = holm(df.loc[mask, 'p_ttest'])

    out_dir = os.path.join(root, 'comparison')
    os.makedirs(out_dir, exist_ok=True)
    df.round(6).to_csv(os.path.join(out_dir, 'vs_reference.csv'), index=False)

    # ---- Markdown -------------------------------------------------------------
    win_labels = [f"{c} ms" if c != AVG else 'Avg' for c in columns]
    n_subj = int(df['n_subjects'].max())
    lines = [
        f"# Slim Conformer B vs Conformer A and B ({args.mode} mode, {len(complete)} pairs)",
        "",
        f"Pairs: {', '.join(s.replace('_', ' vs ') for s in complete)}. "
        f"Reference results: `{ref_root}`.",
        f"Paired over {n_subj} subjects. Differences in points, Slim minus reference. "
        f"95% CI, paired t-test p and Holm-corrected p within each scope; "
        f"W = subjects Slim better / reference better.",
        "",
        "## Parameters",
        "",
        "| Model | " + " | ".join(win_labels[:-1]) + " |",
        "|" + "|".join(['---'] + ['---:'] * (len(columns) - 1)) + "|",
        f"| Slim Conformer B | " + " | ".join(f"{EXPECTED_PARAMS[SLIM]:,}" for _ in columns[:-1]) + " |",
    ]
    for m in REFERENCE_MODELS:
        lines.append(f"| {REFERENCE_LABELS[m]} | " + " | ".join(
            f"{REFERENCE_PARAMS[m].get(int(c), float('nan')):,}" for c in columns[:-1]) + " |")

    lines += ["", "## Mean accuracy, pooled over the selected pairs", "",
              "| Model | " + " | ".join(win_labels) + " |",
              "|" + "|".join(['---'] + ['---:'] * len(columns)) + "|"]
    for m, label in [(SLIM, 'Slim Conformer B')] + [(m, REFERENCE_LABELS[m]) for m in REFERENCE_MODELS]:
        lines.append(f"| {label} | " + " | ".join(f"{pooled[m][c].mean():.2f}" for c in columns) + " |")

    pooled_df = df[df['scope'] == 'pooled'].set_index(['contrast', 'window'])
    lines += ["", f"## Pooled over {len(complete)} pairs", "",
              "| Contrast | " + " | ".join(win_labels) + " |",
              "|" + "|".join(['---'] + ['---:'] * len(columns)) + "|"]
    for m in REFERENCE_MODELS:
        name = f"Slim - {REFERENCE_LABELS[m]}"
        cells = []
        for c in columns:
            r = pooled_df.loc[(name, c)]
            cells.append(f"{r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] "
                         f"(holm={fmt_p(r['p_holm'])}, W {r['n_slim_better']}-{r['n_ref_better']})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", "## Per pair, window average", "",
              "| Pair | Contrast | Diff | 95% CI | p | Holm p | W |",
              "|---|---|---:|---:|---:|---:|---:|"]
    per_pair = df[(df['scope'] == 'pair') & (df['window'] == AVG)]
    for _, r in per_pair.iterrows():
        lines.append(f"| {r['pair'].replace('_', ' vs ')} | {r['contrast']} | {r['diff']:+.2f} | "
                     f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {fmt_p(r['p_ttest'])} | "
                     f"{fmt_p(r['p_holm'])} | {r['n_slim_better']}-{r['n_ref_better']} |")

    lines += ["", f"## On par? (non-inferiority, margin {args.margin:g} points, window average)", ""]
    verdicts = []
    for m in REFERENCE_MODELS:
        r = pooled_df.loc[(f"Slim - {REFERENCE_LABELS[m]}", AVG)]
        if pd.isna(r['ci_lo']):
            v = "cannot tell (too few subjects for an interval)"
        elif r['ci_lo'] > -args.margin:
            v = f"ON PAR: the interval's lower end {r['ci_lo']:+.2f} is above -{args.margin:g}"
        elif r['ci_hi'] < -args.margin:
            v = f"WORSE: the whole interval is below -{args.margin:g} ({r['ci_hi']:+.2f} upper end)"
        else:
            v = (f"NOT SHOWN: the interval [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] crosses "
                 f"-{args.margin:g}; more pairs are needed to decide")
        line = f"- Slim vs {REFERENCE_LABELS[m]}: {v}."
        verdicts.append(line)
        lines.append(line)
    if args.mode != ref_mode or args.mode == 'smoke':
        lines += ["", "Note: smoke-mode numbers only prove the pipeline runs; they say nothing "
                      "about accuracy."]
    lines += ["", "Runs on a different GPU model (or the CPU) than the reference are not "
                  "bit-identical to it, but the comparison is still paired on the same subjects, "
                  "folds and seeds."]
    if skipped:
        lines += ["", "Skipped (missing results): " + "; ".join(skipped)]

    md_path = os.path.join(out_dir, 'vs_reference.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"compare_to_reference: written {md_path}")
    print("\nMean accuracy pooled over the selected pairs:")
    table = pd.DataFrame({label: [pooled[m][c].mean() for c in columns]
                          for m, label in [(SLIM, 'Slim B')] + [(m, REFERENCE_LABELS[m]) for m in REFERENCE_MODELS]},
                         index=win_labels).T
    print(table.round(2).to_string())
    print("\nPooled differences (Slim minus reference):")
    print(pooled_df[['diff', 'ci_lo', 'ci_hi', 'p_holm', 'n_slim_better', 'n_ref_better']]
          .round(3).to_string())
    print("\n" + "\n".join(verdicts))
    return 0


if __name__ == '__main__':
    sys.exit(main())
