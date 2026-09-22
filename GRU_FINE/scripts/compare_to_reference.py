# Compare FINE + GRU and FINE + GRU + CWT with Conformer A / B / B + CWT from
# Conformer_decimated_CWT (no training, no torch). Same data, decimation, folds, seeds
# and training, so every subject pairs up across the five models.
#
# Run from GRU_FINE/ (submit_ls6.slurm runs it after the paired stats):
#
#   python -m scripts.compare_to_reference                      # every completed pair, full mode
#   python -m scripts.compare_to_reference --pairs WAA_SAA HOC_WFE EPS_SPS
#   python -m scripts.compare_to_reference --mode smoke --reference-mode smoke
#
# Contrasts (GRU model minus reference; only those whose GRU model is in --models):
#   GRU     - B        the main question: BiGRU vs transformer, same FINE front-end
#   GRU     - A
#   GRU+CWT - B+CWT    BiGRU vs transformer, same CWT front-end
#   GRU+CWT - B
# For each contrast, each window and the window average:
#   scope=pair    paired t-test across subjects, one per pair
#   scope=pooled  each subject's accuracy averaged over the contrast's pairs first
# Reports the mean difference, its 95% confidence interval, t-test p, Holm-corrected p
# (within each scope) and how many subjects are better / worse. The verdict on the
# pooled window average uses a non-inferiority margin (default 1.0 point, --margin).
#
# Writes results/<mode>/comparison/vs_reference.csv and vs_reference.md.

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.compare import discover_pairs
from src.config import (MODEL_LABELS, MODE_OVERRIDES, MODELS, REFERENCE_LABELS,
                        REFERENCE_ROOT_DEFAULT, RESULTS_ROOT_DEFAULT, mode_root, pair_slug,
                        parse_pairs)
from src.utils import is_pair_done

AVG = 'avg'
CONTRASTS = [('fine_gru', 'conformer_b'), ('fine_gru', 'conformer_a'),
             ('fine_gru_cwt', 'conformer_b_cwt'), ('fine_gru_cwt', 'conformer_b')]
SHORT = {'fine_gru': 'GRU', 'fine_gru_cwt': 'GRU+CWT',
         'conformer_a': 'A', 'conformer_b': 'B', 'conformer_b_cwt': 'B+CWT'}
LABELS = {**MODEL_LABELS, **REFERENCE_LABELS}


def read_wide(pair_dir):
    """subjects x [windows..., avg] with str column names, or None if not complete."""
    path = os.path.join(pair_dir, 'wide_subject_x_window.csv')
    if not is_pair_done(pair_dir) or not os.path.isfile(path):
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
    res = dict(n_subjects=n, mean_gru=x.loc[d.index].mean(), mean_ref=y.loc[d.index].mean(),
               diff=d.mean(), ci_lo=np.nan, ci_hi=np.nan, p_ttest=np.nan,
               n_gru_better=int((d > 1e-9).sum()), n_ref_better=int((d < -1e-9).sum()))
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


def verdict(r, margin):
    if pd.isna(r['ci_lo']):
        return "cannot tell (too few subjects for an interval)"
    if r['ci_lo'] > 0:
        return f"BETTER: the whole interval is above 0 ([{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}])"
    if r['ci_lo'] > -margin:
        return (f"ON PAR: the interval's lower end {r['ci_lo']:+.2f} is above -{margin:g} "
                f"(not worse by more than the margin)")
    if r['ci_hi'] < 0:
        return f"WORSE: the whole interval is below 0 ([{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}])"
    return (f"NOT SHOWN: the interval [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] crosses "
            f"-{margin:g}; more pairs are needed to decide")


def main():
    ap = argparse.ArgumentParser(
        description="Compare FINE + GRU (+ CWT) with Conformer A / B / B + CWT.")
    ap.add_argument('--pairs', nargs='+', metavar='PAIR', default=None,
                    help="pairs to include, or all (default: every pair with GRU results)")
    ap.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS),
                    help="GRU models to compare (default: both)")
    ap.add_argument('--mode', choices=list(MODE_OVERRIDES), default='full',
                    help="which GRU results to read (results/<mode>/)")
    ap.add_argument('--reference-mode', choices=list(MODE_OVERRIDES), default=None,
                    help="which reference results to read (default: same as --mode)")
    ap.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    ap.add_argument('--reference-root', default=REFERENCE_ROOT_DEFAULT)
    ap.add_argument('--margin', type=float, default=1.0,
                    help="non-inferiority margin in points for the verdict (default 1.0)")
    args = ap.parse_args()

    models = list(dict.fromkeys(args.models))
    ref_mode = args.reference_mode or args.mode
    root = mode_root(args.results_root, args.mode)
    ref_root = mode_root(args.reference_root, ref_mode)
    try:
        pairs = parse_pairs(args.pairs) if args.pairs else discover_pairs(root, models)
    except ValueError as e:
        ap.error(str(e))
    slugs = [pair_slug(a, b) for a, b in pairs]
    contrasts = [(g, r) for g, r in CONTRASTS if g in models]

    # every model's wide table per pair (None where missing / not complete)
    wides = {}
    for s in slugs:
        for m in models:
            wides[(s, m)] = read_wide(os.path.join(root, m, s))
        for m in {r for _, r in contrasts}:
            wides[(s, m)] = read_wide(os.path.join(ref_root, m, s))

    # each contrast uses the pairs complete for both of its models
    contrast_pairs = {c: [s for s in slugs if wides[(s, c[0])] is not None
                          and wides[(s, c[1])] is not None] for c in contrasts}
    contrasts = [c for c in contrasts if contrast_pairs[c]]
    if not contrasts:
        print(f"compare_to_reference: nothing to compare (GRU: {root}, reference: {ref_root}). "
              f"Reference results missing? They come from Conformer_decimated_CWT.")
        return 1
    incomplete = [s for s in slugs if any(s not in contrast_pairs[c] for c in contrasts)]
    if incomplete:
        print(f"compare_to_reference: pairs missing from at least one contrast: "
              f"{', '.join(incomplete)}")

    first = next(iter(contrast_pairs[contrasts[0]]))
    columns = list(wides[(first, contrasts[0][0])].columns)   # windows..., avg

    def name(c):
        return f"{SHORT[c[0]]} - {SHORT[c[1]]}"

    rows = []
    for c in contrasts:
        g, r = c
        for s in contrast_pairs[c]:
            for col in columns:
                rows.append(dict(scope='pair', pair=s, contrast=name(c), window=col,
                                 **paired(wides[(s, g)][col], wides[(s, r)][col])))
        ps = contrast_pairs[c]
        pooled_g = pd.concat([wides[(s, g)] for s in ps]).groupby(level=0).mean()
        pooled_r = pd.concat([wides[(s, r)] for s in ps]).groupby(level=0).mean()
        for col in columns:
            rows.append(dict(scope='pooled', pair=f"ALL({len(ps)})", contrast=name(c),
                             window=col, **paired(pooled_g[col], pooled_r[col])))
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
    pooled_df = df[df['scope'] == 'pooled'].set_index(['contrast', 'window'])
    lines = [
        f"# FINE + GRU vs Conformer A / B / B + CWT ({args.mode} mode)",
        "",
        f"GRU results: `{root}`. Reference results: `{ref_root}`.",
        f"Paired over up to {n_subj} subjects. Differences in points, GRU model minus "
        f"reference. 95% CI, paired t-test p and Holm-corrected p within each scope; "
        f"W = subjects GRU model better / reference better.",
        "",
        "## Mean accuracy, pooled over each contrast's pairs",
        "",
        "| Contrast | Pairs | Model | " + " | ".join(win_labels) + " |",
        "|" + "|".join(['---', '---:', '---'] + ['---:'] * len(columns)) + "|",
    ]
    for c in contrasts:
        for side, m in (('mean_gru', c[0]), ('mean_ref', c[1])):
            cells = [f"{pooled_df.loc[(name(c), col), side]:.2f}" for col in columns]
            lines.append(f"| {name(c)} | {len(contrast_pairs[c])} | {LABELS[m]} | "
                         + " | ".join(cells) + " |")

    lines += ["", "## Pooled differences", "",
              "| Contrast | " + " | ".join(win_labels) + " |",
              "|" + "|".join(['---'] + ['---:'] * len(columns)) + "|"]
    for c in contrasts:
        cells = []
        for col in columns:
            r = pooled_df.loc[(name(c), col)]
            txt = (f"{r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] "
                   f"(holm={fmt_p(r['p_holm'])}, W {r['n_gru_better']}-{r['n_ref_better']})")
            cells.append(f"**{txt}**" if r['p_holm'] < 0.05 else txt)
        lines.append(f"| {name(c)} | " + " | ".join(cells) + " |")

    per_pair = df[df['scope'] == 'pair']
    lines += ["", "## Per pair: how many pairs favour each side", "",
              "Pairs where the difference is positive / negative; in brackets, how many of "
              "those are significant after Holm correction over all per-pair tests.", "",
              "| Contrast | " + " | ".join(win_labels) + " |",
              "|" + "|".join(['---'] + ['---:'] * len(columns)) + "|"]
    for c in contrasts:
        cells = []
        for col in columns:
            sub = per_pair[(per_pair['contrast'] == name(c)) & (per_pair['window'] == col)]
            pos, neg = sub[sub['diff'] > 0], sub[sub['diff'] < 0]
            cells.append(f"{len(pos)} [{int((pos['p_holm'] < 0.05).sum())}] / "
                         f"{len(neg)} [{int((neg['p_holm'] < 0.05).sum())}]")
        lines.append(f"| {name(c)} | " + " | ".join(cells) + " |")

    lines += ["", f"## Verdict (pooled window average, margin {args.margin:g} points)", ""]
    verdicts = [f"- {LABELS[c[0]]} vs {LABELS[c[1]]}: "
                f"{verdict(pooled_df.loc[(name(c), AVG)], args.margin)}." for c in contrasts]
    lines += verdicts
    if args.mode != ref_mode or args.mode == 'smoke':
        lines += ["", "Note: smoke-mode numbers only prove the pipeline runs; they say nothing "
                      "about accuracy."]
    lines += ["", "Labels: " + ", ".join(f"{SHORT[m]} = {LABELS[m]}" for m in SHORT)]
    if incomplete:
        lines += ["", "Missing from at least one contrast: " + ", ".join(incomplete)]

    md_path = os.path.join(out_dir, 'vs_reference.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"compare_to_reference: written {md_path}")
    print("\nPooled differences (GRU model minus reference):")
    print(pooled_df[['diff', 'ci_lo', 'ci_hi', 'p_holm', 'n_gru_better', 'n_ref_better']]
          .round(3).to_string())
    print("\n" + "\n".join(verdicts))
    return 0


if __name__ == '__main__':
    sys.exit(main())
