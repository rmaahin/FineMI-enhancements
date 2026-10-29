# LEAN vs EEG Conformer on the same OFNER data, folds and seeds, for one band (no
# training, no torch). Produces the same quantities the ICASSP paper quotes for FineMI,
# so the OFNER results can be reported side by side.
#
#   python -m scripts.compare_models --band 0.5_3hz                  # full mode, all 15 pairs
#   python -m scripts.compare_models --band 8_30hz --mode smoke --pairs EF_EE SUP_PRO HC_HO
#
# Every number is given twice:
#   avg      the mean over the three windows (800 / 1500 / 3000 ms), the counterpart
#            of the window average the FineMI paper reports.
#   3000 ms  the whole 0 to 3 s post-cue window only.
#
# Pair groups: Same-joint = the three antagonist pairs (EF/EE elbow, SUP/PRO forearm,
# HC/HO hand); Cross-joint = the other 12.
#
# Writes results/<band>/<mode>/comparison/:
#   per_pair.csv        pair, group, both models' accuracy per window + avg, gap
#   paired_stats.csv    LEAN minus EEG Conformer, paired over subjects: pooled over the
#                       selected pairs and per pair; mean, 95% CI, t-test p, Holm p,
#                       subjects better / worse
#   ofner_numbers.md    overall averages, retention, gaps, the 3 smallest gaps, Pearson
#                       r, pair groups, and table rows ready to paste into LaTeX

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (BAND_PROFILES, DEFAULT_BAND, EXPECTED_PARAMS, MODE_OVERRIDES,
                        MODEL_LABELS, RESULTS_ROOT_DEFAULT, mode_root, normalize_band,
                        pair_slug, parse_pairs)

LEAN, CONF = 'conformer_b_slim', 'conformer_a'
AVG = 'avg'
LONGEST = '3000'

# Joint each movement belongs to: elbow, forearm (radio-ulnar), hand.
JOINT_OF = {'EF': 'elbow', 'EE': 'elbow', 'SUP': 'forearm', 'PRO': 'forearm',
            'HC': 'hand', 'HO': 'hand'}
GROUPS = ('Cross-joint', 'Same-joint')


def joint_group(slug):
    """Same-joint for the antagonist pairs (EF/EE, SUP/PRO, HC/HO), else Cross-joint."""
    a, b = slug.split('_')
    return 'Same-joint' if JOINT_OF[a] == JOINT_OF[b] else 'Cross-joint'


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
    """x - y paired over the subjects both have."""
    common = x.index.intersection(y.index)
    d = (x.loc[common] - y.loc[common]).dropna()
    n = len(d)
    res = dict(n_subjects=n, mean_lean=x.loc[d.index].mean(), mean_conf=y.loc[d.index].mean(),
               diff=d.mean(), ci_lo=np.nan, ci_hi=np.nan, p_ttest=np.nan,
               n_lean_better=int((d > 1e-9).sum()), n_conf_better=int((d < -1e-9).sum()))
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


def paper_block(per_pair, col, title):
    """Markdown lines with the paper's quantities for one accuracy column (avg or 3000)."""
    lean = per_pair[f'lean_{col}']
    conf = per_pair[f'conf_{col}']
    gap = conf - lean                                   # the paper's "gap": Conformer - LEAN
    n = len(per_pair)
    lines = [f"## {title}", "",
             f"- LEAN overall average: **{lean.mean():.2f}%**",
             f"- EEG Conformer overall average: **{conf.mean():.2f}%**",
             f"- Average gap (Conformer - LEAN): **{gap.mean():.2f}** points",
             f"- LEAN retains **{100 * lean.mean() / conf.mean():.1f}%** of EEG Conformer's "
             f"average accuracy",
             f"- Gap below 4 points in **{int((gap < 4).sum())} of {n}** pairs; "
             f"LEAN ahead in {int((gap < 0).sum())} pair(s)"]
    smallest = gap.sort_values().head(3)
    lines.append("- Three smallest gaps: " + ", ".join(
        f"{p.replace('_', '/')} ({g:.2f})" for p, g in smallest.items()))
    if n >= 3:
        r, p = stats.pearsonr(lean, conf)
        lines.append(f"- Pearson r between the two models' per-pair accuracies: "
                     f"**{r:.2f}** (p {fmt_p(p)})")
    lines += ["", "| Group | Pairs | LEAN | EEG Conformer | Gap |", "|---|---:|---:|---:|---:|"]
    for g in GROUPS:
        m = per_pair['group'] == g
        if m.any():
            lines.append(f"| {g} | {int(m.sum())} | {lean[m].mean():.2f} | {conf[m].mean():.2f} "
                         f"| {gap[m].mean():.2f} |")
    lines.append(f"| Overall | {n} | {lean.mean():.2f} | {conf.mean():.2f} | {gap.mean():.2f} |")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser(description="LEAN vs EEG Conformer on OFNER, one band.")
    ap.add_argument('--band', default=DEFAULT_BAND,
                    help=f"one of {', '.join(BAND_PROFILES)} (default {DEFAULT_BAND})")
    ap.add_argument('--pairs', nargs='+', default=['all'])
    ap.add_argument('--mode', choices=list(MODE_OVERRIDES), default='full')
    ap.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    args = ap.parse_args()

    try:
        pairs = parse_pairs(args.pairs)
        band = normalize_band(args.band)
    except ValueError as e:
        ap.error(str(e))
    root = mode_root(args.results_root, band, args.mode)

    wides, skipped = {}, []
    for a, b in pairs:
        s = pair_slug(a, b)
        w = {m: read_wide(os.path.join(root, m, s)) for m in (LEAN, CONF)}
        missing = [m for m, v in w.items() if v is None]
        if missing:
            skipped.append(f"{s} ({', '.join(missing)})")
        else:
            wides[s] = w
    if skipped:
        print(f"compare_models: skipping pairs with missing results: {'; '.join(skipped)}")
    if not wides:
        print(f"compare_models: nothing to compare in {root}")
        return 1
    complete = list(wides)
    columns = list(wides[complete[0]][LEAN].columns)          # windows..., avg
    win_labels = [f"{c} ms" if c != AVG else 'Avg' for c in columns]

    # ---- per pair ---------------------------------------------------------------
    rows = []
    for s in complete:
        row = dict(pair=s, group=joint_group(s))
        for c in columns:
            row[f'lean_{c}'] = wides[s][LEAN][c].mean()
            row[f'conf_{c}'] = wides[s][CONF][c].mean()
            row[f'gap_{c}'] = row[f'conf_{c}'] - row[f'lean_{c}']
        rows.append(row)
    per_pair = pd.DataFrame(rows).set_index('pair')          # indexed by slug, e.g. EF_EE

    # ---- paired statistics --------------------------------------------------------
    stat_rows = []
    for s in complete:
        for c in columns:
            stat_rows.append(dict(scope='pair', pair=s, window=c,
                                  **paired(wides[s][LEAN][c], wides[s][CONF][c])))
    pooled = {m: pd.concat([wides[s][m] for s in complete]).groupby(level=0).mean()
              for m in (LEAN, CONF)}
    for c in columns:
        stat_rows.append(dict(scope='pooled', pair=f"ALL({len(complete)})", window=c,
                              **paired(pooled[LEAN][c], pooled[CONF][c])))
    st = pd.DataFrame(stat_rows)
    st['p_holm'] = np.nan
    for scope in ('pair', 'pooled'):
        mask = st['scope'] == scope
        st.loc[mask, 'p_holm'] = holm(st.loc[mask, 'p_ttest'])

    out_dir = os.path.join(root, 'comparison')
    os.makedirs(out_dir, exist_ok=True)
    out = per_pair.copy()
    out.index = [p.replace('_', '/') for p in out.index]
    out.index.name = 'pair'
    out.round(4).to_csv(os.path.join(out_dir, 'per_pair.csv'))
    st.round(6).to_csv(os.path.join(out_dir, 'paired_stats.csv'), index=False)

    # ---- ofner_numbers.md -----------------------------------------------------------
    n_subj = int(st['n_subjects'].max())
    decim = BAND_PROFILES[band]['decim']
    rate = 250 // decim
    lines = [f"# LEAN vs EEG Conformer on OFNER, {BAND_PROFILES[band]['label']} "
             f"({args.mode} mode, {len(complete)} pairs, {n_subj} subjects)",
             "",
             f"Data: Ofner et al. 2017, motor imagery, 0 to 3 s after the cue, fixation-cross "
             f"baseline (-2 to 0 s), {BAND_PROFILES[band]['label']}, "
             + (f"decimated to {rate} Hz." if decim > 1 else f"kept at {rate} Hz (no decimation).")
             + " Windows start at the cue.",
             "",
             "## Parameters", "",
             "| Model | " + " | ".join(win_labels[:-1]) + " |",
             "|" + "|".join(['---'] + ['---:'] * (len(columns) - 1)) + "|"]
    for m in (LEAN, CONF):
        lines.append(f"| {MODEL_LABELS[m]} | " + " | ".join(
            f"{EXPECTED_PARAMS[band][m][int(c)]:,}" for c in columns[:-1]) + " |")

    lines += ["", "## Mean accuracy over the selected pairs", "",
              "| Model | " + " | ".join(win_labels) + " |",
              "|" + "|".join(['---'] + ['---:'] * len(columns)) + "|"]
    for m, key in ((LEAN, 'lean'), (CONF, 'conf')):
        lines.append(f"| {MODEL_LABELS[m]} | " + " | ".join(
            f"{per_pair[f'{key}_{c}'].mean():.2f}" for c in columns) + " |")

    pooled_st = st[st['scope'] == 'pooled'].set_index('window')
    lines += ["", f"## LEAN minus EEG Conformer, paired over subjects (pooled over "
                  f"{len(complete)} pairs)", "",
              "| Window | Diff | 95% CI | Holm p | Subjects LEAN better / Conformer better |",
              "|---|---:|---:|---:|---:|"]
    for c, lab in zip(columns, win_labels):
        r = pooled_st.loc[c]
        lines.append(f"| {lab} | {r['diff']:+.2f} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | "
                     f"{fmt_p(r['p_holm'])} | {r['n_lean_better']}-{r['n_conf_better']} |")
    lines.append("")

    lines += paper_block(per_pair, AVG, "Window average (800 / 1500 / 3000 ms)")
    if LONGEST in columns:
        lines += paper_block(per_pair, LONGEST, "3000 ms window only")

    for c, title in ((AVG, "window average"), (LONGEST, "3000 ms")):
        if c not in columns:
            continue
        lines += [f"## Table rows, {title} (EEG Conformer & LEAN; best in bold)", "", "```"]
        for p in per_pair.index:
            l, f = per_pair.loc[p, f'lean_{c}'], per_pair.loc[p, f'conf_{c}']
            fc = f"\\textbf{{{f:.2f}}}" if f >= l else f"{f:.2f}"
            lc = f"\\textbf{{{l:.2f}}}" if l > f else f"{l:.2f}"
            lines.append(f"{p.replace('_', '/')} & {fc} & {lc} \\\\")
        for g in GROUPS:
            m = per_pair['group'] == g
            if m.any():
                lines.append(f"{g} avg. & {per_pair.loc[m, f'conf_{c}'].mean():.2f} & "
                             f"{per_pair.loc[m, f'lean_{c}'].mean():.2f} \\\\")
        lines.append(f"Overall avg. & {per_pair[f'conf_{c}'].mean():.2f} & "
                     f"{per_pair[f'lean_{c}'].mean():.2f} \\\\")
        lines += ["```", ""]

    if args.mode == 'smoke':
        lines += ["Note: smoke-mode numbers only prove the pipeline runs; they say nothing "
                  "about accuracy.", ""]
    if skipped:
        lines += ["Skipped (missing results): " + "; ".join(skipped), ""]

    md_path = os.path.join(out_dir, 'ofner_numbers.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"compare_models: written {md_path}")
    print("\nMean accuracy over the selected pairs:")
    table = pd.DataFrame({MODEL_LABELS[m]: [per_pair[f'{k}_{c}'].mean() for c in columns]
                          for m, k in ((LEAN, 'lean'), (CONF, 'conf'))}, index=win_labels).T
    print(table.round(2).to_string())
    print("\nLEAN minus EEG Conformer (pooled, paired over subjects):")
    print(pooled_st[['diff', 'ci_lo', 'ci_hi', 'p_holm', 'n_lean_better', 'n_conf_better']]
          .round(3).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
