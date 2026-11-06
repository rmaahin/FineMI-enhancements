# EEGNet results (250 Hz and 50 Hz), and how they compare with each other and with
# earlier runs on the same subjects (no training, no torch). run_vast.sh runs it after
# check_results.
#
#   python -m scripts.summarize                    # full mode, all 28 pairs
#   python -m scripts.summarize --mode smoke --pairs WAA_SAA HOC_WFE EPS_SPS
#
# References are the runs in REFERENCES (src/config.py: LEAN, EEG Conformer,
# ShallowConvNet), read from <reference folder>/<PAIR>/wide_subject_x_window.csv. They
# use the same subjects and labels, hence the same CV folds, so differences are paired
# over subjects. A missing reference, or one missing some pairs, is reported and
# compared on the pairs it has.
#
# Comparisons (each "A minus B", paired over subjects):
#   EEGNet 250 Hz - EEGNet 50 Hz
#   each EEGNet model - each reference
#
# Every accuracy is given per window and as "avg", the mean over the four windows
# (800 / 1500 / 3000 / 4000 ms), which is what the paper tables use.
#
# Writes results/<mode>/summary/:
#   per_pair.csv        pair, joint group, each model's and reference's accuracy per
#                       window + avg
#   paired_stats.csv    every comparison, pooled over the pairs and per pair: mean
#                       difference, 95% CI, t-test p, Holm p, subjects better / worse
#   summary.md          the tables above in readable form

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (BAND_LABEL, EXPECTED_PARAMS, MODE_OVERRIDES, MODELS, MODEL_LABELS,
                        PROJECT_ROOT, REFERENCES, RESULTS_ROOT_DEFAULT, make_config,
                        mode_root, pair_slug, parse_pairs)

AVG = 'avg'

# Joint of each movement, in anatomical order hand < wrist < elbow < shoulder.
JOINT_OF = {'HOC': 0, 'WFE': 1, 'WAA': 1, 'EPS': 2, 'EFE': 2, 'SPS': 3, 'SAA': 3, 'SFE': 3}
GROUPS = ('Distant', 'Adjacent', 'Same-joint')


def joint_group(slug):
    """The paper's grouping: same joint; neighbouring joints (hand-wrist, wrist-elbow,
    elbow-shoulder); non-neighbouring joints (hand-elbow, hand-shoulder, wrist-shoulder)."""
    a, b = slug.split('_')
    gap = abs(JOINT_OF[a] - JOINT_OF[b])
    return 'Same-joint' if gap == 0 else 'Adjacent' if gap == 1 else 'Distant'


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
    res = dict(n_subjects=n, mean_a=x.loc[d.index].mean(), mean_b=y.loc[d.index].mean(),
               diff=d.mean(), ci_lo=np.nan, ci_hi=np.nan, p_ttest=np.nan,
               n_a_better=int((d > 1e-9).sum()), n_b_better=int((d < -1e-9).sum()))
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


def md_table(header, rows):
    return (["| " + " | ".join(header) + " |",
             "|" + "|".join(['---'] + ['---:'] * (len(header) - 1)) + "|"]
            + ["| " + " | ".join(r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser(description="EEGNet results and comparisons.")
    ap.add_argument('--pairs', nargs='+', default=['all'])
    ap.add_argument('--mode', choices=list(MODE_OVERRIDES), default='full')
    ap.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    args = ap.parse_args()

    try:
        pairs = parse_pairs(args.pairs)
    except ValueError as e:
        ap.error(str(e))
    root = mode_root(args.results_root, args.mode)
    slugs = [pair_slug(a, b) for a, b in pairs]

    # ---- every run: EEGNet models first, then references -----------------------------
    # runs[label] = {slug: wide}; notes collect anything missing
    runs, notes, own_labels = {}, [], []
    for m in MODELS:
        label = MODEL_LABELS[m]
        w = {s: read_wide(os.path.join(root, m, s)) for s in slugs}
        have = {s: v for s, v in w.items() if v is not None}
        if len(have) < len(slugs):
            notes.append(f"{label}: no results for "
                         f"{', '.join(s for s in slugs if s not in have)}")
        if have:
            runs[label] = have
            own_labels.append(label)
    if not own_labels:
        print(f"summarize: no EEGNet results in {root}")
        return 1
    columns = list(next(iter(runs[own_labels[0]].values())).columns)     # windows..., avg
    win_labels = [f"{c} ms" if c != AVG else 'Avg' for c in columns]

    ref_labels = []
    for label, rel in REFERENCES.items():
        ref_dir = os.path.normpath(os.path.join(PROJECT_ROOT, rel.format(mode=args.mode)))
        have = {s: read_wide(os.path.join(ref_dir, s)) for s in slugs}
        have = {s: v for s, v in have.items() if v is not None}
        if not have:
            notes.append(f"{label}: no results in {ref_dir} - skipped")
            continue
        if list(next(iter(have.values())).columns) != columns:
            notes.append(f"{label}: windows differ from EEGNet's - skipped")
            continue
        if len(have) < len(slugs):
            notes.append(f"{label}: has {len(have)} of {len(slugs)} pairs "
                         f"(missing {', '.join(s for s in slugs if s not in have)})")
        runs[label] = have
        ref_labels.append(label)

    # ---- per-pair table -----------------------------------------------------------------
    per_pair = pd.DataFrame(index=pd.Index(slugs, name='pair'))
    per_pair['group'] = [joint_group(s) for s in slugs]
    for label, have in runs.items():
        for c in columns:
            per_pair[f'{label}_{c}'] = [have[s][c].mean() if s in have else np.nan for s in slugs]

    # ---- comparisons --------------------------------------------------------------------
    comparisons = []
    if len(own_labels) == 2:
        comparisons.append((own_labels[0], own_labels[1]))
    comparisons += [(o, r) for o in own_labels for r in ref_labels]

    stat_rows, comp_pairs = [], {}
    for a_lab, b_lab in comparisons:
        have = [s for s in slugs if s in runs[a_lab] and s in runs[b_lab]]
        comp_pairs[(a_lab, b_lab)] = have
        if not have:
            continue
        for s in have:
            for c in columns:
                stat_rows.append(dict(a=a_lab, b=b_lab, scope='pair', pair=s, window=c,
                                      **paired(runs[a_lab][s][c], runs[b_lab][s][c])))
        # pooled: each subject's mean over the pairs both runs have
        pa = pd.concat([runs[a_lab][s] for s in have]).groupby(level=0).mean()
        pb = pd.concat([runs[b_lab][s] for s in have]).groupby(level=0).mean()
        for c in columns:
            stat_rows.append(dict(a=a_lab, b=b_lab, scope='pooled', pair=f"ALL({len(have)})",
                                  window=c, **paired(pa[c], pb[c])))

    st = pd.DataFrame(stat_rows)
    if len(st):
        st['p_holm'] = np.nan
        for _, idx in st.groupby(['a', 'b', 'scope']).groups.items():
            st.loc[idx, 'p_holm'] = holm(st.loc[idx, 'p_ttest'])

    out_dir = os.path.join(root, 'summary')
    os.makedirs(out_dir, exist_ok=True)
    out = per_pair.copy()
    out.index = [p.replace('_', '/') for p in out.index]
    out.index.name = 'pair'
    out.round(4).to_csv(os.path.join(out_dir, 'per_pair.csv'))
    if len(st):
        st.round(6).to_csv(os.path.join(out_dir, 'paired_stats.csv'), index=False)

    # ---- summary.md ---------------------------------------------------------------------
    n_subj = len(pd.concat(list(runs[own_labels[0]].values())).index.unique())
    lines = [f"# EEGNet-8,2, {BAND_LABEL} ({args.mode} mode, {len(slugs)} pairs, "
             f"{n_subj} subjects)", "",
             f"Data: fixation-cross baseline (-4 to -2 s), motor imagery only (0 to 4 s after "
             f"the trigger), {BAND_LABEL}. Windows start at imagery onset. EEGNet 250 Hz uses "
             f"the full rate; EEGNet 50 Hz is decimated by 5.", "",
             "## Parameters", ""]
    rows = []
    for m in MODELS:
        cfg = make_config(m, args.mode)
        rows.append([f"{MODEL_LABELS[m]} (K1 {cfg.eegnet['kern_length']}, pools "
                     f"{cfg.eegnet['pool1']}/{cfg.eegnet['pool2']}, sep K "
                     f"{cfg.eegnet['sep_kern_length']})"]
                    + [f"{EXPECTED_PARAMS[m][int(c)]:,}" for c in columns[:-1]])
    lines += md_table(['Model'] + win_labels[:-1], rows)

    lines += ["", "## Mean accuracy over the pairs", ""]
    rows = []
    for label in runs:
        n = per_pair[f'{label}_{AVG}'].notna().sum()
        tag = '' if n == len(slugs) else f" ({n} pairs)"
        rows.append([label + tag] + [f"{per_pair[f'{label}_{c}'].mean():.2f}" for c in columns])
    lines += md_table(['Model'] + win_labels, rows)
    for label in own_labels:
        pooled = pd.concat(list(runs[label].values())).groupby(level=0).mean()
        if len(pooled) > 1:
            sd = pooled[columns].std(ddof=1)
            lines += ["", f"{label} SD over subjects (each subject's mean over the pairs): "
                      + ", ".join(f"{lab} {sd[c]:.2f}" for c, lab in zip(columns, win_labels))]

    lines += ["", "## By joint group (window average)", ""]
    header = ['Group', 'Pairs'] + list(runs)
    rows = []
    for g in GROUPS + ('Overall',):
        m = per_pair['group'] == g if g != 'Overall' else np.ones(len(per_pair), bool)
        if not m.any():
            continue
        rows.append([g, str(int(m.sum()))]
                    + [f"{per_pair.loc[m, f'{label}_{AVG}'].mean():.2f}" for label in runs])
    lines += md_table(header, rows)

    for (a_lab, b_lab), have in comp_pairs.items():
        if not have:
            continue
        pst = st[(st['a'] == a_lab) & (st['b'] == b_lab)
                 & (st['scope'] == 'pooled')].set_index('window')
        lines += ["", f"## {a_lab} minus {b_lab}, paired over subjects "
                      f"(pooled over {len(have)} pairs)", ""]
        lines += md_table(['Window', 'Diff', '95% CI', 'Holm p',
                           f'Subjects {a_lab} better / worse'],
                          [[lab, f"{pst.loc[c, 'diff']:+.2f}",
                            f"[{pst.loc[c, 'ci_lo']:+.2f}, {pst.loc[c, 'ci_hi']:+.2f}]",
                            fmt_p(pst.loc[c, 'p_holm']),
                            f"{pst.loc[c, 'n_a_better']}-{pst.loc[c, 'n_b_better']}"]
                           for c, lab in zip(columns, win_labels)])
        d = (per_pair.loc[have, f'{a_lab}_{AVG}'] - per_pair.loc[have, f'{b_lab}_{AVG}'])
        lines += ["", f"- Window average: {a_lab} ahead in {int((d > 0).sum())} of "
                      f"{len(have)} pairs; largest lead {d.idxmax().replace('_', '/')} "
                      f"({d.max():+.2f}), largest deficit {d.idxmin().replace('_', '/')} "
                      f"({d.min():+.2f})"]
        if len(have) >= 3:
            r, p = stats.pearsonr(per_pair.loc[have, f'{a_lab}_{AVG}'],
                                  per_pair.loc[have, f'{b_lab}_{AVG}'])
            lines.append(f"- Pearson r between per-pair window averages: {r:.2f} (p {fmt_p(p)})")

    lines += ["", "## Per pair (sorted by the first EEGNet model's window average)", ""]
    header = ['Pair', 'Group'] + [f"{label} {lab}" for label in own_labels
                                  for lab in win_labels] + \
             [f"{label} Avg" for label in ref_labels]
    rows = []
    for s in per_pair.sort_values(f'{own_labels[0]}_{AVG}', ascending=False).index:
        row = [s.replace('_', '/'), per_pair.loc[s, 'group']]
        for label in own_labels:
            row += ['-' if pd.isna(v) else f"{v:.2f}"
                    for v in (per_pair.loc[s, f'{label}_{c}'] for c in columns)]
        for label in ref_labels:
            v = per_pair.loc[s, f'{label}_{AVG}']
            row.append('-' if pd.isna(v) else f"{v:.2f}")
        rows.append(row)
    lines += md_table(header, rows)
    lines.append("")

    if notes:
        lines += ["Notes: " + "; ".join(notes), ""]
    if args.mode == 'smoke':
        lines += ["Note: smoke-mode numbers only prove the pipeline runs; they say nothing "
                  "about accuracy.", ""]

    md_path = os.path.join(out_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"summarize: written {md_path}")
    for note in notes:
        print(f"  {note}")
    print("\nMean accuracy over the pairs:")
    table = {label: [per_pair[f'{label}_{c}'].mean() for c in columns] for label in runs}
    print(pd.DataFrame(table, index=win_labels).T.round(2).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
