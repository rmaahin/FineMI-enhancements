# ShallowConvNet results, and how they compare with earlier runs on the same subjects
# (no training, no torch). run_vast.sh runs it after check_results.
#
#   python -m scripts.summarize                    # full mode, all 28 pairs
#   python -m scripts.summarize --mode smoke --pairs WAA_SAA HOC_WFE EPS_SPS
#
# References are the runs in REFERENCES (src/config.py), read from
# <reference folder>/<PAIR>/wide_subject_x_window.csv. They use the same subjects and
# labels, hence the same CV folds, so differences are paired over subjects. A missing
# reference, or one missing some pairs, is reported and compared on the pairs it has.
#
# Every accuracy is given per window and as "avg", the mean over the four windows
# (800 / 1500 / 3000 / 4000 ms), which is what the paper tables use.
#
# Writes results/<mode>/summary/:
#   per_pair.csv        pair, joint group, ShallowConvNet per window + avg, and each
#                       reference's accuracy and difference (ShallowConvNet - reference)
#   paired_stats.csv    ShallowConvNet minus each reference, paired over subjects:
#                       pooled over the pairs and per pair; mean, 95% CI, t-test p,
#                       Holm p, subjects better / worse
#   summary.md          the tables above in readable form

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (BAND_LABEL, EXPECTED_PARAMS, MODE_OVERRIDES, PROJECT_ROOT,
                        REFERENCES, RESULTS_ROOT_DEFAULT, mode_root, pair_slug, parse_pairs)

MODEL = 'shallowconvnet'
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
    res = dict(n_subjects=n, mean_shallow=x.loc[d.index].mean(), mean_ref=y.loc[d.index].mean(),
               diff=d.mean(), ci_lo=np.nan, ci_hi=np.nan, p_ttest=np.nan,
               n_shallow_better=int((d > 1e-9).sum()), n_ref_better=int((d < -1e-9).sum()))
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
    ap = argparse.ArgumentParser(description="ShallowConvNet results and comparisons.")
    ap.add_argument('--pairs', nargs='+', default=['all'])
    ap.add_argument('--mode', choices=list(MODE_OVERRIDES), default='full')
    ap.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    args = ap.parse_args()

    try:
        pairs = parse_pairs(args.pairs)
    except ValueError as e:
        ap.error(str(e))
    root = mode_root(args.results_root, args.mode)

    # ---- ShallowConvNet ------------------------------------------------------------
    own, missing = {}, []
    for a, b in pairs:
        s = pair_slug(a, b)
        w = read_wide(os.path.join(root, MODEL, s))
        if w is None:
            missing.append(s)
        else:
            own[s] = w
    if missing:
        print(f"summarize: no ShallowConvNet results for {', '.join(missing)}")
    if not own:
        print(f"summarize: nothing to summarize in {os.path.join(root, MODEL)}")
        return 1
    complete = list(own)
    columns = list(own[complete[0]].columns)                  # windows..., avg
    win_labels = [f"{c} ms" if c != AVG else 'Avg' for c in columns]

    per_pair = pd.DataFrame(
        [dict(pair=s, group=joint_group(s), **{f'shallow_{c}': own[s][c].mean() for c in columns})
         for s in complete]).set_index('pair')
    pooled_own = pd.concat([own[s] for s in complete]).groupby(level=0).mean()

    # ---- references ----------------------------------------------------------------
    refs, ref_notes, stat_rows = {}, [], []
    for label, rel in REFERENCES.items():
        ref_dir = os.path.normpath(os.path.join(PROJECT_ROOT, rel.format(mode=args.mode)))
        wides = {s: read_wide(os.path.join(ref_dir, s)) for s in complete}
        have = [s for s in complete if wides[s] is not None]
        if not have:
            ref_notes.append(f"{label}: no results in {ref_dir} - skipped")
            continue
        if len(have) < len(complete):
            ref_notes.append(f"{label}: compared on {len(have)} of {len(complete)} pairs "
                             f"(missing {', '.join(s for s in complete if s not in have)})")
        if list(wides[have[0]].columns) != columns:
            ref_notes.append(f"{label}: windows {list(wides[have[0]].columns)[:-1]} differ "
                             f"from ShallowConvNet's - skipped")
            continue
        refs[label] = have
        for s in have:
            for c in columns:
                per_pair.loc[s, f'{label}_{c}'] = wides[s][c].mean()
                per_pair.loc[s, f'diff_{label}_{c}'] = per_pair.loc[s, f'shallow_{c}'] - wides[s][c].mean()
                stat_rows.append(dict(reference=label, scope='pair', pair=s, window=c,
                                      **paired(own[s][c], wides[s][c])))
        # pooled: each subject's mean over the pairs both runs have
        p_own = pd.concat([own[s] for s in have]).groupby(level=0).mean()
        p_ref = pd.concat([wides[s] for s in have]).groupby(level=0).mean()
        for c in columns:
            stat_rows.append(dict(reference=label, scope='pooled', pair=f"ALL({len(have)})",
                                  window=c, **paired(p_own[c], p_ref[c])))

    st = pd.DataFrame(stat_rows)
    if len(st):
        st['p_holm'] = np.nan
        for (label, scope), idx in st.groupby(['reference', 'scope']).groups.items():
            st.loc[idx, 'p_holm'] = holm(st.loc[idx, 'p_ttest'])

    out_dir = os.path.join(root, 'summary')
    os.makedirs(out_dir, exist_ok=True)
    out = per_pair.copy()
    out.index = [p.replace('_', '/') for p in out.index]
    out.index.name = 'pair'
    out.round(4).to_csv(os.path.join(out_dir, 'per_pair.csv'))
    if len(st):
        st.round(6).to_csv(os.path.join(out_dir, 'paired_stats.csv'), index=False)

    # ---- summary.md -------------------------------------------------------------------
    n_subj = len(pooled_own)
    lines = [f"# ShallowConvNet, {BAND_LABEL} ({args.mode} mode, {len(complete)} pairs, "
             f"{n_subj} subjects)", "",
             f"Data: fixation-cross baseline (-4 to -2 s), motor imagery only (0 to 4 s after "
             f"the trigger), {BAND_LABEL}, 250 Hz. Windows start at imagery onset.", "",
             "## Parameters", ""]
    lines += md_table(['Model'] + win_labels[:-1],
                      [['ShallowConvNet'] + [f"{EXPECTED_PARAMS[MODEL][int(c)]:,}"
                                             for c in columns[:-1]]])

    lines += ["", "## Mean accuracy over the pairs", ""]
    rows = [['ShallowConvNet'] + [f"{per_pair[f'shallow_{c}'].mean():.2f}" for c in columns]]
    for label, have in refs.items():
        tag = '' if len(have) == len(complete) else f" ({len(have)} pairs)"
        rows.append([label + tag] + [f"{per_pair.loc[have, f'{label}_{c}'].mean():.2f}"
                                     for c in columns])
    lines += md_table(['Model'] + win_labels, rows)
    sd = pooled_own[columns].std(ddof=1) if n_subj > 1 else None
    if sd is not None:
        lines += ["", "ShallowConvNet SD over subjects (each subject's mean over the pairs): "
                  + ", ".join(f"{lab} {sd[c]:.2f}" for c, lab in zip(columns, win_labels))]

    lines += ["", "## By joint group (window average)", ""]
    header = ['Group', 'Pairs', 'ShallowConvNet'] + list(refs)
    rows = []
    for g in GROUPS + ('Overall',):
        m = per_pair['group'] == g if g != 'Overall' else per_pair.index == per_pair.index
        if not m.any():
            continue
        row = [g, str(int(m.sum())), f"{per_pair.loc[m, f'shallow_{AVG}'].mean():.2f}"]
        for label in refs:
            row.append(f"{per_pair.loc[m, f'{label}_{AVG}'].mean():.2f}")
        rows.append(row)
    lines += md_table(header, rows)

    for label, have in refs.items():
        pst = st[(st['reference'] == label) & (st['scope'] == 'pooled')].set_index('window')
        lines += ["", f"## ShallowConvNet minus {label}, paired over subjects "
                      f"(pooled over {len(have)} pairs)", ""]
        lines += md_table(['Window', 'Diff', '95% CI', 'Holm p',
                           'Subjects ShallowConvNet better / worse'],
                          [[lab, f"{pst.loc[c, 'diff']:+.2f}",
                            f"[{pst.loc[c, 'ci_lo']:+.2f}, {pst.loc[c, 'ci_hi']:+.2f}]",
                            fmt_p(pst.loc[c, 'p_holm']),
                            f"{pst.loc[c, 'n_shallow_better']}-{pst.loc[c, 'n_ref_better']}"]
                           for c, lab in zip(columns, win_labels)])
        d = per_pair.loc[have, f'diff_{label}_{AVG}']
        lines += ["", f"- Window average: ShallowConvNet ahead in {int((d > 0).sum())} of "
                      f"{len(have)} pairs; largest lead {d.idxmax().replace('_', '/')} "
                      f"({d.max():+.2f}), largest deficit {d.idxmin().replace('_', '/')} "
                      f"({d.min():+.2f})"]
        if len(have) >= 3:
            r, p = stats.pearsonr(per_pair.loc[have, f'shallow_{AVG}'],
                                  per_pair.loc[have, f'{label}_{AVG}'])
            lines.append(f"- Pearson r between per-pair window averages: {r:.2f} (p {fmt_p(p)})")

    lines += ["", "## Per pair", ""]
    header = ['Pair', 'Group'] + [f"Shallow {lab}" for lab in win_labels] + \
             [f"{label} Avg" for label in refs]
    rows = []
    for s in per_pair.sort_values(f'shallow_{AVG}', ascending=False).index:
        row = [s.replace('_', '/'), per_pair.loc[s, 'group']] + \
              [f"{per_pair.loc[s, f'shallow_{c}']:.2f}" for c in columns]
        for label in refs:
            v = per_pair.loc[s].get(f'{label}_{AVG}', np.nan)
            row.append('-' if pd.isna(v) else f"{v:.2f}")
        rows.append(row)
    lines += md_table(header, rows)
    lines.append("")

    if ref_notes:
        lines += ["References: " + "; ".join(ref_notes), ""]
    if args.mode == 'smoke':
        lines += ["Note: smoke-mode numbers only prove the pipeline runs; they say nothing "
                  "about accuracy.", ""]
    if missing:
        lines += ["No ShallowConvNet results for: " + ", ".join(missing), ""]

    md_path = os.path.join(out_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"summarize: written {md_path}")
    for note in ref_notes:
        print(f"  {note}")
    print("\nMean accuracy over the pairs:")
    table = {'ShallowConvNet': [per_pair[f'shallow_{c}'].mean() for c in columns]}
    for label, have in refs.items():
        table[label] = [per_pair.loc[have, f'{label}_{c}'].mean() for c in columns]
    print(pd.DataFrame(table, index=win_labels).T.round(2).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
