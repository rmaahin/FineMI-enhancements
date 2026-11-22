# LEAN_v2 results and paired comparisons (no training, no torch). run_vast.sh runs it
# after check_results.
#
#   python -m scripts.summarize --mode screen
#   python -m scripts.summarize --mode full --pairs all
#
# Reads every preset that has results in results/<mode>/ (or --models), plus the
# REFERENCES in src/config.py (LEAN and EEGNet 50 Hz full runs, restricted to this run's
# pairs and windows; skipped in smoke mode). All runs share subjects, labels and CV
# folds, so every comparison is paired over subjects: each subject's accuracy averaged
# over the pairs both runs have, then a paired t-test per window, Holm-corrected over
# the windows of that comparison.
#
# Comparisons (each "A minus B"):
#   Ladder steps        R1-R0, R2-R1, ... (each change on top of the previous rungs)
#   Leave-one-out       R7 - (R7 without one change); R7 - R6 is the crops term
#   Training only       LEAN + recipe change - R0
#   Baselines           R7 - EEGNet (own run, own run with R7's recipe, paper run), ...
#   Versus references   every preset - EEGNet 50 Hz (paper run), R0 - LEAN (paper run)
#
# Writes results/<mode>/summary/: per_pair.csv, paired_stats.csv, summary.md

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (BAND_LABEL, FINAL, LADDER, LOO, MODE_OVERRIDES, MODEL_LABELS, MODELS,
                        PROJECT_ROOT, R5_STUDY, REFERENCES, RESULTS_ROOT_DEFAULT,
                        TRAINING_ONLY,
                        expand_models, make_config, mode_root, pair_slug)
from src.param_budget import expected_params
from scripts.run_selected_pairs import resolve_pairs

AVG = 'avg'
JOINT_OF = {'HOC': 0, 'WFE': 1, 'WAA': 1, 'EPS': 2, 'EFE': 2, 'SPS': 3, 'SAA': 3, 'SFE': 3}
GROUPS = ('Distant', 'Adjacent', 'Same-joint')
EEGNET_REF = 'EEGNet 50 Hz (paper run)'
LEAN_REF = 'LEAN (paper run)'


def joint_group(slug):
    a, b = slug.split('_')
    gap = abs(JOINT_OF[a] - JOINT_OF[b])
    return 'Same-joint' if gap == 0 else 'Adjacent' if gap == 1 else 'Distant'


def read_wide(pair_dir, windows=None):
    """subjects x [windows..., avg] (str columns), restricted to `windows`; None if
    missing or if a requested window is absent."""
    path = os.path.join(pair_dir, 'wide_subject_x_window.csv')
    if not os.path.isfile(path):
        return None
    wide = pd.read_csv(path, index_col=0)
    wide.columns = [str(int(float(c))) for c in wide.columns]
    if windows is not None:
        if any(w not in wide.columns for w in windows):
            return None
        wide = wide[list(windows)]
    wide = wide.copy()
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
    ap = argparse.ArgumentParser(description="LEAN_v2 results and comparisons.")
    ap.add_argument('--mode', choices=list(MODE_OVERRIDES), default='screen')
    ap.add_argument('--pairs', nargs='+', default=None)
    ap.add_argument('--models', nargs='+', default=['all'],
                   help="presets/groups to include (only those with results are shown)")
    ap.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    args = ap.parse_args()

    try:
        pairs = resolve_pairs(args.pairs or list(MODE_OVERRIDES[args.mode]['pairs']))
        wanted = expand_models(args.models)
    except ValueError as e:
        ap.error(str(e))
    root = mode_root(args.results_root, args.mode)
    slugs = [pair_slug(a, b) for a, b in pairs]
    windows = [str(w) for w in MODE_OVERRIDES[args.mode]['test_time_windows_ms']]
    columns = windows + [AVG]
    win_labels = [f"{c} ms" for c in windows] + ['Avg']

    # ---- runs: own presets (in MODELS order), then references ---------------------
    runs, notes, own = {}, [], []
    for m in [m for m in MODELS if m in wanted]:
        have = {s: read_wide(os.path.join(root, m, s), windows) for s in slugs}
        have = {s: v for s, v in have.items() if v is not None}
        if not have:
            continue
        if len(have) < len(slugs):
            notes.append(f"{m}: {len(have)} of {len(slugs)} pairs")
        runs[m] = have
        own.append(m)
    if not own:
        print(f"summarize: no results in {root}")
        return 1

    refs = []
    if args.mode != 'smoke':
        for label, rel in REFERENCES.items():
            ref_dir = os.path.normpath(os.path.join(PROJECT_ROOT, rel))
            have = {s: read_wide(os.path.join(ref_dir, s), windows) for s in slugs}
            have = {s: v for s, v in have.items() if v is not None}
            if not have:
                notes.append(f"{label}: no results in {ref_dir} - skipped")
                continue
            if len(have) < len(slugs):
                notes.append(f"{label}: has {len(have)} of {len(slugs)} pairs")
            runs[label] = have
            refs.append(label)

    def name(r):
        return MODEL_LABELS.get(r, r)

    # ---- comparisons ------------------------------------------------------------------
    sections = []   # (title, [(a, b, row label)])
    ladder = [m for m in LADDER if m in runs]
    steps = [(ladder[i], ladder[i - 1], f"{name(ladder[i])} vs {ladder[i - 1][-2:].upper()}")
             for i in range(1, len(ladder))]
    if len(ladder) > 2 and ladder[0] == 'lean_v2_r0':
        steps.append((ladder[-1], ladder[0], f"{ladder[-1][-2:].upper()} vs R0 (all changes)"))
    sections.append(('Ladder steps (each change on top of the rungs before it)', steps))

    loo = []
    if FINAL in runs:
        loo = [(FINAL, m, f"{LOO[m][0].split(' - ', 1)[1]}") for m in LOO if m in runs]
        if 'lean_v2_r6' in runs:
            loo.append((FINAL, 'lean_v2_r6', 'crops'))
    sections.append(('Leave-one-out: R7 minus R7 without the change (positive = the change helps)',
                     loo))

    tr = [(m, 'lean_v2_r0', name(m)) for m in TRAINING_ONLY if m in runs and 'lean_v2_r0' in runs]
    sections.append(('Training changes alone, on LEAN (minus R0)', tr))

    base = []
    for a, b in (('eegnet_50_recipe', 'eegnet_50'), (FINAL, 'eegnet_50_recipe'),
                 (FINAL, 'eegnet_50'), ('lean_v2_r0', LEAN_REF), ('eegnet_50', EEGNET_REF)):
        if a in runs and b in runs:
            base.append((a, b, f"{name(a)} vs {name(b)}"))
    sections.append(('Baselines and reproduction checks', base))

    r5 = 'lean_v2_r5'
    dec = []
    for a, b, label in (
            ('lean_v2_r4_sched', 'lean_v2_r4', 'fixed schedule alone (R4 + schedule vs R4)'),
            ('lean_v2_r4_reg', 'lean_v2_r4', 'regularisation alone (R4 + reg vs R4)'),
            (r5, 'lean_v2_r4_sched', 'regularisation, given the schedule'),
            (r5, 'lean_v2_r4_reg', 'schedule, given the regularisation'),
            *[(r5, m, f"R5 minus R5 without {R5_STUDY[m][0].split(' - ', 1)[1]}")
              for m in R5_STUDY if m.startswith('lean_v2_r5_loo_')],
            ('lean_schedule', 'lean_v2_r0', 'LEAN + schedule vs LEAN'),
            ('lean_r5recipe', 'lean_v2_r0', 'LEAN + R5 recipe vs LEAN'),
            (r5, 'lean_r5recipe', 'R5 vs LEAN, both with the R5 recipe (architecture)'),
            ('eegnet_50_sched', 'eegnet_50', 'EEGNet + schedule vs EEGNet'),
            (r5, 'eegnet_50', 'R5 vs EEGNet (both rerun here)'),
            (r5, 'eegnet_50_sched', 'R5 vs EEGNet + schedule (same schedule)')):
        if a in runs and b in runs:
            dec.append((a, b, label))
    sections.append(('Decomposing R5 (positive = A better)', dec))

    fin = []
    for b in ('eegnet_50_sched', 'eegnet_50', EEGNET_REF, 'lean_v2_r0', LEAN_REF):
        if 'lean_v2_final' in runs and b in runs:
            fin.append(('lean_v2_final', b, f"LEAN v2 (final) vs {name(b)}"))
    sections.append(('Final model (positive = LEAN v2 better)', fin))

    vs_ref = [(m, EEGNET_REF, name(m)) for m in own if EEGNET_REF in runs]
    sections.append((f'Every preset minus {EEGNET_REF} (positive = beats EEGNet)', vs_ref))

    stat_rows = []
    for title, comps in sections:
        for a, b, label in comps:
            have = [s for s in slugs if s in runs[a] and s in runs[b]]
            if not have:
                continue
            pa = pd.concat([runs[a][s] for s in have]).groupby(level=0).mean()
            pb = pd.concat([runs[b][s] for s in have]).groupby(level=0).mean()
            for c in columns:
                stat_rows.append(dict(section=title, label=label, a=a, b=b,
                                      n_pairs=len(have), window=c, **paired(pa[c], pb[c])))
    st = pd.DataFrame(stat_rows)
    if len(st):
        st['p_holm'] = np.nan
        for _, idx in st.groupby(['section', 'a', 'b']).groups.items():
            st.loc[idx, 'p_holm'] = holm(st.loc[idx, 'p_ttest'])

    # ---- per-pair table -----------------------------------------------------------------
    per_pair = pd.DataFrame(index=pd.Index(slugs, name='pair'))
    per_pair['group'] = [joint_group(s) for s in slugs]
    for r, have in runs.items():
        for c in columns:
            per_pair[f'{r}_{c}'] = [have[s][c].mean() if s in have else np.nan for s in slugs]

    out_dir = os.path.join(root, 'summary')
    os.makedirs(out_dir, exist_ok=True)
    pp = per_pair.copy()
    pp.index = [p.replace('_', '/') for p in pp.index]
    pp.index.name = 'pair'
    pp.round(4).to_csv(os.path.join(out_dir, 'per_pair.csv'))
    if len(st):
        st.round(6).to_csv(os.path.join(out_dir, 'paired_stats.csv'), index=False)

    # ---- summary.md -----------------------------------------------------------------------
    n_subj = len(pd.concat(list(runs[own[0]].values())).index.unique())
    lines = [f"# LEAN_v2, {BAND_LABEL}, 50 Hz ({args.mode} mode, {len(slugs)} pairs, "
             f"{n_subj} subjects, windows {', '.join(win_labels[:-1])})", "",
             "Accuracies are means over pairs; comparisons are paired over subjects "
             "(each subject's mean over the pairs). Diff = A minus B in percentage points, "
             "Holm p over this comparison's windows.", ""]

    lines += ["## Mean accuracy", ""]
    rows = []
    for r in runs:
        if r in own:
            cfg = make_config(r, args.mode)
            ps = sorted({expected_params(cfg, s) for s in cfg.window_samples()})
            params = ' / '.join(f"{p:,}" for p in ps)
        else:
            params = '-'
        n = per_pair[f'{r}_{AVG}'].notna().sum()
        rows.append([name(r) + ('' if n == len(slugs) else f" ({n} pairs)"), params]
                    + [f"{per_pair[f'{r}_{c}'].mean():.2f}" for c in columns])
    lines += md_table(['Model', 'Params'] + win_labels, rows)

    for title, comps in sections:
        if not len(st) or not comps:
            continue
        sub = st[st['section'] == title]
        if not len(sub):
            continue
        lines += ["", f"## {title}", ""]
        rows = []
        for a, b, label in comps:
            cs = sub[(sub['a'] == a) & (sub['b'] == b)].set_index('window')
            if not len(cs):
                continue
            cells = [f"{cs.loc[c, 'diff']:+.2f} (p {fmt_p(cs.loc[c, 'p_holm'])})" for c in columns]
            avg = cs.loc[AVG]
            rows.append([label] + cells
                        + [f"[{avg['ci_lo']:+.2f}, {avg['ci_hi']:+.2f}]",
                           f"{avg['n_a_better']}-{avg['n_b_better']}"])
        lines += md_table(['Comparison'] + win_labels + ['Avg 95% CI', 'Subjects +/-'], rows)

    lines += ["", "## By joint group (window average)", ""]
    rows = []
    for g in GROUPS + ('Overall',):
        msk = per_pair['group'] == g if g != 'Overall' else np.ones(len(per_pair), bool)
        if not msk.any():
            continue
        rows.append([g, str(int(msk.sum()))]
                    + [f"{per_pair.loc[msk, f'{r}_{AVG}'].mean():.2f}" for r in runs])
    lines += md_table(['Group', 'Pairs'] + [name(r) for r in runs], rows)

    lines += ["", "## Per pair (window average)", ""]
    rows = []
    for s in slugs:
        rows.append([s.replace('_', '/'), per_pair.loc[s, 'group']]
                    + ['-' if pd.isna(v) else f"{v:.2f}"
                       for v in (per_pair.loc[s, f'{r}_{AVG}'] for r in runs)])
    lines += md_table(['Pair', 'Group'] + [name(r) for r in runs], rows)
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
    table = {name(r): [per_pair[f'{r}_{c}'].mean() for c in columns] for r in runs}
    print(pd.DataFrame(table, index=win_labels).T.round(2).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
