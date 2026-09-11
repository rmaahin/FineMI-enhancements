"""
Full analysis of the Riemannian sweep: every arm vs the raw FINE baseline, paired.

    python analyze_results.py
    python analyze_results.py --results-root results --no-extra

Everything is recomputed from predictions.csv (per trial), so every number traces back to
saved predictions. All arms share fold indices and seeds, so each (pair, subject) cell is
a matched unit across arms. If the FBCNet (fbc) and Morlet (cwt) sweeps are present
they are pulled in as extra arms - their raw arms are the same code path, so everything
is paired against one baseline (the script checks the raw arms really are identical).

Writes <results-root>/analysis/*.csv + report_data.json, and prints the tables.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# sd / CI of a partially finished sweep (one pair, one subject) is NaN by design
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Table III of the FINE paper (18 subjects, 4000 ms).
SHALLOW = {
    "HOC/WFE": 55.90, "HOC/WAA": 56.88, "HOC/EPS": 56.32, "HOC/EFE": 61.18,
    "HOC/SPS": 65.49, "HOC/SAA": 63.13, "HOC/SFE": 63.37, "WFE/WAA": 56.88,
    "WFE/EPS": 56.25, "WFE/EFE": 56.60, "WFE/SPS": 61.81, "WFE/SAA": 59.44,
    "WFE/SFE": 61.14, "WAA/EPS": 52.22, "WAA/EFE": 57.85, "WAA/SPS": 59.72,
    "WAA/SAA": 60.56, "WAA/SFE": 56.83, "EPS/EFE": 53.89, "EPS/SPS": 59.24,
    "EPS/SAA": 55.69, "EPS/SFE": 58.97, "EFE/SPS": 58.61, "EFE/SAA": 56.88,
    "EFE/SFE": 58.23, "SPS/SAA": 57.36, "SPS/SFE": 59.97, "SAA/SFE": 56.02,
}
FINE_PAPER = {
    "HOC/WFE": 56.41, "HOC/WAA": 60.32, "HOC/EPS": 57.97, "HOC/EFE": 62.23,
    "HOC/SPS": 68.51, "HOC/SAA": 67.85, "HOC/SFE": 65.17, "WFE/WAA": 60.12,
    "WFE/EPS": 54.95, "WFE/EFE": 57.38, "WFE/SPS": 63.35, "WFE/SAA": 65.01,
    "WFE/SFE": 61.66, "WAA/EPS": 55.41, "WAA/EFE": 59.12, "WAA/SPS": 63.85,
    "WAA/SAA": 63.06, "WAA/SFE": 63.57, "EPS/EFE": 56.30, "EPS/SPS": 62.25,
    "EPS/SAA": 62.95, "EPS/SFE": 61.68, "EFE/SPS": 58.83, "EFE/SAA": 58.73,
    "EFE/SFE": 58.11, "SPS/SAA": 55.50, "SPS/SFE": 57.08, "SAA/SFE": 53.37,
}
JOINT_OF = {"HOC": "hand", "WFE": "wrist", "WAA": "wrist", "EPS": "elbow",
            "EFE": "elbow", "SPS": "shoulder", "SAA": "shoulder", "SFE": "shoulder"}
DIST = {frozenset(("hand", "wrist")): "adjacent", frozenset(("hand", "elbow")): "adjacent",
        frozenset(("wrist", "elbow")): "adjacent",
        frozenset(("elbow", "shoulder")): "adjacent",
        frozenset(("hand", "shoulder")): "distant",
        frozenset(("wrist", "shoulder")): "distant"}
ARM_ORDER = ['raw', 'ts', 'tsimg', 'fbts', 'fbtsimg', 'mdrm', 'tslda', 'fbc', 'cwt']


def proximity(pair):
    a, b = (JOINT_OF[x] for x in pair.split("/"))
    return "same" if a == b else DIST[frozenset((a, b))]


def joints(pair):
    a, b = (JOINT_OF[x] for x in pair.split("/"))
    return "-".join(sorted((a, b), key=["hand", "wrist", "elbow", "shoulder"].index))


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / n
    p_e = (((tp + fn) * (tp + fp)) + ((tn + fp) * (tn + fn))) / (n * n)
    kappa = (acc - p_e) / (1 - p_e) if p_e < 1 else np.nan

    def f1(t, a, b):
        d = 2 * t + a + b
        return 2 * t / d if d else 0.0

    return dict(acc=100 * acc, kappa=kappa, f1=0.5 * (f1(tp, fp, fn) + f1(tn, fn, fp)),
                pred1=100 * float((y_pred == 1).mean()),
                p_chance=stats.binomtest(tp + tn, n, 0.5, alternative='greater').pvalue,
                n=n)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for rank, i in enumerate(order):
        running = min(running, p[i] * m / (m - rank))
        adj[i] = running
    return adj


def paired(a, b):
    """a - b over matched units: effect size, 95 % CI and Wilcoxon / t tests."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    nz = d[d != 0]
    try:
        p_w = stats.wilcoxon(a, b).pvalue if len(nz) else 1.0
    except ValueError:
        p_w = 1.0
    sd = d.std(ddof=1)
    lo, hi = (stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=sd / np.sqrt(len(d)))
              if sd > 0 else (d.mean(), d.mean()))
    return dict(delta=d.mean(), ci_lo=lo, ci_hi=hi, sd_delta=sd,
                dz=d.mean() / sd if sd else np.nan,
                better=int((d > 0).sum()), worse=int((d < 0).sum()),
                tied=int((d == 0).sum()), p_wilcoxon=p_w,
                p_ttest=stats.ttest_rel(a, b).pvalue if sd > 0 else 1.0, n=len(d))


def load_arm(root, mode):
    mdir = os.path.join(root, mode)
    if not os.path.isdir(mdir):
        return None
    frames = [pd.read_csv(os.path.join(mdir, s, 'predictions.csv'))
              for s in sorted(os.listdir(mdir))
              if os.path.isfile(os.path.join(mdir, s, 'predictions.csv'))]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df['mode'] = mode
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', default=os.path.join(here, 'results'))
    ap.add_argument('--fbc-root', default=os.path.join(here, '../FBCNet/results'))
    ap.add_argument('--cwt-root', default=os.path.join(here, '../morlet-scalogram/results'))
    ap.add_argument('--no-extra', action='store_true',
                    help='do not pull in the FBCNet / Morlet arms')
    args = ap.parse_args()
    out_dir = os.path.join(args.results_root, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- load ----------------------------------------------------
    arms = {}
    for mode in ARM_ORDER[:7]:
        df = load_arm(args.results_root, mode)
        if df is not None:
            arms[mode] = df
    raw_src = args.results_root
    if 'raw' not in arms:                       # fall back to a sibling sweep's raw arm
        for root in (args.fbc_root, args.cwt_root):
            df = load_arm(root, 'raw')
            if df is not None:
                arms['raw'], raw_src = df, root
                break
    if 'raw' not in arms:
        raise SystemExit("no raw arm found - run `run_local.py --mode raw` (or the FBCNet "
                         "sweep) first; every comparison is against it")

    key = ['pair', 'subject', 'fold', 'trial', 'y_true']
    raw_identical = {}
    if not args.no_extra:
        for root, mode in ((args.fbc_root, 'fbc'), (args.cwt_root, 'cwt')):
            df = load_arm(root, mode)
            if df is None:
                continue
            arms[mode] = df
            other_raw = load_arm(root, 'raw')
            if other_raw is not None and os.path.abspath(root) != os.path.abspath(raw_src):
                m = arms['raw'].merge(other_raw, on=key, suffixes=('', '_o'))
                raw_identical[mode] = bool(len(m) == len(arms['raw'])
                                           and (m.y_pred == m.y_pred_o).all())
    # sibling channel-set sweeps written by run_riemannian.sh: results_<set>/<arm>/...
    base = os.path.abspath(args.results_root).rstrip(os.sep)
    parent, stem = os.path.dirname(base), os.path.basename(base)
    set_arms = []
    for d in sorted(os.listdir(parent)):
        full = os.path.join(parent, d)
        if not (d.startswith(stem + '_') and os.path.isdir(full)):
            continue
        cs = d[len(stem) + 1:]
        for mode in ARM_ORDER[1:7]:
            df = load_arm(full, mode)
            if df is not None:
                df['mode'] = f"{mode}@{cs}"
                arms[f"{mode}@{cs}"] = df
                set_arms.append(f"{mode}@{cs}")

    modes = [m for m in ARM_ORDER if m in arms] + set_arms
    others = [m for m in modes if m != 'raw']
    print(f"arms: {', '.join(modes)}   (raw from {raw_src})")
    for m, ok in raw_identical.items():
        print(f"  {m} sweep's raw arm identical to this baseline: {ok}"
              + ("" if ok else "  <- different GPU/torch; its pairing is approximate"))

    # ---------------- per (pair, mode, subject) -------------------------------
    rows = []
    for mode, df in arms.items():
        for (pair, subj), g in df.groupby(['pair', 'subject']):
            rows.append(dict(pair=pair, mode=mode, subject=int(subj),
                             **metrics(g.y_true, g.y_pred)))
    S = pd.DataFrame(rows)
    S['proximity'] = S.pair.map(proximity)
    S['joints'] = S.pair.map(joints)
    S.to_csv(os.path.join(out_dir, 'per_subject_pair_metrics.csv'), index=False)

    acc = S.pivot_table(index=['pair', 'subject'], columns='mode', values='acc')
    kap = S.pivot_table(index=['pair', 'subject'], columns='mode', values='kappa')
    pairs = sorted(S.pair.unique(), key=lambda p: -FINE_PAPER[p])

    # ---------------- per pair: every arm vs raw ------------------------------
    P = []
    for pair in pairs:
        a = acc.loc[pair]
        r = dict(pair=pair, proximity=proximity(pair), joints=joints(pair),
                 paper_fine=FINE_PAPER[pair], shallow=SHALLOW[pair])
        for mode in modes:
            if mode in a and a[mode].notna().any():
                r[f'{mode}_acc'] = a[mode].mean()
                r[f'{mode}_sd'] = a[mode].std(ddof=1)
        for mode in others:
            if mode not in a:
                continue
            sub = a[['raw', mode]].dropna()
            if len(sub) < 3:
                continue
            t = paired(sub[mode], sub['raw'])
            r.update({f'{mode}_delta': t['delta'], f'{mode}_p': t['p_wilcoxon'],
                      f'{mode}_better': t['better'], f'{mode}_worse': t['worse']})
        P.append(r)
    P = pd.DataFrame(P)
    for mode in others:
        if f'{mode}_p' in P:
            ok = P[f'{mode}_p'].notna()
            P.loc[ok, f'{mode}_p_fdr'] = bh(P.loc[ok, f'{mode}_p'])
            P.loc[ok, f'{mode}_p_holm'] = holm(P.loc[ok, f'{mode}_p'])
    P.to_csv(os.path.join(out_dir, 'per_pair.csv'), index=False)

    # ---------------- grand: subject mean over pairs (n = 18) -----------------
    # Only pairs every compared arm has, so each arm is averaged over the same pairs.
    G = []
    for mode in modes:
        if mode == 'raw':
            sub = acc[['raw']].dropna()
        else:
            sub = acc[['raw', mode]].dropna()
        n_pairs = sub.index.get_level_values('pair').nunique()
        sm = sub.groupby(level='subject').mean()
        km = kap.loc[sub.index].groupby(level='subject').mean()
        row = dict(arm=mode, n_pairs=n_pairs, n_subjects=len(sm),
                   acc=sm[mode].mean(), sd=sm[mode].std(ddof=1), kappa=km[mode].mean())
        if mode != 'raw':
            t = paired(sm[mode], sm['raw'])
            tk = paired(km[mode], km['raw'])
            pm = sub.groupby(level='pair').mean()
            tp = paired(pm[mode], pm['raw'])
            row.update(raw_acc_same_pairs=sm['raw'].mean(), delta=t['delta'],
                       ci_lo=t['ci_lo'], ci_hi=t['ci_hi'], dz=t['dz'],
                       subjects_better=t['better'], subjects_worse=t['worse'],
                       p_wilcoxon=t['p_wilcoxon'], p_ttest=t['p_ttest'],
                       delta_kappa=tk['delta'], p_kappa=tk['p_wilcoxon'],
                       pairs_better=tp['better'], pairs_worse=tp['worse'],
                       p_pairs=tp['p_wilcoxon'])
        G.append(row)
    G = pd.DataFrame(G)
    cmp_ = G.arm != 'raw'
    if cmp_.any():
        G.loc[cmp_, 'p_holm_arms'] = holm(G.loc[cmp_, 'p_wilcoxon'])
    G.to_csv(os.path.join(out_dir, 'arms_summary.csv'), index=False)

    # ---------------- groups: proximity + joints (subject level) --------------
    GR = []
    for col in ('proximity', 'joints'):
        for grp in sorted(S[col].unique()):
            ps = [p for p in pairs if (proximity(p) if col == 'proximity'
                                       else joints(p)) == grp]
            sub = acc.loc[acc.index.get_level_values('pair').isin(ps)]
            sm = sub.groupby(level='subject').mean()
            row = dict(grouping=col, group=grp, n_pairs=len(ps),
                       paper_fine=np.mean([FINE_PAPER[p] for p in ps]),
                       shallow=np.mean([SHALLOW[p] for p in ps]))
            for mode in modes:
                if mode in sm and sm[mode].notna().all():
                    row[f'{mode}_acc'] = sm[mode].mean()
                    if mode != 'raw':
                        t = paired(sm[mode], sm['raw'])
                        row[f'{mode}_delta'] = t['delta']
                        row[f'{mode}_p'] = t['p_wilcoxon']
            GR.append(row)
    GR = pd.DataFrame(GR)
    GR.to_csv(os.path.join(out_dir, 'groups.csv'), index=False)

    # ---------------- the proposal's hypothesis: same-joint pairs -------------
    # "Same-limb tasks differ mainly in how motor cortex regions co-activate" - if the
    # covariance features help anywhere it should be here.
    same = [p for p in pairs if proximity(p) == 'same']
    hyp = {}
    for mode in others:
        sub = acc.loc[acc.index.get_level_values('pair').isin(same), ['raw', mode]].dropna()
        if sub.empty:
            continue
        sm = sub.groupby(level='subject').mean()
        hyp[mode] = dict(pairs=same, **{k: v for k, v in paired(sm[mode], sm['raw']).items()
                                        if k in ('delta', 'ci_lo', 'ci_hi', 'p_wilcoxon',
                                                 'better', 'worse', 'dz')},
                         raw=sm['raw'].mean(), arm=sm[mode].mean())

    # ---------------- decoder behaviour ---------------------------------------
    beh = {}
    for mode in modes:
        sub = S[S['mode'] == mode]
        beh[mode] = dict(above_chance_cells=int((sub.p_chance < 0.05).sum()),
                         cells=len(sub), mean_bias=float(np.abs(sub.pred1 - 50).mean()),
                         degenerate_cells=int(((sub.pred1 < 20) | (sub.pred1 > 80)).sum()))

    # wide table for a quick look
    wide = P[['pair', 'proximity', 'paper_fine', 'shallow']
             + [f'{m}_acc' for m in modes if f'{m}_acc' in P]]
    wide.to_csv(os.path.join(out_dir, 'comparison.csv'), index=False)

    report = dict(arms=modes, raw_source=raw_src, raw_identical=raw_identical,
                  grand=G.to_dict(orient='records'), same_joint=hyp, behaviour=beh,
                  pairs=P.to_dict(orient='records'), groups=GR.to_dict(orient='records'))
    with open(os.path.join(out_dir, 'report_data.json'), 'w') as f:
        json.dump(report, f, indent=1, default=float)

    # ---------------- print ---------------------------------------------------
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 60)
    print("\nGRAND  (subject mean over pairs, n = subjects; delta = arm - raw on the same pairs)")
    show = [c for c in ['arm', 'n_pairs', 'acc', 'sd', 'kappa', 'delta', 'ci_lo', 'ci_hi',
                        'subjects_better', 'p_wilcoxon', 'p_holm_arms', 'pairs_better',
                        'p_pairs', 'dz'] if c in G]
    print(G[show].round(3).to_string(index=False))

    print("\nPER PAIR  (accuracy %, mean over subjects)")
    print(wide.round(2).to_string(index=False))

    print("\nPER PAIR  delta vs raw  (* = BH-FDR < .05 across the 28 pairs of that arm)")
    dcols = ['pair']
    for m in others:
        if f'{m}_delta' in P:
            P[f'{m}'] = ["" if np.isnan(d) else f"{d:+6.2f}{'*' if q < .05 else ' '}"
                         for d, q in zip(P[f'{m}_delta'], P[f'{m}_p_fdr'].fillna(1))]
            dcols.append(m)
    print(P[dcols].to_string(index=False))

    print("\nGROUPS  (subject-level)")
    print(GR[[c for c in GR if c.endswith('_acc') or c.endswith('_delta')
              or c in ('grouping', 'group', 'n_pairs')]].round(2).to_string(index=False))

    print("\nSAME-JOINT PAIRS  (the proposal's hypothesis)")
    for mode, h in hyp.items():
        print(f"  {mode:8s} raw {h['raw']:.2f} -> {h['arm']:.2f}  delta {h['delta']:+.2f} "
              f"[{h['ci_lo']:+.2f}, {h['ci_hi']:+.2f}]  better {h['better']}/{h['better'] + h['worse']}"
              f"  p_w={h['p_wilcoxon']:.4f}")

    print("\nBEHAVIOUR")
    for mode, b in beh.items():
        print(f"  {mode:8s} above-chance cells {b['above_chance_cells']}/{b['cells']}  "
              f"mean |bias| {b['mean_bias']:.1f} pp  degenerate {b['degenerate_cells']}")
    print(f"\nwrote {out_dir}")


if __name__ == '__main__':
    main()
