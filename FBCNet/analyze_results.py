"""
Full analysis of the raw-vs-filter-bank sweep (28 pairs x 18 subjects).

    python analyze_results.py
    python analyze_results.py --results-root results --cwt-root ../morlet-scalogram/results

Everything is recomputed from predictions.csv (per-trial), so every metric is
traceable to the saved predictions. If the Morlet sweep's results are present, the
CWT arm is included as a third arm: its raw arm is bit-identical to this one (same
code path, same folds, same seeds, same GPU model), so all three arms are paired.

Writes <results-root>/analysis/*.csv and report_data.json, and prints the tables.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

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

    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return dict(acc=100 * acc, kappa=kappa, f1=0.5 * (f1(tp, fp, fn) + f1(tn, fn, fp)),
                sens=100 * sens, spec=100 * spec,
                pred1=100 * float((y_pred == 1).mean()),
                # one-sided binomial test vs chance (50 %) for this subject's trials
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
    """a - b, paired over matched units. Returns a dict of effect + tests."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    nz = d[d != 0]
    try:
        p_w = stats.wilcoxon(a, b).pvalue if len(nz) else 1.0
    except ValueError:
        p_w = 1.0
    ranks = stats.rankdata(np.abs(nz))
    rbc = ((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()) if len(nz) else 0.0
    sd = d.std(ddof=1)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=sd / np.sqrt(len(d)))
    return dict(delta=d.mean(), ci_lo=lo, ci_hi=hi, sd_delta=sd,
                dz=d.mean() / sd if sd else np.nan, rank_biserial=rbc,
                better=int((d > 0).sum()), worse=int((d < 0).sum()),
                tied=int((d == 0).sum()), p_wilcoxon=p_w,
                p_ttest=stats.ttest_rel(a, b).pvalue, n=len(d))


def load_arm(root, mode):
    frames = []
    mdir = os.path.join(root, mode)
    for slug in sorted(os.listdir(mdir)):
        f = os.path.join(mdir, slug, 'predictions.csv')
        if os.path.isfile(f):
            frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df['mode'] = mode
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', default=os.path.join(here, 'results'))
    ap.add_argument('--cwt-root', default=os.path.join(here, '../morlet-scalogram/results'))
    args = ap.parse_args()
    out_dir = os.path.join(args.results_root, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    arms = {'raw': load_arm(args.results_root, 'raw'),
            'fbc': load_arm(args.results_root, 'fbc')}
    has_cwt = os.path.isdir(os.path.join(args.cwt_root, 'cwt'))
    if has_cwt:
        cwt_raw = load_arm(args.cwt_root, 'raw')
        key = ['pair', 'subject', 'fold', 'trial', 'y_true']
        m = arms['raw'].merge(cwt_raw, on=key, suffixes=('', '_c'))
        identical = bool(len(m) == len(arms['raw'])
                         and (m.y_pred == m.y_pred_c).all())
        arms['cwt'] = load_arm(args.cwt_root, 'cwt')
    else:
        identical = None
    modes = list(arms)

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
    subjects = sorted(S.subject.unique())

    # ---------------- per pair ------------------------------------------------
    P = []
    for pair in pairs:
        a = acc.loc[pair]
        k = kap.loc[pair]
        r = dict(pair=pair, proximity=proximity(pair), joints=joints(pair),
                 paper_fine=FINE_PAPER[pair], shallow=SHALLOW[pair])
        for mode in modes:
            r[f'{mode}_acc'] = a[mode].mean()
            r[f'{mode}_sd'] = a[mode].std(ddof=1)
            r[f'{mode}_kappa'] = k[mode].mean()
            sub = S[(S.pair == pair) & (S['mode'] == mode)]
            r[f'{mode}_above_chance'] = int((sub.p_chance < 0.05).sum())
            r[f'{mode}_bias'] = float(np.abs(sub.pred1 - 50).mean())
        t = paired(a['fbc'], a['raw'])
        r.update({f'fbc_{key}': v for key, v in t.items()})
        r['fbc_dkappa'] = (k['fbc'] - k['raw']).mean()
        if has_cwt:
            t2 = paired(a['cwt'], a['raw'])
            r.update(cwt_delta=t2['delta'], cwt_p_wilcoxon=t2['p_wilcoxon'],
                     cwt_better=t2['better'], cwt_worse=t2['worse'])
            t3 = paired(a['fbc'], a['cwt'])
            r.update(fbc_vs_cwt=t3['delta'], fbc_vs_cwt_p=t3['p_wilcoxon'])
        P.append(r)
    P = pd.DataFrame(P)
    P['fbc_p_holm'] = holm(P.fbc_p_wilcoxon)
    P['fbc_p_fdr'] = bh(P.fbc_p_wilcoxon)
    if has_cwt:
        P['cwt_p_holm'] = holm(P.cwt_p_wilcoxon)
        P['cwt_p_fdr'] = bh(P.cwt_p_wilcoxon)
    P.to_csv(os.path.join(out_dir, 'per_pair.csv'), index=False)

    # ---------------- grand: subject mean over 28 pairs (n = 18) --------------
    subj_mean = acc.groupby(level='subject').mean()
    subj_kappa = kap.groupby(level='subject').mean()
    grand = {}
    for mode in modes:
        grand[mode] = dict(acc=subj_mean[mode].mean(), sd=subj_mean[mode].std(ddof=1),
                           kappa=subj_kappa[mode].mean())
    g_fbc = paired(subj_mean['fbc'], subj_mean['raw'])
    g_fbc_k = paired(subj_kappa['fbc'], subj_kappa['raw'])
    g_cwt = paired(subj_mean['cwt'], subj_mean['raw']) if has_cwt else None
    g_fvc = paired(subj_mean['fbc'], subj_mean['cwt']) if has_cwt else None

    # pairs as units (n = 28): does the typical pair move?
    pair_level = paired(P.fbc_acc, P.raw_acc)

    # ---------------- proximity + joint groups (subject-level, n = 18) --------
    G = []
    for col in ('proximity', 'joints'):
        for grp in sorted(S[col].unique()):
            ps = [p for p in pairs if (proximity(p) if col == 'proximity' else joints(p)) == grp]
            sub = acc.loc[ps].groupby(level='subject').mean()
            t = paired(sub['fbc'], sub['raw'])
            row = dict(grouping=col, group=grp, n_pairs=len(ps),
                       paper_fine=np.mean([FINE_PAPER[p] for p in ps]),
                       shallow=np.mean([SHALLOW[p] for p in ps]),
                       **{f'{m}_acc': sub[m].mean() for m in modes},
                       fbc_delta=t['delta'], fbc_ci_lo=t['ci_lo'], fbc_ci_hi=t['ci_hi'],
                       fbc_p=t['p_wilcoxon'], fbc_better=t['better'], fbc_worse=t['worse'])
            if has_cwt:
                t2 = paired(sub['cwt'], sub['raw'])
                row.update(cwt_delta=t2['delta'], cwt_p=t2['p_wilcoxon'])
            G.append(row)
    G = pd.DataFrame(G)
    G.to_csv(os.path.join(out_dir, 'groups.csv'), index=False)

    # ---------------- the proposal's specific hypotheses ----------------------
    losing5 = [p for p in pairs if FINE_PAPER[p] < SHALLOW[p]]
    hyp = {}
    for name, ps in [('same_joint', [p for p in pairs if proximity(p) == 'same']),
                     ('shoulder_shoulder', [p for p in pairs if joints(p) == 'shoulder-shoulder']),
                     ('wrist_wrist', [p for p in pairs if joints(p) == 'wrist-wrist']),
                     ('fine_loses_to_shallow', losing5),
                     ('hardest_7_by_raw', list(P.nsmallest(7, 'raw_acc').pair)),
                     ('easiest_7_by_raw', list(P.nlargest(7, 'raw_acc').pair))]:
        sub = acc.loc[ps].groupby(level='subject').mean()
        t = paired(sub['fbc'], sub['raw'])
        hyp[name] = dict(pairs=ps, raw=sub['raw'].mean(), fbc=sub['fbc'].mean(),
                         **{k: t[k] for k in ('delta', 'ci_lo', 'ci_hi', 'p_wilcoxon',
                                              'better', 'worse', 'dz')})
    # does the gain grow as the pair gets harder?  (regression to the mean caveat:
    # raw_acc is on both sides, so use the paper's FINE accuracy - independent data -
    # as the difficulty axis as well)
    r_raw = stats.spearmanr(P.raw_acc, P.fbc_delta)
    r_paper = stats.spearmanr(P.paper_fine, P.fbc_delta)
    hyp['difficulty_vs_gain'] = dict(spearman_raw=r_raw.statistic, p_raw=r_raw.pvalue,
                                     spearman_paper=r_paper.statistic,
                                     p_paper=r_paper.pvalue)

    # ---------------- per subject ---------------------------------------------
    SU = pd.DataFrame({f'{m}_acc': subj_mean[m] for m in modes})
    SU['fbc_delta'] = SU.fbc_acc - SU.raw_acc
    SU['fbc_pairs_better'] = [(acc.xs(s, level='subject').fbc
                               > acc.xs(s, level='subject').raw).sum() for s in SU.index]
    SU['fbc_pairs_worse'] = [(acc.xs(s, level='subject').fbc
                              < acc.xs(s, level='subject').raw).sum() for s in SU.index]
    SU['p_wilcoxon_28pairs'] = [paired(acc.xs(s, level='subject').fbc,
                                       acc.xs(s, level='subject').raw)['p_wilcoxon']
                                for s in SU.index]
    if has_cwt:
        SU['cwt_delta'] = SU.cwt_acc - SU.raw_acc
    SU.to_csv(os.path.join(out_dir, 'per_subject.csv'))
    r_subj = stats.spearmanr(SU.raw_acc, SU.fbc_delta)

    # ---------------- decoder behaviour ---------------------------------------
    beh = {}
    for mode in modes:
        sub = S[S['mode'] == mode]
        beh[mode] = dict(
            above_chance_cells=int((sub.p_chance < 0.05).sum()), cells=len(sub),
            mean_bias=float(np.abs(sub.pred1 - 50).mean()),
            degenerate_cells=int(((sub.pred1 < 20) | (sub.pred1 > 80)).sum()),
            mean_between_subject_sd=float(P[f'{mode}_sd'].mean()))
    # agreement between raw and fbc, per trial
    key = ['pair', 'subject', 'fold', 'trial', 'y_true']
    j = arms['raw'].merge(arms['fbc'], on=key, suffixes=('_r', '_f'))
    rc, fc = j.y_pred_r == j.y_true, j.y_pred_f == j.y_true
    agree = dict(agree=float((j.y_pred_r == j.y_pred_f).mean() * 100),
                 both_right=float((rc & fc).mean() * 100),
                 only_raw=float((rc & ~fc).mean() * 100),
                 only_fbc=float((~rc & fc).mean() * 100),
                 both_wrong=float((~rc & ~fc).mean() * 100),
                 n=len(j))
    # McNemar over all trials (exact binomial on the discordant cells)
    b_, c_ = int((rc & ~fc).sum()), int((~rc & fc).sum())
    agree['mcnemar_p'] = stats.binomtest(c_, b_ + c_, 0.5).pvalue
    # the agreement one would expect if the two decoders erred independently
    pr, pf = rc.mean(), fc.mean()
    agree['agree_if_independent'] = float((pr * pf + (1 - pr) * (1 - pf)) * 100)

    # ---------------- vs the paper --------------------------------------------
    paper = dict(
        raw_minus_paper=float((P.raw_acc - P.paper_fine).mean()),
        raw_minus_paper_sd=float((P.raw_acc - P.paper_fine).std(ddof=1)),
        spearman_raw_paper=stats.spearmanr(P.raw_acc, P.paper_fine).statistic,
        wins_vs_shallow={m: int((P[f'{m}_acc'] > P.shallow).sum()) for m in modes},
        paper_wins_vs_shallow=int(sum(FINE_PAPER[p] > SHALLOW[p] for p in pairs)),
        mean_vs_shallow={m: float((P[f'{m}_acc'] - P.shallow).mean()) for m in modes})

    # ---------------- power ---------------------------------------------------
    # smallest paired effect (dz) detectable with 80 % power, alpha .05 two-sided
    from scipy.optimize import brentq
    n = len(subjects)

    def pw(dz):
        nc = dz * np.sqrt(n)
        tc = stats.t.ppf(0.975, n - 1)
        return 1 - stats.nct.cdf(tc, n - 1, nc) + stats.nct.cdf(-tc, n - 1, nc)

    mde_dz = brentq(lambda d: pw(d) - 0.8, 0.01, 1.5)   # nct cdf is NaN far out
    mde_pp = mde_dz * g_fbc['sd_delta']

    report = dict(
        n_subjects=len(subjects), n_pairs=len(pairs), modes=modes,
        raw_bit_identical_across_sweeps=identical,
        grand=grand, grand_fbc=g_fbc, grand_fbc_kappa=g_fbc_k,
        grand_cwt=g_cwt, grand_fbc_vs_cwt=g_fvc, pair_level_fbc=pair_level,
        hypotheses=hyp, subject_skill_vs_gain=dict(spearman=r_subj.statistic,
                                                   p=r_subj.pvalue),
        behaviour=beh, agreement=agree, paper=paper,
        power=dict(mde_dz=mde_dz, mde_pp=mde_pp, sd_delta=g_fbc['sd_delta']),
        pairs=P.to_dict(orient='records'), groups=G.to_dict(orient='records'),
        subjects=SU.reset_index().to_dict(orient='records'))
    with open(os.path.join(out_dir, 'report_data.json'), 'w') as f:
        json.dump(report, f, indent=1, default=float)

    # ---------------- print ---------------------------------------------------
    pd.set_option('display.width', 200)
    print(f"raw arm bit-identical to the Morlet sweep's raw arm: {identical}\n")
    print("GRAND (subject mean over 28 pairs, n=18)")
    for m in modes:
        print(f"  {m:4s} acc {grand[m]['acc']:.2f}  sd {grand[m]['sd']:.2f}  "
              f"kappa {grand[m]['kappa']:.4f}")
    for name, t in [('fbc-raw', g_fbc), ('cwt-raw', g_cwt), ('fbc-cwt', g_fvc)]:
        if t:
            print(f"  {name}: {t['delta']:+.2f} pp [{t['ci_lo']:+.2f}, {t['ci_hi']:+.2f}]  "
                  f"better {t['better']}/18  p_w={t['p_wilcoxon']:.4f}  dz={t['dz']:.2f}")
    print(f"  fbc-raw kappa: {g_fbc_k['delta']:+.4f}  p_w={g_fbc_k['p_wilcoxon']:.4f}")
    print(f"  pair level (n=28): {pair_level['delta']:+.2f}, better "
          f"{pair_level['better']}/28, p_w={pair_level['p_wilcoxon']:.4f}")
    print(f"  power: MDE dz={mde_dz:.2f} = {mde_pp:.2f} pp at n={n}\n")

    cols = ['pair', 'proximity', 'paper_fine', 'shallow', 'raw_acc', 'fbc_acc',
            'fbc_delta', 'fbc_better', 'fbc_worse', 'fbc_p_wilcoxon', 'fbc_p_fdr']
    if has_cwt:
        cols += ['cwt_acc', 'cwt_delta', 'cwt_p_fdr']
    print(P[cols].round(3).to_string(index=False))
    print("\nGROUPS")
    print(G.round(3).to_string(index=False))
    print("\nHYPOTHESES")
    for k, v in hyp.items():
        print(f"  {k}: " + ", ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}"
                                     for a, b in v.items() if a != 'pairs'))
    print("\nPER SUBJECT")
    print(SU.round(2).to_string())
    print(f"  skill vs gain spearman {r_subj.statistic:.3f} p {r_subj.pvalue:.3f}")
    print("\nBEHAVIOUR", json.dumps(beh, indent=1))
    print("AGREEMENT", json.dumps(agree, indent=1))
    print("PAPER", json.dumps(paper, indent=1, default=float))
    print(f"\nwrote {out_dir}")


if __name__ == '__main__':
    main()
