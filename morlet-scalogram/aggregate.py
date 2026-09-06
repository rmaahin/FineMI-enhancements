"""
Build the FINE-raw vs FINE-CWT vs ShallowConvNet comparison from saved results.

    python aggregate.py
    python aggregate.py --results-root mi_results_v2

Writes, under <results-root>/:
    comparison_table.csv    per pair: both accuracy definitions, kappa/F1, delta, stats
    per_pair_summary.csv    the paper's Fig. 2 view  (pairs sorted, sd across subjects)
    per_subject_summary.csv the paper's Fig. 3 view  (subjects, sd across task pairs)
    metrics_long.csv        tidy per (pair, mode, subject) metrics

Safe to run mid-sweep - it reports whatever exists so far.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

# ---------------------------------------------------------------- paper reference
# Table III of the FINE paper. Verified: these two dicts reproduce every aggregate the
# paper states about itself - same/adjacent/distant = 56.47/59.02/64.67, 23/28 wins,
# max gain +7.26 pp on EPS/SAA, mean underperformance 1.76 pp on the 5 losing pairs.
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
DIST = {("hand", "wrist"): "adjacent", ("hand", "elbow"): "adjacent",
        ("wrist", "elbow"): "adjacent", ("elbow", "shoulder"): "adjacent",
        ("hand", "shoulder"): "distant", ("wrist", "shoulder"): "distant"}


def proximity(pair):
    a, b = (JOINT_OF[x] for x in pair.split("/"))
    return "same" if a == b else DIST.get((a, b), DIST.get((b, a), "adjacent"))


# ---------------------------------------------------------------- metrics
def binary_metrics(y_true, y_pred):
    """Accuracy, Cohen's kappa, macro-F1, sensitivity, specificity + confusion counts.

    kappa is NOT derivable from accuracy alone - it depends on the marginal
    distribution of the predictions - which is why predictions.csv has to be saved.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / n if n else np.nan

    p_e = (((tp + fn) * (tp + fp)) + ((tn + fp) * (tn + fn))) / (n * n) if n else np.nan
    kappa = (acc - p_e) / (1 - p_e) if n and p_e < 1 else np.nan

    def f1_of(t, f_p, f_n):
        denom = 2 * t + f_p + f_n
        return 2 * t / denom if denom else 0.0

    f1 = 0.5 * (f1_of(tp, fp, fn) + f1_of(tn, fn, fp))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(accuracy=acc * 100, kappa=kappa, f1_macro=f1 * 100,
                sensitivity=sens * 100, specificity=spec * 100,
                tp=tp, tn=tn, fp=fp, fn=fn, n=n)


# ---------------------------------------------------------------- loading
def load_mode(results_root, mode):
    """-> {pair: {'wide': Series, 'fold': DataFrame, 'pred': DataFrame|None}}"""
    out = {}
    root = os.path.join(results_root, mode)
    if not os.path.isdir(root):
        return out
    for slug in sorted(os.listdir(root)):
        d = os.path.join(root, slug)
        wide_p = os.path.join(d, "wide_subject_x_window.csv")
        if not os.path.isfile(wide_p):
            continue
        pair = slug.replace("_", "/", 1)
        entry = {'wide': pd.read_csv(wide_p, index_col=0).iloc[:, 0]}
        fold_p = os.path.join(d, "per_fold.csv")
        entry['fold'] = pd.read_csv(fold_p) if os.path.isfile(fold_p) else None
        pred_p = os.path.join(d, "predictions.csv")
        entry['pred'] = pd.read_csv(pred_p) if os.path.isfile(pred_p) else None
        out[pair] = entry
    return out


def subject_metrics(entry):
    """-> DataFrame indexed by subject with accuracy_pooled / accuracy_foldmean / kappa / ..."""
    rows = {}
    if entry['pred'] is not None:
        for sid, g in entry['pred'].groupby('subject'):
            rows[int(sid)] = binary_metrics(g['y_true'].values, g['y_pred'].values)
    df = pd.DataFrame.from_dict(rows, orient='index') if rows else pd.DataFrame()
    if not df.empty:
        df = df.rename(columns={'accuracy': 'accuracy_pooled'})
    else:
        df = pd.DataFrame(index=entry['wide'].index)
        df['accuracy_pooled'] = entry['wide']
    # the paper's definition: "Performance was averaged across all folds"
    if entry['fold'] is not None:
        fm = entry['fold'].groupby('subject')['accuracy'].mean()
        df['accuracy_foldmean'] = fm.reindex(df.index)
    df.index.name = 'subject'
    return df.sort_index()


# ---------------------------------------------------------------- stats helpers
def holm(pvals):
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full(p.shape, np.nan)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * p[ok][i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full(p.shape, np.nan)
    vals = p[ok]
    m = len(vals)
    order = np.argsort(vals)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, vals[i] * m / (rank + 1))
        adj[i] = min(1.0, prev)
    out[ok] = adj
    return out


def rank_biserial(diff):
    """Effect size for the paired Wilcoxon: (favourable - unfavourable) rank mass."""
    d = np.asarray(diff, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    total = r.sum()
    return float((r[d > 0].sum() - r[d < 0].sum()) / total)


def paired_test(c, r):
    d = (c - r).values
    out = {'delta': float(np.mean(d)),
           'n_better': int((d > 0).sum()), 'n_worse': int((d < 0).sum()),
           'n_tied': int((d == 0).sum()),
           'cohen_dz': float(np.mean(d) / np.std(d, ddof=1)) if np.std(d, ddof=1) > 0 else np.nan,
           'rank_biserial': rank_biserial(d)}
    try:
        out['p'] = float(wilcoxon(c.values, r.values).pvalue)
    except ValueError:                       # all differences zero
        out['p'] = 1.0
    return out


# ---------------------------------------------------------------- main table
def build_table(results_root):
    raw, cwt = load_mode(results_root, 'raw'), load_mode(results_root, 'cwt')
    print(f"loaded raw={len(raw)} pairs, cwt={len(cwt)} pairs")
    if not raw and not cwt:
        return None, None

    per_subj = {}
    for mode, store in (('raw', raw), ('cwt', cwt)):
        for pair, entry in store.items():
            per_subj[(mode, pair)] = subject_metrics(entry)

    rows, long_rows = [], []
    for pair in sorted(set(raw) | set(cwt)):
        rec = dict(pair=pair, proximity=proximity(pair),
                   shallow=SHALLOW.get(pair), fine_paper=FINE_PAPER.get(pair))
        mr = per_subj.get(('raw', pair))
        mc = per_subj.get(('cwt', pair))
        for tag, m in (('raw', mr), ('cwt', mc)):
            if m is None:
                continue
            for col, out in (('accuracy_pooled', 'acc'), ('accuracy_foldmean', 'accM'),
                             ('kappa', 'kappa'), ('f1_macro', 'f1'),
                             ('sensitivity', 'sens'), ('specificity', 'spec')):
                if col in m:
                    rec[f'{tag}_{out}'] = round(float(m[col].mean()), 4)
            rec[f'{tag}_sd'] = round(float(m['accuracy_pooled'].std(ddof=1)), 2)
            for sid, r_ in m.iterrows():
                long_rows.append(dict(pair=pair, proximity=proximity(pair), mode=tag,
                                      subject=int(sid), **r_.to_dict()))
        if mr is not None and mc is not None:
            common = mr.index.intersection(mc.index)
            rec['n_subj'] = len(common)
            rec.update(paired_test(mc.loc[common, 'accuracy_pooled'],
                                   mr.loc[common, 'accuracy_pooled']))
            if 'kappa' in mr and 'kappa' in mc:
                rec['delta_kappa'] = round(float((mc.loc[common, 'kappa']
                                                  - mr.loc[common, 'kappa']).mean()), 4)
        else:
            rec['n_subj'] = len(mr if mr is not None else mc)
        rows.append(rec)

    tab = pd.DataFrame(rows)
    if 'p' in tab and tab['p'].notna().any():
        tab['p_holm'] = holm(tab['p'].values)
        tab['p_fdr'] = benjamini_hochberg(tab['p'].values)
    if 'acc' in ''.join(tab.columns):
        sort_key = 'cwt_acc' if 'cwt_acc' in tab else 'raw_acc'
        tab = tab.sort_values(sort_key, ascending=False).reset_index(drop=True)
    return tab, pd.DataFrame(long_rows)


def report(tab, long_df, results_root):
    show = [c for c in ['pair', 'proximity', 'n_subj', 'shallow', 'fine_paper',
                        'raw_acc', 'cwt_acc', 'delta', 'n_better', 'cohen_dz',
                        'p', 'p_fdr'] if c in tab]
    print("\n=== PER-PAIR (accuracy %, pooled over folds) ===")
    print(tab[show].round(3).to_string(index=False))

    if 'raw_kappa' in tab or 'cwt_kappa' in tab:
        kcols = [c for c in ['pair', 'raw_acc', 'raw_kappa', 'raw_f1', 'raw_sens',
                             'raw_spec', 'cwt_acc', 'cwt_kappa', 'cwt_f1',
                             'cwt_sens', 'cwt_spec'] if c in tab]
        print("\n=== EXTRA METRICS (not reported in the paper) ===")
        print(tab[kcols].round(3).to_string(index=False))

    if 'raw_acc' in tab and 'raw_accM' in tab:
        d = (tab['raw_acc'] - tab['raw_accM']).abs()
        print(f"\naccuracy definition check: pooled vs mean-of-folds differs by at most "
              f"{d.max():.4f} pp across pairs (paper uses mean-of-folds)")

    if 'delta' in tab and tab['delta'].notna().any():
        d = tab.dropna(subset=['delta'])
        print("\n=== CWT - RAW by joint proximity ===")
        print(d.groupby('proximity')['delta'].agg(['count', 'mean', 'std'])
              .round(3).to_string())
        print(f"\noverall: CWT beats raw on {int((d['delta'] > 0).sum())}/{len(d)} pairs, "
              f"mean delta {d['delta'].mean():+.3f} pp")
        if len(d) > 2:
            st = paired_test(d['cwt_acc'], d['raw_acc'])
            print(f"pair-level paired Wilcoxon: p={st['p']:.4g}, "
                  f"dz={st['cohen_dz']:.3f}, rank-biserial={st['rank_biserial']:.3f}")
        if 'p_fdr' in d:
            print(f"per-pair significance: {int((d['p'] < .05).sum())} uncorrected, "
                  f"{int((d['p_fdr'] < .05).sum())} after BH-FDR, "
                  f"{int((d['p_holm'] < .05).sum())} after Holm  (of {len(d)})")

    for col, label in (('shallow', 'ShallowConvNet'), ('fine_paper', 'FINE (paper)')):
        for arm in ('raw_acc', 'cwt_acc'):
            if arm not in tab:
                continue
            sub = tab.dropna(subset=[col, arm])
            if len(sub):
                print(f"{arm[:3]:3s} beats {label:15s} on "
                      f"{int((sub[arm] > sub[col]).sum())}/{len(sub)} pairs "
                      f"(mean {sub[arm].mean() - sub[col].mean():+.2f} pp)")

    os.makedirs(results_root, exist_ok=True)
    tab.to_csv(os.path.join(results_root, "comparison_table.csv"), index=False)

    # Fig. 2 view: per pair, mean and sd ACROSS SUBJECTS, sorted descending
    if long_df is not None and not long_df.empty:
        fig2 = (long_df.groupby(['mode', 'pair', 'proximity'])['accuracy_pooled']
                .agg(mean='mean', sd=lambda s: s.std(ddof=1), n='size')
                .reset_index().sort_values(['mode', 'mean'], ascending=[True, False]))
        fig2.round(3).to_csv(os.path.join(results_root, "per_pair_summary.csv"), index=False)

        # Fig. 3 view: per subject, mean and sd ACROSS TASK PAIRS
        fig3 = (long_df.groupby(['mode', 'subject'])['accuracy_pooled']
                .agg(mean='mean', sd=lambda s: s.std(ddof=1), n_pairs='size')
                .reset_index())
        fig3.round(3).to_csv(os.path.join(results_root, "per_subject_summary.csv"),
                             index=False)
        long_df.round(4).to_csv(os.path.join(results_root, "metrics_long.csv"), index=False)

        print("\n=== PER-SUBJECT across task pairs (the paper's Fig. 3) ===")
        print(fig3.round(2).to_string(index=False))
        print(f"\nwrote comparison_table.csv, per_pair_summary.csv, "
              f"per_subject_summary.csv, metrics_long.csv -> {results_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', default=None)
    args = ap.parse_args()
    root = args.results_root or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'mi_results_v2')
    tab, long_df = build_table(root)
    if tab is None:
        print(f"no results under {root} yet")
        return
    report(tab, long_df, root)


if __name__ == '__main__':
    main()
