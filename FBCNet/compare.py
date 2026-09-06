"""
Paired raw-vs-filter-bank comparison from saved results.

    python compare.py --results-root test_2pair

Both arms share fold indices, seeds, normalisation and augmentation draws, so every
(pair, subject, fold) cell is a matched pair and the comparison can be made paired
rather than between-groups.

Writes <results-root>/comparison.csv and prints the tables.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

# Table III of the FINE paper, for the two pairs under test (18 subjects, 4000 ms).
PAPER = {"HOC/SPS": 68.51, "SAA/SFE": 53.37}
SHALLOW = {"HOC/SPS": 65.49, "SAA/SFE": 56.02}


def binary_metrics(y_true, y_pred):
    """Accuracy, Cohen's kappa, macro-F1. kappa needs the prediction marginals, which
    is why predictions.csv is saved rather than accuracy alone."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / n if n else np.nan
    p_e = (((tp + fn) * (tp + fp)) + ((tn + fp) * (tn + fn))) / (n * n) if n else np.nan
    kappa = (acc - p_e) / (1 - p_e) if n and p_e < 1 else np.nan

    def f1_of(t, f_p, f_n):
        d = 2 * t + f_p + f_n
        return 2 * t / d if d else 0.0

    f1 = 0.5 * (f1_of(tp, fp, fn) + f1_of(tn, fn, fp))
    return acc * 100, kappa, f1


def load(results_root, modes=('raw', 'fbc')):
    rows = []
    for mode in modes:
        mdir = os.path.join(results_root, mode)
        if not os.path.isdir(mdir):
            continue
        for slug in sorted(os.listdir(mdir)):
            path = os.path.join(mdir, slug, 'predictions.csv')
            if os.path.isfile(path):
                rows.append(pd.read_csv(path))
    if not rows:
        raise SystemExit(f"no predictions.csv under {results_root}")
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', default='test_2pair')
    args = ap.parse_args()

    df = load(args.results_root)

    # ---- per (pair, mode, subject, fold): the matched unit -------------------
    fold_rows = []
    for (pair, mode, subj, fold), g in df.groupby(['pair', 'mode', 'subject', 'fold']):
        acc, kap, f1 = binary_metrics(g.y_true, g.y_pred)
        fold_rows.append(dict(pair=pair, mode=mode, subject=subj, fold=fold,
                              acc=acc, kappa=kap, f1=f1, n=len(g)))
    folds = pd.DataFrame(fold_rows)

    # ---- per (pair, mode, subject): trials pooled over folds, as the paper does ----
    subj_rows = []
    for (pair, mode, subj), g in df.groupby(['pair', 'mode', 'subject']):
        acc, kap, f1 = binary_metrics(g.y_true, g.y_pred)
        subj_rows.append(dict(pair=pair, mode=mode, subject=subj,
                              acc=acc, kappa=kap, f1=f1, n=len(g)))
    subs = pd.DataFrame(subj_rows)

    pairs = sorted(subs.pair.unique())
    n_subj = subs.subject.nunique()

    print("=" * 78)
    print(f"RAW vs FILTER BANK  |  {n_subj} subjects, 5-fold CV, 4000 ms window")
    print("=" * 78)

    # ---------------- per-subject table, side by side ----------------
    for pair in pairs:
        p = subs[subs.pair == pair]
        w = p.pivot(index='subject', columns='mode', values='acc').sort_index()
        if not {'raw', 'fbc'} <= set(w.columns):
            continue
        w['delta'] = w['fbc'] - w['raw']
        print(f"\n--- {pair} ---   (paper, 18 subj: FINE {PAPER.get(pair, float('nan')):.2f}"
              f" | ShallowConvNet {SHALLOW.get(pair, float('nan')):.2f})")
        print(w.round(2).to_string())
        print(f"{'mean':>7}  " + f"{w['fbc'].mean():>6.2f}  {w['raw'].mean():>6.2f}  "
              f"{w['delta'].mean():>6.2f}")

    # ---------------- headline comparison ----------------
    out = []
    for pair in pairs:
        p = subs[subs.pair == pair]
        w = p.pivot(index='subject', columns='mode', values='acc').sort_index()
        k = p.pivot(index='subject', columns='mode', values='kappa').sort_index()
        f = p.pivot(index='subject', columns='mode', values='f1').sort_index()
        if not {'raw', 'fbc'} <= set(w.columns):
            continue
        d = (w['fbc'] - w['raw']).to_numpy()

        # subject level (n = n_subj): the unit the paper reports
        t_s = stats.ttest_rel(w['fbc'], w['raw'])
        try:
            wil = stats.wilcoxon(w['fbc'], w['raw'])
            wil_p = wil.pvalue
        except ValueError:
            wil_p = np.nan

        # fold level (n = n_subj * 5): same splits in both arms, more power but the
        # folds within a subject are not independent - reported as a secondary check
        fp = folds[folds.pair == pair].pivot_table(index=['subject', 'fold'],
                                                   columns='mode', values='acc')
        t_f = stats.ttest_rel(fp['fbc'], fp['raw'])

        out.append(dict(
            pair=pair,
            raw_acc=w['raw'].mean(), fbc_acc=w['fbc'].mean(), delta=d.mean(),
            raw_sd=w['raw'].std(ddof=1), fbc_sd=w['fbc'].std(ddof=1),
            raw_kappa=k['raw'].mean(), fbc_kappa=k['fbc'].mean(),
            raw_f1=f['raw'].mean(), fbc_f1=f['fbc'].mean(),
            wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()),
            t_subj=t_s.statistic, p_subj=t_s.pvalue, p_wilcoxon=wil_p,
            p_fold=t_f.pvalue,
            cohen_dz=d.mean() / d.std(ddof=1) if d.std(ddof=1) else np.nan,
            paper_fine=PAPER.get(pair), paper_shallow=SHALLOW.get(pair)))

    res = pd.DataFrame(out)
    print("\n" + "=" * 78)
    print("SUMMARY  (accuracy %, mean over subjects; delta = fbc - raw)")
    print("=" * 78)
    show = res[['pair', 'raw_acc', 'fbc_acc', 'delta', 'raw_sd', 'fbc_sd',
                'wins', 'losses', 'p_subj', 'p_wilcoxon', 'p_fold', 'cohen_dz']]
    print(show.round(3).to_string(index=False))

    print("\nkappa / macro-F1")
    print(res[['pair', 'raw_kappa', 'fbc_kappa', 'raw_f1', 'fbc_f1']]
          .round(4).to_string(index=False))

    print("\nvs the paper (18 subjects - this run uses %d, so these are NOT "
          "directly comparable)" % n_subj)
    print(res[['pair', 'paper_fine', 'paper_shallow', 'raw_acc', 'fbc_acc']]
          .round(2).to_string(index=False))

    path = os.path.join(args.results_root, 'comparison.csv')
    res.to_csv(path, index=False)
    folds.to_csv(os.path.join(args.results_root, 'per_fold_metrics.csv'), index=False)
    subs.to_csv(os.path.join(args.results_root, 'per_subject_metrics.csv'), index=False)
    print(f"\nwrote {path}")


if __name__ == '__main__':
    main()
