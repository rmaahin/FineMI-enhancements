"""Architecture x pair comparison table (no torch needed), decimated 50 Hz runs.

Reads <mode_root>/<model>/<PAIR_SLUG>/wide_subject_x_window.csv for every
selected (pair, model) and writes to <mode_root>/comparison/:

  comparison_long.csv        pair, model, window, mean_acc, sd_acc, n_subjects
  comparison_mean.csv        rows pair x model, columns windows + 'avg' (mean accuracy)
  comparison_sd.csv          same layout, across-subject SD
  comparison_pooled.csv      model x window pooled over pairs (see below), plus how many
                             pairs each model is best on
  comparison_table.md        pooled table + per-pair mean (SD); best architecture in bold
  all_subjects_long.csv      every per-subject accuracy with a model column (for stats later)

Stats are across subjects (ddof=1), matching the per-pair "mean"/"sd" printout.
'avg' is each subject's mean over windows, then mean/SD across subjects.
Pooled = each subject's mean over the pairs that ALL selected models have completed,
then mean/SD across subjects (subjects are not independent across pairs, so they
are averaged rather than treated as extra samples).
"""
from __future__ import annotations   # `X | None` annotations on Python < 3.10

import os

import numpy as np
import pandas as pd

from src.config import MODELS, MODEL_LABELS, PAIRS, pair_slug
from src.utils import is_pair_done

AVG_COL = 'avg'


def discover_pairs(mode_root: str, models=MODELS) -> list[tuple[int, int]]:
    """Every pair with a completed results.json under any of `models`, in PAIRS order."""
    found = []
    for a, b in PAIRS:
        slug = pair_slug(a, b)
        if any(is_pair_done(os.path.join(mode_root, m, slug)) for m in models):
            found.append((a, b))
    return found


def _read_wide(pair_dir: str) -> pd.DataFrame:
    wide = pd.read_csv(os.path.join(pair_dir, 'wide_subject_x_window.csv'), index_col=0)
    wide.columns = [int(c) for c in wide.columns]
    return wide


def _fmt(mean, sd):
    if pd.isna(mean):
        return '-'
    return f"{mean:.2f}" if pd.isna(sd) else f"{mean:.2f} ({sd:.2f})"


def build_comparison(mode_root: str, pairs, models=MODELS) -> pd.DataFrame | None:
    """Write the comparison files; returns the mean table (or None if nothing is done)."""
    rows, subj_frames, missing = [], [], []
    windows = []
    wides = {}   # (slug, model) -> subjects x [windows..., avg], str columns

    for a, b in pairs:
        slug = pair_slug(a, b)
        for m in models:
            pair_dir = os.path.join(mode_root, m, slug)
            if not is_pair_done(pair_dir):
                missing.append(f"{slug} [{m}]")
                continue
            wide = _read_wide(pair_dir)
            windows.extend(w for w in wide.columns if w not in windows)

            per_window = {w: wide[w] for w in wide.columns}
            per_window[AVG_COL] = wide.mean(axis=1)
            wides[(slug, m)] = pd.DataFrame({str(w): col for w, col in per_window.items()})
            for w, col in per_window.items():
                # str keys: int windows + 'avg' in one column would break pivot's sort
                rows.append(dict(pair=slug, model=m, window=str(w),
                                 mean_acc=float(col.mean()),
                                 sd_acc=float(col.std(ddof=1)) if col.count() > 1 else np.nan,
                                 n_subjects=int(col.count())))

            subj = pd.read_csv(os.path.join(pair_dir, 'per_subject.csv'))
            subj.insert(1, 'model', m)
            subj_frames.append(subj)

    if missing:
        print(f"comparison: not yet complete (skipped): {', '.join(missing)}", flush=True)
    if not rows:
        print(f"comparison: no completed results under {mode_root}", flush=True)
        return None

    windows = sorted(windows)
    columns = [str(w) for w in windows] + [AVG_COL]
    pair_order = [pair_slug(a, b) for a, b in pairs]

    # rows were appended in (pair order, model order), so no re-sorting is needed
    long = pd.DataFrame(rows)

    row_order = pd.MultiIndex.from_tuples(
        list(dict.fromkeys(zip(long['pair'], long['model']))), names=['pair', 'model'])
    mean_tbl = (long.pivot(index=['pair', 'model'], columns='window', values='mean_acc')
                .reindex(index=row_order, columns=columns))
    sd_tbl = (long.pivot(index=['pair', 'model'], columns='window', values='sd_acc')
              .reindex(index=row_order, columns=columns))

    out_dir = os.path.join(mode_root, 'comparison')
    os.makedirs(out_dir, exist_ok=True)
    long.to_csv(os.path.join(out_dir, 'comparison_long.csv'), index=False)
    mean_tbl.round(4).to_csv(os.path.join(out_dir, 'comparison_mean.csv'))
    sd_tbl.round(4).to_csv(os.path.join(out_dir, 'comparison_sd.csv'))
    pd.concat(subj_frames, ignore_index=True).to_csv(
        os.path.join(out_dir, 'all_subjects_long.csv'), index=False)

    # ---- Pooled over pairs that every model has completed ------------------
    complete = [s for s in pair_order if all((s, m) in wides for m in models)]
    pooled = None
    if len(complete) >= 2:
        prow = []
        for m in models:
            # subject x column, averaged over the complete pairs
            subj_mean = pd.concat([wides[(s, m)] for s in complete]).groupby(level=0).mean()
            for c in columns:
                col = subj_mean[c]
                best_on = sum(
                    1 for s in complete
                    if mean_tbl.loc[(s, m), c] == max(mean_tbl.loc[(s, mm), c] for mm in models))
                prow.append(dict(model=m, window=c, mean_acc=float(col.mean()),
                                 sd_acc=float(col.std(ddof=1)) if col.count() > 1 else np.nan,
                                 n_subjects=int(col.count()), n_pairs=len(complete),
                                 n_pairs_best=best_on))
        pooled = pd.DataFrame(prow)
        pooled.to_csv(os.path.join(out_dir, 'comparison_pooled.csv'), index=False)

    # ---- Markdown table ---------------------------------------------------
    header_cells = ['Pair', 'Architecture'] + [f"{w} ms" for w in windows] + ['Avg']
    lines = [
        f"# FineMI 0.5-3 Hz, decimated to 50 Hz (D=5): architecture comparison ({os.path.basename(mode_root)} mode)",
        "",
        "Within-subject CV test accuracy (%), mean (SD) across subjects. "
        "Bold = best architecture for that pair and window.",
        "",
    ]
    if pooled is not None:
        pm = pooled.set_index(['model', 'window'])
        lines += [
            f"## Pooled over {len(complete)} pairs",
            "",
            "Each subject's accuracy averaged over the pairs first; "
            "'pairs best' = number of pairs on which the architecture has the highest mean "
            "(ties count for each tied architecture).",
            "",
            "| Architecture | " + " | ".join(header_cells[2:]) + " |",
            "|" + "|".join(['---'] + ['---:'] * len(columns)) + "|",
        ]
        best_pooled = {c: max(pm.loc[(m, c), 'mean_acc'] for m in models) for c in columns}
        for m in models:
            cells = [MODEL_LABELS.get(m, m)]
            for c in columns:
                txt = _fmt(pm.loc[(m, c), 'mean_acc'], pm.loc[(m, c), 'sd_acc'])
                if len(models) > 1 and pm.loc[(m, c), 'mean_acc'] == best_pooled[c]:
                    txt = f"**{txt}**"
                cells.append(txt)
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            "| Pairs best | " + " | ".join(header_cells[2:]) + " |",
            "|" + "|".join(['---'] + ['---:'] * len(columns)) + "|",
        ]
        for m in models:
            lines.append("| " + " | ".join(
                [MODEL_LABELS.get(m, m)]
                + [f"{int(pm.loc[(m, c), 'n_pairs_best'])}/{len(complete)}" for c in columns]) + " |")
        lines += ["", "## Per pair", ""]

    lines += [
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(['---', '---'] + ['---:'] * len(columns)) + "|",
    ]
    for slug in pair_order:
        if slug not in mean_tbl.index.get_level_values('pair'):
            continue
        block = mean_tbl.loc[slug]
        best = {c: block[c].max() for c in mean_tbl.columns}
        for i, m in enumerate(block.index):
            cells = [slug.replace('_', ' vs ') if i == 0 else '', MODEL_LABELS.get(m, m)]
            for c in mean_tbl.columns:
                mu, sd = block.loc[m, c], sd_tbl.loc[(slug, m), c]
                txt = _fmt(mu, sd)
                if len(block) > 1 and not pd.isna(mu) and mu == best[c]:
                    txt = f"**{txt}**"
                cells.append(txt)
            lines.append("| " + " | ".join(cells) + " |")

    lines += ["", f"Subjects per cell: {sorted(set(int(n) for n in long['n_subjects']))}"]
    if missing:
        lines += ["", "Missing (not yet run): " + ", ".join(missing)]

    md_path = os.path.join(out_dir, 'comparison_table.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"\ncomparison written to {out_dir}", flush=True)
    print(mean_tbl.round(2).to_string(), flush=True)
    if pooled is not None:
        print(f"\npooled over {len(complete)} pairs (mean accuracy):", flush=True)
        print(pooled.pivot(index='model', columns='window', values='mean_acc')
              .reindex(index=list(models), columns=columns).round(2).to_string(), flush=True)
    return mean_tbl
