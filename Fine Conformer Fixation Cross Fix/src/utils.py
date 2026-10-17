def init_weights(m):
    import torch.nn as nn, torch
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)


def read_results_json(pair_dir):
    """results.json as a dict, or None if it is missing or unreadable."""
    import json, os
    path = os.path.join(pair_dir, 'results.json')
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_pair_done(pair_dir):
    """A pair is complete once results.json exists (it is written last, atomically)
    AND it carries this pipeline's stamp (pipeline_version + decimation). A folder
    written by other code counts as not done, so a sweep recomputes it."""
    from src.config import PIPELINE_VERSION, DECIM
    meta = read_results_json(pair_dir)
    if meta is None:
        return False
    return meta.get('pipeline_version') == PIPELINE_VERSION and meta.get('decim') == DECIM


def write_pair_results(pair_dir, pair_name, class_a, class_b, windows_ms,
                       all_subject_results, meta):
    """[P7] Save per-subject + per-fold results (verbatim from the notebook).

    Writes per_subject.csv, per_fold.csv, wide_subject_x_window.csv, results.pkl,
    and finally results.json (atomically, so a killed run never leaves a
    half-written completion marker). `meta` carries seed / gpu / torch / model /
    mode / dataset / pipeline stamp into the payload. Returns the wide
    subjects x windows DataFrame.
    """
    import json, os, pickle
    import pandas as pd

    os.makedirs(pair_dir, exist_ok=True)

    subj_rows, fold_rows = [], []
    for r in sorted(all_subject_results, key=lambda x: x['subject_id']):
        sid = int(r['subject_id'])
        for w in windows_ms:
            twr = r['time_window_results'][w]
            subj_rows.append(dict(pair=pair_name, subject=sid, window_ms=w,
                                  accuracy=float(twr['test_accuracy']),
                                  n_test=int(twr['n_test_samples'])))
            for f in twr['fold_results']:
                fold_rows.append(dict(pair=pair_name, subject=sid, window_ms=w,
                                      fold=int(f['fold']),
                                      accuracy=float(f['test_accuracy']),
                                      val_accuracy=float(f['val_accuracy']),
                                      best_epoch=int(f['best_epoch']),
                                      n_test=int(f['n_test_samples'])))

    df_subj = pd.DataFrame(subj_rows)
    df_fold = pd.DataFrame(fold_rows)

    # subjects x windows, fixed row/column order — this is what the paired tests consume
    wide = (df_subj.pivot(index='subject', columns='window_ms', values='accuracy')
            .reindex(columns=list(windows_ms)).sort_index())

    df_subj.to_csv(f"{pair_dir}/per_subject.csv", index=False)
    df_fold.to_csv(f"{pair_dir}/per_fold.csv", index=False)
    wide.to_csv(f"{pair_dir}/wide_subject_x_window.csv")

    payload = dict(pair=pair_name, class_a=int(class_a), class_b=int(class_b),
                   windows=list(windows_ms), **meta,
                   subjects=[int(s) for s in wide.index],
                   matrix=wide.to_numpy().tolist())
    with open(f"{pair_dir}/results.pkl", "wb") as f:
        pickle.dump(payload, f)
    tmp_path = f"{pair_dir}/results.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, f"{pair_dir}/results.json")

    return wide
