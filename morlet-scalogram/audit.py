"""
Correctness audit. Empirical, not argumentative - every claim is tested.

    .venv\\Scripts\\python audit.py

Covers: fold structure, arm equivalence, leakage, checkpoint semantics, determinism,
label alignment, model semantics, save/aggregate round-trip.
"""
import copy
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

import fine_mi as F

device, gpu = F.setup_determinism()
RESULTS = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    ok = bool(cond)
    RESULTS["pass" if ok else "fail"] += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def section(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


print(F.describe_environment(device, gpu).splitlines()[0])
print(f"dataset: {F.DATASET_ROOT}")

# ===========================================================================
section("T1. FOLD STRUCTURE - every trial tested exactly once, splits disjoint")
# ===========================================================================
subs_all, n_ch = F.load_pair(0, 5, verbose=False)
print(f"  {len(subs_all)} subjects, {n_ch} channels")

all_ok = True
for sid in sorted(subs_all):
    y = subs_all[sid]['y']
    skf = StratifiedKFold(n_splits=F.N_FOLDS, shuffle=True, random_state=int(42 + sid))
    splits = list(skf.split(np.arange(len(y)), y))

    test_union = np.concatenate([te for _, te in splits])
    if not (len(test_union) == len(y) and len(set(test_union)) == len(y)):
        all_ok = False
        print(f"    subject {sid}: test sets do not partition the trials")

    for fold, (tv, te) in enumerate(splits):
        tr, va = train_test_split(tv, test_size=0.25, random_state=int(42 + sid),
                                  stratify=y[tv])
        if set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
            all_ok = False
            print(f"    subject {sid} fold {fold}: overlapping splits")
        if len(np.unique(y[te])) < 2 or len(np.unique(y[va])) < 2:
            all_ok = False
            print(f"    subject {sid} fold {fold}: a split lost a class")

check("test folds partition every subject's trials exactly once", all_ok)
y0 = subs_all[sorted(subs_all)[0]]['y']
check("classes balanced per subject", abs((y0 == 0).sum() - (y0 == 1).sum()) <= 1,
      f"{int((y0==0).sum())} vs {int((y0==1).sum())}")
check("window reproduces the ORIGINAL slice X[:, :, :1000]",
      F.CUE_SAMPLE == 0 and F.WIN_SLICE.start == 0
      and F.WIN_SLICE.stop == F.WINDOW_SAMPLES,
      f"cue={F.CUE_SAMPLE} win={F.WIN_SLICE}")
check("context length matches the front-end contract",
      subs_all[sorted(subs_all)[0]]['X'].shape[2] == F.CTX_LEN, f"{F.CTX_LEN}")

# ===========================================================================
section("T2. ARM EQUIVALENCE - raw and cwt must see the same splits and same data")
# ===========================================================================
sid = sorted(subs_all)[0]
X, y = subs_all[sid]['X'], subs_all[sid]['y']
skf = StratifiedKFold(n_splits=F.N_FOLDS, shuffle=True, random_state=int(42 + sid))
tv, te = list(skf.split(np.arange(len(y)), y))[0]
tr, va = train_test_split(tv, test_size=0.25, random_state=int(42 + sid), stratify=y[tv])


def prep(seed_val):
    a, b, c = F.z_score_normalize_ctx(X[tr], X[va], X[te], F.WIN_SLICE)
    aug = F.add_gaussian_noise_augmentation(a, F.NOISE_LEVEL, random_seed=seed_val)
    return np.concatenate([a, aug]), np.concatenate([y[tr], y[tr]]), b, c


p1 = prep(int(F.SEED + sid + 0))
p2 = prep(int(F.SEED + sid + 0))
check("pre-front-end pipeline is deterministic",
      np.array_equal(p1[0], p2[0]) and np.array_equal(p1[2], p2[2]))

raw_tr, _, raw_va, raw_te = [x for x in (p1[0], None, p1[2], p1[3])] if False else (
    p1[0], None, p1[2], p1[3])
fr_raw = F.apply_frontend('raw', p1[0], p1[2], p1[3], device)
fr_cwt = F.apply_frontend('cwt', p1[0], p1[2], p1[3], device)
check("raw arm is exactly the original window of the shared tensor",
      np.array_equal(fr_raw[0], p1[0][:, :, F.WIN_SLICE].astype(np.float32)))
check("both arms consume the identical augmented tensor",
      fr_raw[0].shape[0] == fr_cwt[0].shape[0] == len(tr) * 2,
      f"{len(tr)} train -> {fr_raw[0].shape[0]} after augmentation")
check("raw shape (C,T)", fr_raw[0].shape[1:] == (n_ch, F.WINDOW_SAMPLES), str(fr_raw[0].shape))
check("cwt shape (C,F,T)", fr_cwt[0].shape[1:] == (n_ch, F.N_FREQS, F.T_OUT), str(fr_cwt[0].shape))

# ===========================================================================
section("T3. LEAKAGE")
# ===========================================================================
_, _, te_a = F.z_score_normalize_ctx(X[tr], X[va], X[te], F.WIN_SLICE)
Xtr_mod = X[tr].copy()
Xtr_mod[0] += 1000.0
_, _, te_b = F.z_score_normalize_ctx(Xtr_mod, X[va], X[te], F.WIN_SLICE)
check("test set DOES depend on train stats (expected, that is the design)",
      not np.array_equal(te_a, te_b))

tr_n, va_n, te_n = F.z_score_normalize_ctx(X[tr], X[va], X[te], F.WIN_SLICE)
wm = tr_n[:, :, F.WIN_SLICE].mean(axis=(0, 2))
ws = tr_n[:, :, F.WIN_SLICE].std(axis=(0, 2))
check("train window standardised by construction",
      np.abs(wm).max() < 1e-10 and np.abs(ws - 1).max() < 1e-10)
check("val/test NOT independently standardised (no per-split leakage)",
      np.abs(va_n[:, :, F.WIN_SLICE].mean(axis=(0, 2))).max() > 1e-6,
      f"val mean deviates {np.abs(va_n[:,:,F.WIN_SLICE].mean(axis=(0,2))).max():.4f}")

s_tr = F.morlet_scalogram(p1[0], device)
s_te1 = F.morlet_scalogram(p1[3], device)
s_te2 = F.morlet_scalogram(p1[3], device)
check("scalogram of a test trial is a pure function of that trial",
      np.array_equal(s_te1, s_te2))
n_tr, _, n_te = F.scalogram_normalize(s_tr, s_te1, s_te1)
check("scalogram norm uses train stats only",
      np.abs(n_tr.mean(axis=(0, 3))).max() < 1e-4
      and np.abs(n_te.mean(axis=(0, 3))).max() > 1e-6)

# ===========================================================================
section("T4. CHECKPOINT SEMANTICS")
# ===========================================================================
m = F.build_model('cwt', n_ch).to(device)
m.apply(F.init_weights)
opt = optim.Adam(m.parameters(), lr=0.1)
key = next(k for k, v in m.state_dict().items()
           if v.dtype.is_floating_point and v.numel() > 10)
shallow = m.state_dict().copy()
deep = {k: v.detach().clone() for k, v in m.state_dict().items()}
deepc = copy.deepcopy(m.state_dict())
before = float(m.state_dict()[key].flatten()[0])
xb = torch.randn(8, n_ch, F.N_FREQS, F.T_OUT, device=device)
nn.CrossEntropyLoss()(m(xb), torch.randint(0, 2, (8,), device=device)).backward()
opt.step()
after = float(m.state_dict()[key].flatten()[0])

check("optimizer.step() actually moved the weights", abs(after - before) > 1e-9)
check("state_dict().copy() is a BROKEN snapshot (current code)",
      abs(float(shallow[key].flatten()[0]) - after) < 1e-12,
      "it tracks the live weights")
check("{k: v.detach().clone()} is a correct snapshot",
      abs(float(deep[key].flatten()[0]) - before) < 1e-12)
check("copy.deepcopy(state_dict()) is also correct",
      abs(float(deepc[key].flatten()[0]) - before) < 1e-12)
import inspect
src = inspect.getsource(F.run_pair)
check("run_pair PRESERVES the original .state_dict().copy()",
      "state_dict().copy()" in src,
      "intentional: reproducing the paper's pipeline, not repairing it")

# functional: per fold, validate() is called 5x (val) then 1x (test). Fake only the val
# calls. Under the ORIGINAL semantics the restore is a no-op, so forcing the validation
# peak to different epochs must NOT change the tested weights.
_real_validate = F.validate


def _make_fake(pattern):
    ctr = {'i': 0}

    def f(model, loader, crit, dev):
        l, a, pr, lb = _real_validate(model, loader, crit, dev)
        i = ctr['i']
        ctr['i'] += 1
        pos = i % (len(pattern) + 1)
        return (l, pattern[pos], pr, lb) if pos < len(pattern) else (l, a, pr, lb)
    return f


def _run_with(pattern):
    F.validate = _make_fake(pattern)
    try:
        F.setup_determinism()
        r = F.run_pair(0, 5, 'raw', device, n_epochs=5, max_subjects=1, verbose=False)
    finally:
        F.validate = _real_validate
    return r[0]['time_window_results'][F.WINDOW_MS]['test_accuracy']


_first = _run_with([99.0, 1.0, 1.0, 1.0, 1.0])
_last = _run_with([1.0, 1.0, 1.0, 1.0, 99.0])
check("original semantics preserved: val peak does NOT change the tested weights",
      abs(_first - _last) < 1e-9,
      f"peak@epoch0 -> {_first:.2f}%   peak@epoch4 -> {_last:.2f}%   (final-epoch always)")

del m, opt, xb

# ===========================================================================
section("T5. MODEL SEMANTICS - does the 2D lift do what the design says?")
# ===========================================================================
mm = F.MultiscaleTemporalBlock2d(n_ch, 32).to(device).eval()
with torch.no_grad():
    base = torch.zeros(1, n_ch, F.N_FREQS, F.T_OUT, device=device)
    o1 = mm(base)
    poked = base.clone()
    poked[0, :, 10, 100] = 5.0                     # perturb one (freq, time) cell
    o2 = mm(poked)
    d = (o2 - o1).abs().sum(dim=(0, 1, 3))[0:F.N_FREQS].cpu().numpy()
check("MTC kernels do NOT mix across frequency",
      d[10] > 0 and np.delete(d, 10).max() == 0.0,
      f"response confined to freq row 10")

with torch.no_grad():
    t_spread = (o2 - o1).abs().sum(dim=(0, 1, 2)).cpu().numpy()
    nz = np.nonzero(t_spread)[0]
check("MTC temporal span matches the largest kernel",
      len(nz) == 31, f"{len(nz)} time cols affected, k_max=31")

full = F.build_model('cwt', n_ch).to(device).eval()
with torch.no_grad():
    out = full(torch.randn(4, n_ch, F.N_FREQS, F.T_OUT, device=device))
check("2D model emits (B, 2) logits", tuple(out.shape) == (4, 2))
check("channel dim is the electrode axis",
      full.multiscale_temporal.branch1[0].in_channels == n_ch, f"in_channels={n_ch}")
del mm, full, o1, o2, out

# ===========================================================================
section("T6. LABEL ALIGNMENT through augmentation")
# ===========================================================================
Xa, ya, _, _ = p1
half = len(ya) // 2
check("augmented labels are the train labels, duplicated",
      np.array_equal(ya[:half], ya[half:]) and np.array_equal(ya[:half], y[tr]))
check("augmented data differs from the original (noise was actually added)",
      not np.array_equal(Xa[:half], Xa[half:]))
check("augmentation preserves per-trial correspondence",
      np.abs(Xa[:half] - Xa[half:]).mean() < 1.0,
      "augmented copy stays close to its source trial")
ds = F.EEGDataset(fr_cwt[0], ya)
check("EEGDataset keeps (sample, label) aligned",
      int(ds[7]['label']) == int(ya[7])
      and np.allclose(ds[7]['eeg'].numpy(), fr_cwt[0][7]))

# ===========================================================================
section("T7. END-TO-END DETERMINISM - same config twice, bit-identical")
# ===========================================================================
print("  running 2 subjects x 5 folds x 3 epochs, twice, per arm...")
det_ok = True
for mode in ('raw', 'cwt'):
    F.setup_determinism()
    r1 = F.run_pair(0, 5, mode, device, n_epochs=3, max_subjects=2, verbose=False)
    F.setup_determinism()
    r2 = F.run_pair(0, 5, mode, device, n_epochs=3, max_subjects=2, verbose=False)
    a1 = [s['time_window_results'][F.WINDOW_MS]['test_accuracy'] for s in r1]
    a2 = [s['time_window_results'][F.WINDOW_MS]['test_accuracy'] for s in r2]
    p_1 = np.concatenate([s['time_window_results'][F.WINDOW_MS]['test_preds'] for s in r1])
    p_2 = np.concatenate([s['time_window_results'][F.WINDOW_MS]['test_preds'] for s in r2])
    same = a1 == a2 and np.array_equal(p_1, p_2)
    det_ok = det_ok and same
    check(f"{mode}: identical accuracies and predictions across runs", same,
          f"{[round(a,2) for a in a1]}")

# ===========================================================================
section("T8. SPLIT SHARING - both arms must use the SAME folds")
# ===========================================================================
F.setup_determinism()
rr = F.run_pair(0, 5, 'raw', device, n_epochs=1, max_subjects=2, verbose=False)
F.setup_determinism()
rc = F.run_pair(0, 5, 'cwt', device, n_epochs=1, max_subjects=2, verbose=False)
lab_r = [np.concatenate([f['test_labels'] for f in
                         s['time_window_results'][F.WINDOW_MS]['fold_results']]) for s in rr]
lab_c = [np.concatenate([f['test_labels'] for f in
                         s['time_window_results'][F.WINDOW_MS]['fold_results']]) for s in rc]
check("both arms evaluate the identical test trials in the identical order",
      all(np.array_equal(a, b) for a, b in zip(lab_r, lab_c)))
check("both arms cover the same subjects",
      [s['subject_id'] for s in rr] == [s['subject_id'] for s in rc])

# ===========================================================================
section("T9. SAVE / AGGREGATE ROUND-TRIP")
# ===========================================================================
tmp = tempfile.mkdtemp()
try:
    _, wide = F.save_results(rc, 0, 5, 'cwt', gpu, results_root=tmp, verbose=False)
    computed = {int(s['subject_id']):
                round(s['time_window_results'][F.WINDOW_MS]['test_accuracy'], 6)
                for s in rc}
    import pandas as pd
    back = pd.read_csv(os.path.join(tmp, 'cwt', 'HOC_SPS', 'wide_subject_x_window.csv'),
                       index_col=0).iloc[:, 0]
    check("saved CSV matches the in-memory accuracies",
          all(abs(back.loc[k] - v) < 1e-6 for k, v in computed.items()),
          f"{dict(back.round(2))}")
    check("is_done() detects the saved pair", F.is_done(0, 5, 'cwt', results_root=tmp))
    check("is_done() is False for an unsaved pair",
          not F.is_done(1, 2, 'cwt', results_root=tmp))

    F.save_results(rr, 0, 5, 'raw', gpu, results_root=tmp, verbose=False)
    import aggregate
    tab, long_df = aggregate.build_table(tmp)
    check("aggregate reconstructs the pair name", list(tab['pair']) == ['HOC/SPS'])
    row = tab.iloc[0]
    check("aggregate delta = cwt - raw",
          abs(row["delta"] - (row["cwt_acc"] - row["raw_acc"])) < 0.011,
          f"raw={row['raw_acc']:.2f} cwt={row['cwt_acc']:.2f} delta={row['delta']:.2f}")
    check("aggregate carries the paper's reference numbers",
          row["shallow"] == 65.49 and row["fine_paper"] == 68.51)

    # the reference tables must reproduce every aggregate the paper states about itself
    import numpy as _np
    _g = {}
    for _n in aggregate.FINE_PAPER:
        _g.setdefault(aggregate.proximity(_n), []).append(aggregate.FINE_PAPER[_n])
    for _k, _exp in (("same", 56.47), ("adjacent", 59.02), ("distant", 64.67)):
        check(f"paper self-consistency: {_k} group mean",
              abs(_np.mean(_g[_k]) - _exp) < 0.005, f"{_np.mean(_g[_k]):.2f} vs {_exp}")
    _loss = [n for n in aggregate.FINE_PAPER
             if aggregate.FINE_PAPER[n] < aggregate.SHALLOW[n]]
    _win = [n for n in aggregate.FINE_PAPER
            if aggregate.FINE_PAPER[n] > aggregate.SHALLOW[n]]
    check("paper self-consistency: 23/28 wins", len(_win) == 23, f"{len(_win)}")
    check("paper self-consistency: 5 losing pairs, mean 1.76 pp",
          len(_loss) == 5 and abs(_np.mean([aggregate.SHALLOW[n] - aggregate.FINE_PAPER[n]
                                             for n in _loss]) - 1.76) < 0.005)

    # per-trial predictions must be persisted and must reproduce the saved accuracy
    import pandas as _pd
    _pp = os.path.join(tmp, "cwt", "HOC_SPS", "predictions.csv")
    check("predictions.csv written", os.path.isfile(_pp))
    _pr = _pd.read_csv(_pp)
    _acc = _pr.groupby("subject").apply(
        lambda g: (g.y_true == g.y_pred).mean() * 100, include_groups=False)
    check("predictions reproduce the saved per-subject accuracy",
          all(abs(_acc.loc[k] - v) < 1e-6 for k, v in computed.items()))
    check("every trial appears exactly once per subject",
          all(g["trial"].is_unique for _, g in _pr.groupby("subject")))
    _km = aggregate.binary_metrics(_pr.y_true.values, _pr.y_pred.values)
    check("kappa computable from predictions",
          _np.isfinite(_km["kappa"]), f"kappa={_km['kappa']:.4f} f1={_km['f1_macro']:.2f}")

    # version stamping must invalidate stale results
    check("is_done True for a current-version pair", F.is_done(0, 5, "cwt", results_root=tmp))
    _mp = os.path.join(tmp, "cwt", "HOC_SPS", "results.json")
    _meta = json.load(open(_mp))
    _meta["pipeline_version"] = 999
    json.dump(_meta, open(_mp, "w"))
    check("is_done False when pipeline_version differs",
          not F.is_done(0, 5, "cwt", results_root=tmp))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ===========================================================================
section("T10. POOLED ACCURACY ARITHMETIC")
# ===========================================================================
s0 = rc[0]['time_window_results'][F.WINDOW_MS]
manual = (s0['test_preds'] == s0['test_labels']).mean() * 100
check("subject accuracy = pooled over folds, not a mean of fold means",
      abs(manual - s0['test_accuracy']) < 1e-9, f"{manual:.4f}%")
fold_mean = np.mean([f['test_accuracy'] for f in s0['fold_results']])
print(f"  (mean-of-folds would be {fold_mean:.4f}% - differs when folds are uneven)")
check("all trials accounted for exactly once",
      s0['n_test_samples'] == len(subs_all[rc[0]['subject_id']]['y']),
      f"{s0['n_test_samples']} tested")

# ===========================================================================
print("\n" + "=" * 74)
print(f"AUDIT: {RESULTS['pass']} passed, {RESULTS['fail']} failed")
print("=" * 74)
sys.exit(1 if RESULTS['fail'] else 0)
