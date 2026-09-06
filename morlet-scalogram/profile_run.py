"""Diagnostic: where does the per-pair time actually go, and is the checkpoint logic sound?"""
import time
import numpy as np
import fine_mi as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device, gpu = F.setup_determinism()
print(F.describe_environment(device, gpu))

# ---------------------------------------------------------------- A. checkpoint semantics
print("\n" + "=" * 70)
print("A. does `state_dict().copy()` actually snapshot the best model?")
print("=" * 70)
m = F.build_model('cwt', 62).to(device)
m.apply(F.init_weights)
opt = optim.Adam(m.parameters(), lr=0.1)

shallow = m.state_dict().copy()
deep = {k: v.detach().clone() for k, v in m.state_dict().items()}
key = next(k for k, v in m.state_dict().items() if v.dtype.is_floating_point and v.numel() > 10)
before = float(m.state_dict()[key].flatten()[0])

x = torch.randn(8, 62, F.N_FREQS, F.T_OUT, device=device)
loss = nn.CrossEntropyLoss()(m(x), torch.randint(0, 2, (8,), device=device))
opt.zero_grad(); loss.backward(); opt.step()

after = float(m.state_dict()[key].flatten()[0])
s_val = float(shallow[key].flatten()[0])
d_val = float(deep[key].flatten()[0])
print(f"  param '{key}'[0]")
print(f"    before step          {before:+.6f}")
print(f"    after  step          {after:+.6f}   (changed by {after-before:+.2e})")
print(f"    .copy()   snapshot   {s_val:+.6f}   {'TRACKS LIVE WEIGHTS' if abs(s_val-after)<1e-12 else 'independent'}")
print(f"    .clone()  snapshot   {d_val:+.6f}   {'tracks live weights' if abs(d_val-after)<1e-12 else 'INDEPENDENT (correct)'}")
BROKEN = abs(s_val - after) < 1e-12 and abs(after - before) > 0
print(f"\n  => shallow .copy() is {'BROKEN: best-epoch selection is a no-op' if BROKEN else 'fine'}")

# ---------------------------------------------------------------- B. time breakdown
print("\n" + "=" * 70)
print("B. per-fold time breakdown (real data, one subject, one fold)")
print("=" * 70)
subs, n_ch = F.load_pair(0, 5, max_subjects=1, verbose=False)
sid = sorted(subs)[0]
X, y = subs[sid]['X'], subs[sid]['y']
from sklearn.model_selection import StratifiedKFold, train_test_split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + sid)
tv, te = list(skf.split(np.arange(len(y)), y))[0]
tr, va = train_test_split(tv, test_size=0.25, random_state=42 + sid, stratify=y[tv])

t0 = time.perf_counter()
Xtr, Xva, Xte = F.z_score_normalize_ctx(X[tr], X[va], X[te], F.WIN_SLICE)
aug = F.add_gaussian_noise_augmentation(Xtr, 0.15, random_seed=42)
Xtr_f = np.concatenate([Xtr, aug]); ytr_f = np.concatenate([y[tr], y[tr]])
t_prep = time.perf_counter() - t0

t0 = time.perf_counter()
ftr, fva, fte = F.apply_frontend('cwt', Xtr_f, Xva, Xte, device)
torch.cuda.synchronize()
t_cwt = time.perf_counter() - t0

ds = F.EEGDataset(ftr, ytr_f)
g = torch.Generator(); g.manual_seed(0)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, generator=g)
vloader = DataLoader(F.EEGDataset(fva, y[va]), batch_size=32, shuffle=False)

model = F.build_model('cwt', n_ch).to(device)
model.apply(F.init_weights)
crit = nn.CrossEntropyLoss()
optm = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for _ in range(3):   # warmup
    F.train_epoch(model, loader, crit, optm, device)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    F.train_epoch(model, loader, crit, optm, device)
torch.cuda.synchronize()
t_train = (time.perf_counter() - t0) / 10

t0 = time.perf_counter()
for _ in range(10):
    F.validate(model, vloader, crit, device)
torch.cuda.synchronize()
t_val = (time.perf_counter() - t0) / 10

t0 = time.perf_counter()
for _ in range(10):
    _ = {k: v.detach().clone() for k, v in model.state_dict().items()}
torch.cuda.synchronize()
t_ckpt = (time.perf_counter() - t0) / 10

n_batches = len(loader)
print(f"  train trials {len(ytr_f)} -> {n_batches} batches/epoch of 32")
print(f"  numpy prep (z-score + augment) : {t_prep*1000:8.1f} ms   (once per fold)")
print(f"  CWT front-end                  : {t_cwt*1000:8.1f} ms   (once per fold)")
print(f"  train epoch                    : {t_train*1000:8.1f} ms   x50")
print(f"  validate epoch                 : {t_val*1000:8.1f} ms   x50")
print(f"  deepcopy checkpoint            : {t_ckpt*1000:8.1f} ms   (only on improvement)")
per_fold = t_prep + t_cwt + 50 * (t_train + t_val)
print(f"  -> projected per fold          : {per_fold:8.2f} s")
print(f"  -> projected per pair (90)     : {per_fold*90/60:8.2f} min   (measured: 13.5)")
print(f"\n  epoch loop is {50*(t_train+t_val)/per_fold*100:.1f}% of fold time; "
      f"front-end is {t_cwt/per_fold*100:.1f}%")

# ---------------------------------------------------------------- C. is the GPU busy?
print("\n" + "=" * 70)
print("C. GPU saturation - one batch, forward+backward")
print("=" * 70)
xb = torch.randn(32, 62, F.N_FREQS, F.T_OUT, device=device)
yb = torch.randint(0, 2, (32,), device=device)
for _ in range(5):
    optm.zero_grad(); crit(model(xb), yb).backward(); optm.step()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    optm.zero_grad(); crit(model(xb), yb).backward(); optm.step()
torch.cuda.synchronize()
t_step = (time.perf_counter() - t0) / 50
print(f"  step time (data already on GPU) : {t_step*1000:6.2f} ms")
print(f"  measured train epoch / batches  : {t_train/n_batches*1000:6.2f} ms per batch")
overhead = (t_train / n_batches - t_step) * 1000
print(f"  => per-batch host overhead      : {overhead:6.2f} ms "
      f"({overhead/(t_train/n_batches*1000)*100:.0f}% of batch time)")
print(f"  peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB of "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
