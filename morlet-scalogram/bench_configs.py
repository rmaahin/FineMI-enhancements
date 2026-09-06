"""Where the 2D model's FLOPs go, and what each knob actually buys."""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import fine_mi as F

device, gpu = F.setup_determinism()
print(F.describe_environment(device, gpu).splitlines()[0])


def timed(model, x, iters=100):
    y = torch.randint(0, 2, (x.shape[0],), device=device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(10):
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def macs_2d(n_ch, ks, n_f, t):
    mtc = n_ch * 32 * sum(ks) * n_f * t
    spatial = 96 * 64 * 9 * n_f * t
    fuse = (64 * 15 + 64 * 64) * (n_f // 2) * (t // 2)
    return (mtc + spatial + fuse) / 1e6, mtc / 1e6, spatial / 1e6


class MTC2d(nn.Module):
    def __init__(self, cin, ks, cout=32):
        super().__init__()
        self.b = nn.ModuleList([nn.Sequential(
            nn.Conv2d(cin, cout, (1, k), padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(cout), nn.ReLU()) for k in ks])

    def forward(self, x):
        return torch.cat([b(x) for b in self.b], dim=1)


def build(ks, n_f, t):
    m = F.FINEScalogram2D(62, n_f, t)
    m.multiscale_temporal = MTC2d(62, ks)
    return m.to(device)


print(f"\nbaseline raw-1D model for reference:")
m1d = F.build_model('raw', 62).to(device)
x1d = torch.randn(32, 62, F.WINDOW_SAMPLES, device=device)
t_raw = timed(m1d, x1d)
print(f"  raw 1D  (62,1000)               {t_raw:7.2f} ms/step")
del m1d, x1d
torch.cuda.empty_cache()

CONFIGS = [
    ("CURRENT  F=24 T=250 k={7,15,31}", (7, 15, 31), 24, 250),
    ("k scaled to match raw's span     ", (2, 4, 8), 24, 250),
    ("k={3,7,15}                       ", (3, 7, 15), 24, 250),
    ("decim 8: F=24 T=125 k={7,15,31}  ", (7, 15, 31), 24, 125),
    ("decim 8 + scaled k               ", (2, 4, 8), 24, 125),
    ("F=16 T=250 k={7,15,31}           ", (7, 15, 31), 16, 250),
    ("F=16 decim8 + scaled k           ", (2, 4, 8), 16, 125),
]

print(f"\n{'config':36s} {'ms/step':>9} {'vs cur':>7} {'MMACs':>8} {'MTC%':>6} {'pair est':>9}")
base = None
for name, ks, n_f, t in CONFIGS:
    m = build(ks, n_f, t)
    m.apply(F.init_weights)
    x = torch.randn(32, 62, n_f, t, device=device)
    ms = timed(m, x)
    if base is None:
        base = ms
    tot, mtc, sp = macs_2d(62, ks, n_f, t)
    # 13.5 min measured at the current config
    est = 13.5 * ms / base
    print(f"{name:36s} {ms:>9.2f} {base/ms:>6.2f}x {tot:>8.0f} {mtc/tot*100:>5.0f}% {est:>7.1f}min")
    del m, x
    torch.cuda.empty_cache()

print(f"\nraw 1D is {base/t_raw:.1f}x cheaper per step than the current cwt config")
print("NOTE: kernel spans. raw {7,15,31}@250Hz = 28/60/124 ms.")
print("      cwt {7,15,31}@62.5Hz = 112/240/496 ms  <- 4x LONGER receptive field")
print("      cwt {2,4,8}@62.5Hz  = 32/64/128 ms  <- matches raw")
