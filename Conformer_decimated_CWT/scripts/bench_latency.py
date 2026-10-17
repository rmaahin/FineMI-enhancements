# Inference latency benchmark. No training, no data: random inputs of the real shapes.
#
# Measures the forward pass only, in eval mode under torch.no_grad, after warm-up, with
# the device synchronized around every timed iteration. Deterministic algorithms are NOT
# enabled here (they are a training-reproducibility setting and would distort deployment
# latency).
#
# Run from a project folder (Slim_conformer_fine/ or Conformer_decimated_CWT/):
#
#   python -m scripts.bench_latency                          # this project's models, auto device
#   python -m scripts.bench_latency --device cpu --threads 4
#   python -m scripts.bench_latency --batch-sizes 1 32 --iters 300
#   python -m scripts.bench_latency --out ../latency.csv     # append; run again in the
#                                                            # other project for one table
#
# Columns: project, model, params, window_ms, T, device, device_name, threads, batch_size,
# iters, ms_median, ms_p95, ms_mean, ms_min, ms_per_trial_median, trials_per_s,
# frontend_ms_per_trial (CWT models only), torch.
#
# batch_size 1 is the number to quote for a real-time interface: one 4 s window, one
# decision. batch_size 32 shows throughput when trials are processed in bulk.

import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np
import torch

from src.config import MODELS, PROJECT_ROOT, make_config
from src.train import build_model

try:                                   # only Conformer_decimated_CWT has CWT models
    from src.config import CWT_MODELS
except ImportError:
    CWT_MODELS = ()

FIELDS = ['project', 'model', 'params', 'window_ms', 'T', 'device', 'device_name', 'threads',
          'batch_size', 'iters', 'ms_median', 'ms_p95', 'ms_mean', 'ms_min',
          'ms_per_trial_median', 'trials_per_s', 'frontend_ms_per_trial', 'torch']


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def make_input(cfg, T, batch, device, rng):
    """Model-ready batch plus the front-end cost per trial (0 for raw-input models).

    The front-end is timed on the CPU/GPU path the pipeline actually uses, so for a CWT
    model the caller can add it to the forward time for an end-to-end figure.
    """
    x = rng.standard_normal((batch, cfg.n_channels, T))
    front_ms = 0.0
    if cfg.model in CWT_MODELS:
        from src.cwt import get_cwt, scalogram_normalize
        cwt = get_cwt(cfg)
        cwt(x, device)                                       # warm kernels AND the FFT plan for this batch
        sync(device)
        t0 = time.perf_counter()
        s = cwt(x, device)
        sync(device)
        front_ms = (time.perf_counter() - t0) * 1000.0 / batch
        s, _, _ = scalogram_normalize(s, s[:1], s[:1])
        x = s
    return torch.as_tensor(x, dtype=torch.float32, device=device), front_ms


def bench_one(cfg, T, batch, device, iters, warmup, rng):
    model = build_model(cfg, n_channels=cfg.n_channels, n_timepoints=T).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    x, front_ms = make_input(cfg, T, batch, device, rng)

    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync(device)
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    med = statistics.median(times)
    row = dict(model=cfg.model, params=params, window_ms=None, T=T,
               batch_size=batch, iters=iters,
               ms_median=round(med, 4),
               ms_p95=round(times[min(len(times) - 1, int(0.95 * len(times)))], 4),
               ms_mean=round(statistics.fmean(times), 4),
               ms_min=round(times[0], 4),
               ms_per_trial_median=round(med / batch, 4),
               trials_per_s=round(batch / (med / 1000.0), 1),
               frontend_ms_per_trial=round(front_ms, 4) if front_ms else '')
    del model, x
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser(description="Forward-pass latency of this project's models.")
    ap.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    ap.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    ap.add_argument('--threads', type=int, default=0,
                    help="torch CPU threads (0 = leave as is). Use 1 for an embedded-style figure.")
    ap.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 32])
    ap.add_argument('--iters', type=int, default=200, help="timed forward passes per cell")
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--windows', nargs='+', type=int, default=None,
                    help="windows in ms (default: all four)")
    ap.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'results', 'latency.csv'),
                    help="CSV to write; appended to if it already exists")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    dev = args.device
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    project = os.path.basename(PROJECT_ROOT)
    rng = np.random.default_rng(0)

    print(f"Latency benchmark: project={project} device={device_name} "
          f"threads={torch.get_num_threads()} torch={torch.__version__}", flush=True)
    print(f"  models={args.models} batches={args.batch_sizes} iters={args.iters} "
          f"(+{args.warmup} warm-up)", flush=True)
    print(f"{'model':<18}{'window':>8}{'batch':>7}{'median ms':>12}{'p95 ms':>10}"
          f"{'ms/trial':>10}{'trials/s':>11}", flush=True)

    rows = []
    for m in args.models:
        cfg = make_config(m, 'full', dataset_root='')
        windows = list(zip(cfg.test_time_windows_ms, cfg.window_samples()))
        if args.windows:
            windows = [(w, t) for w, t in windows if w in args.windows]
        for w_ms, T in windows:
            for batch in args.batch_sizes:
                r = bench_one(cfg, T, batch, device, args.iters, args.warmup, rng)
                r.update(project=project, window_ms=w_ms, device=device.type,
                         device_name=device_name, threads=torch.get_num_threads(),
                         torch=torch.__version__)
                rows.append(r)
                print(f"{m:<18}{w_ms:>7}{'ms'}{batch:>7}{r['ms_median']:>12.3f}"
                      f"{r['ms_p95']:>10.3f}{r['ms_per_trial_median']:>10.3f}"
                      f"{r['trials_per_s']:>11.1f}", flush=True)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new = not os.path.isfile(out)
    with open(out, 'a', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, '') for k in FIELDS})
    print(f"\n{len(rows)} rows {'written to' if new else 'appended to'} {out}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
