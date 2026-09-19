import src.determinism  # noqa: F401 — must be first (sets CUBLAS env before torch loads)
# Preflight: everything the full sweep needs, checked in a few minutes, no training run.
#
# submit_ls6.slurm runs this first in BOTH modes and stops the job if any check fails.
# In smoke mode it also times one epoch of every model x window at FULL-sweep shapes
# and projects how long the full 28-pair sweep will take on the node, failing if that
# does not fit in the walltime the full job will ask for.
#
#   python -m scripts.preflight --dataset-root $SCRATCH/finemi-dataset/FineMI_0.5_3hz
#   python -m scripts.preflight --skip-timing            # checks only
#
# Checks (each prints [ok] or [FAIL]):
#   packages      torch, numpy, scipy, sklearn, pandas import (scipy is needed by the stats)
#   cuda          a GPU is visible and the strict determinism settings apply
#   results       the results folder is writable
#   pairs         all 28 pairs parse to 28 distinct folders
#   data          EVERY subject file loads, decimates to 50 Hz, has 62 channels, finite
#                 values, all 8 classes, and enough trials of every class for 5-fold CV
#                 (smoke training only touches 2 subjects; this covers the other 16)
#   cwt           Morlet kernels zero-mean + L1-normalised, bank peaks at the right
#                 frequency, a 1 / 2 Hz sine peaks at the right row end to end, every
#                 window length transforms to finite (C, 24, T)
#   models        every model x every window (the FULL config: 4 windows) builds, runs
#                 forward + backward + an Adam step on a batch of 32 under the strict
#                 deterministic algorithms, with finite loss and gradients
#   determinism   training the same model twice from the same seed gives bit-identical
#                 weights (every model)
#   timing        (smoke) full-sweep time projection vs --walltime-hours

import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np

from src.config import (DATASET_ROOT_DEFAULT, RESULTS_ROOT_DEFAULT, MODELS, CWT_MODELS,
                        MODE_OVERRIDES, PAIRS, make_config, mode_root, pair_slug, parse_pairs)

RESULTS = []          # (status, name, detail)
REPORT = {}           # written to results/<mode>/preflight/


def run_check(name, fn, *args, **kwargs):
    """Run one check; print [ok]/[FAIL] and record it. Returns fn's value or None."""
    print(f"\n--- {name} ---", flush=True)
    t0 = time.time()
    try:
        detail = fn(*args, **kwargs)
    except Exception as e:
        traceback.print_exc()
        sys.stderr.flush()
        msg = f"{type(e).__name__}: {e}"
        print(f"[FAIL] {name}: {msg}", flush=True)
        RESULTS.append(('FAIL', name, msg))
        return None
    summary = detail if isinstance(detail, str) else ''
    if isinstance(detail, tuple):
        summary, detail = detail
    print(f"[ok]   {name}{': ' + summary if summary else ''}  ({time.time() - t0:.1f}s)",
          flush=True)
    RESULTS.append(('ok', name, summary))
    return detail


# ---------------------------------------------------------------------------
def check_packages():
    import torch, scipy, sklearn, pandas
    from scipy import stats  # noqa: F401 — paired_stats needs it
    v = (f"python {sys.version.split()[0]}, torch {torch.__version__}, numpy {np.__version__}, "
         f"scipy {scipy.__version__}, sklearn {sklearn.__version__}, pandas {pandas.__version__}")
    REPORT['packages'] = v
    return v


def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (is this a GPU node, and is the venv's "
                           "torch a CUDA build?)")
    src.determinism.apply_all(seed=42)          # the exact settings every worker uses
    props = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    info = (f"{name}, {props.total_memory / 1e9:.1f} GB, cuda {torch.version.cuda}, "
            f"visible devices {torch.cuda.device_count()}")
    if 'A100' not in name:
        print(f"WARNING: GPU is {name}, not an A100. Timings and bit-exact results are "
              f"per GPU model.", flush=True)
    REPORT['gpu'] = info
    return info


def check_results_writable(root):
    os.makedirs(root, exist_ok=True)
    probe = os.path.join(root, f".write_probe_{os.getpid()}")
    with open(probe, 'w') as f:
        f.write('ok')
    os.remove(probe)
    return root


def check_pairs():
    pairs = parse_pairs(['all'])
    slugs = [pair_slug(a, b) for a, b in pairs]
    if len(pairs) != 28 or len(set(slugs)) != 28 or pairs != PAIRS:
        raise RuntimeError(f"expected 28 distinct pairs, got {len(pairs)} / {len(set(slugs))}")
    return f"28 pairs, {slugs[0]} ... {slugs[-1]}"


def check_data(dataset_root, cfg_full):
    """Loads every subject exactly as the workers do (decimated, cached per process)."""
    from src.data import load_subjects
    subjects = load_subjects(dataset_root, cfg_full.band_tag, cfg_full.decim)
    max_win = max(cfg_full.window_samples())
    problems = []
    lengths, counts = set(), {}
    for sid, data, labels, orig_len in subjects:
        tag = f"subject {sid}"
        if data.ndim != 3:
            problems.append(f"{tag}: data is {data.ndim}-D, expected (trials, channels, time)")
            continue
        if data.shape[1] != cfg_full.n_channels:
            problems.append(f"{tag}: {data.shape[1]} channels, expected {cfg_full.n_channels}")
        if data.shape[2] < max_win:
            problems.append(f"{tag}: {data.shape[2]} samples after decimation "
                            f"({orig_len} before), need {max_win} for the 4000 ms window")
        expected_len = math.ceil(orig_len / cfg_full.decim)
        if data.shape[2] != expected_len:
            problems.append(f"{tag}: decimated length {data.shape[2]} != ceil({orig_len}/"
                            f"{cfg_full.decim}) = {expected_len}")
        if labels.shape[0] != data.shape[0]:
            problems.append(f"{tag}: {labels.shape[0]} labels for {data.shape[0]} trials")
        if not np.isfinite(data).all():
            problems.append(f"{tag}: non-finite values in the EEG")
        uniq, cnt = np.unique(labels, return_counts=True)
        c = {int(u): int(n) for u, n in zip(uniq, cnt)}
        counts[sid] = c
        missing = [k for k in range(8) if k not in c]
        if missing:
            problems.append(f"{tag}: classes {missing} missing (pairs with them would skip it)")
        extra = [k for k in c if not 0 <= k < 8]
        if extra:
            problems.append(f"{tag}: unexpected labels {extra}")
        few = {k: n for k, n in c.items() if 0 <= k < 8 and n < cfg_full.n_folds}
        if few:
            problems.append(f"{tag}: classes {few} have fewer than {cfg_full.n_folds} trials "
                            f"(5-fold stratified CV would fail)")
        lengths.add((data.shape[1], data.shape[2], orig_len))
    if len(subjects) != 18:
        print(f"WARNING: found {len(subjects)} subject files, expected 18", flush=True)
    if problems:
        for p in problems:
            print(f"  problem: {p}", flush=True)
        raise RuntimeError(f"{len(problems)} data problem(s), see above")
    shapes = ', '.join(f"(C={c}, T={t} @ {cfg_full.sampling_rate} Hz, was {o})"
                       for c, t, o in sorted(lengths))
    REPORT['data'] = dict(n_subjects=len(subjects), shapes=sorted(lengths),
                          class_counts={str(k): v for k, v in counts.items()})
    first = subjects[0]
    return (f"{len(subjects)} subjects, {shapes}, trials/class {min(min(v.values()) for v in counts.values())}"
            f"-{max(max(v.values()) for v in counts.values())}",
            dict(n_subjects=len(subjects), first_labels=first[2]))


def check_cwt(cfg_cwt, device):
    from src.cwt import get_cwt, scalogram_normalize
    cwt = get_cwt(cfg_cwt)
    fs = cwt.fs
    print(f"bank {cwt.freqs[0]:.2f}-{cwt.freqs[-1]:.2f} Hz x{cwt.n_freqs} @ {fs} Hz | "
          f"kernel K={cwt.K} ({cwt.K / fs:.1f} s), pad={cwt.pad}", flush=True)

    # 1. kernel geometry
    if abs(cwt.psi_r.mean(axis=1)).max() > 1e-12:
        raise AssertionError("kernels are not zero-mean")
    if not np.allclose(np.abs(cwt.psi_r + 1j * cwt.psi_i).sum(axis=1), 1.0):
        raise AssertionError("kernels are not L1-normalised")

    # 2. frequency selectivity of the bank itself (exact, no edge effects)
    step = np.log10(cwt.freqs[1] / cwt.freqs[0])
    tk = np.arange(cwt.K) / fs
    psi = cwt.psi_r + 1j * cwt.psi_i
    for f0 in (0.7, 1.0, 2.0, 3.0):
        resp = np.abs(psi @ np.exp(-2j * np.pi * f0 * tk))
        peak = cwt.freqs[int(np.argmax(resp))]
        if abs(np.log10(peak / f0)) > step / 2 * 1.01:
            raise AssertionError(f"kernel bank: {f0} Hz tone peaked at the {peak:.2f} Hz row")

    # 3. end to end on a 4 s sine (loose: every column is inside the wavelet's edge region)
    L = max(cfg_cwt.window_samples())
    t = np.arange(L) / fs
    c0, c1 = L // 4, 3 * L // 4
    for f0 in (1.0, 2.0):
        sig = np.tile(np.sin(2 * np.pi * f0 * t), (2, 2, 1))
        mag = np.exp(cwt(sig, device))[..., c0:c1].mean(axis=(0, 1, 3))
        peak = cwt.freqs[int(np.argmax(mag))]
        if abs(np.log10(peak / f0)) > 2 * step * 1.01:
            raise AssertionError(f"{f0} Hz sine peaked at {peak:.2f} Hz end to end")

    # 4. every window length -> finite (N, C, F, T), and the normalisation keeps shapes
    rng = np.random.default_rng(0)
    for T in cfg_cwt.window_samples():
        x = rng.standard_normal((5, cfg_cwt.n_channels, T))
        s = cwt(x, device)
        if s.shape != (5, cfg_cwt.n_channels, cwt.n_freqs, T) or not np.isfinite(s).all():
            raise AssertionError(f"window {T}: scalogram shape {s.shape} or non-finite values")
        a, b, c = scalogram_normalize(s, s[:2], s[:2])
        if a.shape != s.shape or not np.isfinite(a).all():
            raise AssertionError(f"window {T}: scalogram_normalize broke the shape/values")
    return (f"{cwt.n_freqs} kernels ok, peaks ok, windows "
            f"{cfg_cwt.window_samples()} -> (C, {cwt.n_freqs}, T) finite")


def _batch(cfg, T, n, device, rng):
    """A z-scored random batch pushed through the model's real front-end."""
    from src.train import apply_frontend
    x = rng.standard_normal((n, cfg.n_channels, T))
    xin, _, _ = apply_frontend(cfg, x, x[:2], x[:2], device)
    return xin


def check_models(cfgs, device):
    import torch
    import torch.nn as nn
    from src.train import build_model
    from src.utils import init_weights
    rng = np.random.default_rng(1)
    lines = []
    for m, cfg in cfgs.items():
        for t_ms, T in zip(cfg.test_time_windows_ms, cfg.window_samples()):
            torch.cuda.reset_peak_memory_stats()
            model = build_model(cfg, n_channels=cfg.n_channels, n_timepoints=T).to(device)
            model.apply(init_weights)
            n_params = sum(p.numel() for p in model.parameters())
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            x = torch.as_tensor(_batch(cfg, T, cfg.batch_size, device, rng),
                                dtype=torch.float32, device=device)
            y = torch.as_tensor(rng.integers(0, 2, cfg.batch_size), device=device)
            model.train()
            out = model(x)
            if tuple(out.shape) != (cfg.batch_size, 2):
                raise AssertionError(f"{m} {t_ms} ms: output {tuple(out.shape)}")
            loss = nn.CrossEntropyLoss()(out, y)
            opt.zero_grad()
            loss.backward()
            bad = [n for n, p in model.named_parameters()
                   if p.grad is None or not torch.isfinite(p.grad).all()]
            # the positional embedding rows past this window's token count get no gradient
            bad = [n for n in bad if 'pos_embedding' not in n]
            if not torch.isfinite(loss) or bad:
                raise AssertionError(f"{m} {t_ms} ms: loss {loss.item()} / bad grads {bad[:5]}")
            opt.step()
            model.eval()
            with torch.no_grad():
                if not torch.isfinite(model(x)).all():
                    raise AssertionError(f"{m} {t_ms} ms: non-finite eval output")
            peak = torch.cuda.max_memory_allocated() / 1e9
            line = (f"{m:16s} {t_ms:5d} ms  T={T:3d}  input {tuple(x.shape[1:])}  "
                    f"tokens {model.n_tokens:3d}  params {n_params:>9,d}  peak {peak:.2f} GB")
            print("  " + line, flush=True)
            lines.append(line)
            del model, opt, x, y, out, loss
            torch.cuda.empty_cache()
    REPORT['models'] = lines
    return f"{len(lines)} model x window combinations build, train and evaluate"


def check_determinism(cfgs, device):
    import torch
    import torch.nn as nn
    from src.train import build_model
    from src.utils import init_weights

    def run_once(cfg, T):
        src.determinism.apply_all(seed=42)
        rng = np.random.default_rng(3)
        x = torch.as_tensor(_batch(cfg, T, cfg.batch_size, device, rng),
                            dtype=torch.float32, device=device)
        y = torch.as_tensor(rng.integers(0, 2, cfg.batch_size), device=device)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        model = build_model(cfg, n_channels=cfg.n_channels, n_timepoints=T).to(device)
        model.apply(init_weights)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        model.train()
        for _ in range(3):                      # dropout active -> exercises the CUDA RNG
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            opt.step()
        return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])

    for m, cfg in cfgs.items():
        T = max(cfg.window_samples())
        a, b = run_once(cfg, T), run_once(cfg, T)
        if not torch.equal(a, b):
            raise AssertionError(f"{m}: two identical runs differ (max |diff| "
                                 f"{(a - b).abs().max().item():.3e})")
    return f"bit-identical weights after 3 steps for {', '.join(cfgs)}"


def fold_sizes(labels, cfg_full):
    """(n_train_final, n_val, n_test) of one full-mode fold for pair (0, 1), computed
    with the same splitters train_pair uses."""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    mask = (labels == 0) | (labels == 1)
    y = np.where(labels[mask] == 1, 1, 0)
    idx = np.arange(len(y))
    tr_val, te = next(StratifiedKFold(cfg_full.n_folds, shuffle=True, random_state=42)
                      .split(idx, y))
    tr, va = train_test_split(tr_val, test_size=cfg_full.val_size_of_train,
                              random_state=42, stratify=y[tr_val])
    return 2 * len(tr), len(va), len(te)          # training set is doubled by augmentation


def check_timing(cfgs, device, n_subjects, sizes, n_pairs, n_gpus, walltime_h, reps):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.train import apply_frontend, build_model, make_loaders, train_epoch, validate
    from src.utils import init_weights

    n_tr, n_va, n_te = sizes
    rng = np.random.default_rng(5)
    rows, per_pair = [], 0.0
    for m, cfg in cfgs.items():
        model_pair_s = 0.0
        for t_ms, T in zip(cfg.test_time_windows_ms, cfg.window_samples()):
            X_tr = rng.standard_normal((n_tr, cfg.n_channels, T))
            X_va = rng.standard_normal((n_va, cfg.n_channels, T))
            X_te = rng.standard_normal((n_te, cfg.n_channels, T))
            y_tr, y_va, y_te = (rng.integers(0, 2, n) for n in (n_tr, n_va, n_te))

            torch.cuda.synchronize()
            t0 = time.time()
            a, b, c = apply_frontend(cfg, X_tr, X_va, X_te, device)
            torch.cuda.synchronize()
            t_front = time.time() - t0

            tl, vl, _ = make_loaders(cfg, a, y_tr, b, y_va, c, y_te, loader_seed=1)
            model = build_model(cfg, n_channels=cfg.n_channels, n_timepoints=T).to(device)
            model.apply(init_weights)
            crit = nn.CrossEntropyLoss()
            opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

            def epoch():
                train_epoch(model, tl, crit, opt, device)
                vloss, *_ = validate(model, vl, crit, device)
                sched.step(vloss)

            epoch()                                      # warm-up (cuDNN, allocator)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(reps):
                epoch()
            torch.cuda.synchronize()
            t_epoch = (time.time() - t0) / reps

            fold_s = t_front + cfg.n_epochs * t_epoch * 1.05    # +5%: checkpoint copies, test pass
            window_pair_s = fold_s * cfg.n_folds * n_subjects
            model_pair_s += window_pair_s
            rows.append(dict(model=m, window_ms=t_ms, T=T, epoch_s=round(t_epoch, 4),
                             frontend_s=round(t_front, 3), pair_min=round(window_pair_s / 60, 2)))
            print(f"  {m:16s} {t_ms:5d} ms: epoch {t_epoch * 1000:7.1f} ms, front-end "
                  f"{t_front * 1000:7.1f} ms -> {window_pair_s / 60:6.2f} min per pair", flush=True)
            del model, opt, tl, vl, a, b, c
            torch.cuda.empty_cache()
        print(f"  {m:16s} total: {model_pair_s / 60:6.1f} min per pair", flush=True)
        per_pair += model_pair_s

    pairs_per_gpu = math.ceil(n_pairs / n_gpus)
    load_s = 300                                          # worker start-up + data load, generous
    projected_h = (per_pair * pairs_per_gpu + load_s) * 1.10 / 3600   # +10% margin
    REPORT['timing'] = dict(rows=rows, per_pair_min=round(per_pair / 60, 1),
                            n_pairs=n_pairs, n_gpus=n_gpus, pairs_per_gpu=pairs_per_gpu,
                            projected_hours=round(projected_h, 2), walltime_hours=walltime_h)
    msg = (f"{per_pair / 60:.0f} min per pair (all models) x {pairs_per_gpu} pairs on the "
           f"busiest of {n_gpus} GPUs -> ~{projected_h:.1f} h projected, walltime {walltime_h:g} h")
    if projected_h > walltime_h:
        raise RuntimeError(
            msg + ". The full sweep would time out: raise '#SBATCH -t' in submit_ls6.slurm "
            "and FULL_WALLTIME_H together (gpu-a100 allows up to 48 h), or plan to resubmit "
            "(finished pairs are skipped).")
    if projected_h > 0.8 * walltime_h:
        print(f"WARNING: projection uses {projected_h / walltime_h:.0%} of the walltime; "
              f"little margin if the node is slower.", flush=True)
    return msg


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Preflight checks for the decimated Conformer sweep.")
    ap.add_argument('--dataset-root', default=DATASET_ROOT_DEFAULT)
    ap.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    ap.add_argument('--mode', choices=list(MODE_OVERRIDES), default='smoke',
                    help="results/<mode>/ gets the write probe and the preflight report")
    ap.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    ap.add_argument('--skip-timing', action='store_true',
                    help="skip the full-sweep time projection")
    ap.add_argument('--walltime-hours', type=float, default=24.0,
                    help="walltime of the FULL job (#SBATCH -t) the projection must fit in")
    ap.add_argument('--project-pairs', type=int, default=len(PAIRS),
                    help="pairs in the full sweep (default 28)")
    ap.add_argument('--n-gpus', type=int, default=3,
                    help="GPUs the full sweep spreads pairs over (default 3 on LS6 gpu-a100)")
    ap.add_argument('--timing-epochs', type=int, default=5,
                    help="timed epochs per model x window (after one warm-up)")
    args = ap.parse_args()

    import torch
    models = list(dict.fromkeys(args.models))
    # the checks always use the FULL-mode config: that is what has to survive the sweep
    cfgs = {m: make_config(m, 'full', args.dataset_root, args.results_root) for m in models}
    cfg0 = cfgs[models[0]]
    root = mode_root(args.results_root, args.mode)

    print("=" * 72, flush=True)
    print(f"PREFLIGHT  models={models}  dataset={args.dataset_root}", flush=True)
    print(f"  decim {cfg0.decim}: {cfg0.orig_sampling_rate} -> {cfg0.sampling_rate} Hz | windows "
          f"{list(cfg0.test_time_windows_ms)} ms = {cfg0.window_samples()} samples | "
          f"full: {cfg0.n_folds} folds x {cfg0.n_epochs} epochs", flush=True)
    print("=" * 72, flush=True)

    run_check('packages', check_packages)
    cuda_ok = run_check('cuda', check_cuda) is not None
    run_check('results folder writable', check_results_writable, root)
    run_check('pairs', check_pairs)
    data = run_check('data (all subjects)', check_data, args.dataset_root, cfg0)

    if cuda_ok:
        device = torch.device('cuda')
        cwt_cfgs = [cfgs[m] for m in models if m in CWT_MODELS]
        if cwt_cfgs:
            run_check('cwt front-end', check_cwt, cwt_cfgs[0], device)
        run_check('models x windows (full config)', check_models, cfgs, device)
        run_check('determinism', check_determinism, cfgs, device)
        if not args.skip_timing:
            if data is None:
                RESULTS.append(('FAIL', 'timing', 'needs the data check to pass'))
                print("[FAIL] timing: needs the data check to pass", flush=True)
            else:
                sizes = fold_sizes(data['first_labels'], cfg0)
                print(f"  full-mode fold: train {sizes[0]} (incl. augmentation), val {sizes[1]}, "
                      f"test {sizes[2]}; {data['n_subjects']} subjects x {cfg0.n_folds} folds",
                      flush=True)
                run_check('timing projection (full sweep)', check_timing, cfgs, device,
                          data['n_subjects'], sizes, args.project_pairs, args.n_gpus,
                          args.walltime_hours, args.timing_epochs)
    else:
        print("\nSkipping GPU checks (cwt, models, determinism, timing): no CUDA.", flush=True)
        RESULTS.append(('FAIL', 'gpu checks', 'skipped because CUDA is unavailable'))

    failed = [r for r in RESULTS if r[0] == 'FAIL']
    REPORT['checks'] = [dict(status=s, name=n, detail=d) for s, n, d in RESULTS]
    try:
        out_dir = os.path.join(root, 'preflight')
        os.makedirs(out_dir, exist_ok=True)
        tag = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(out_dir, f"preflight_{tag}.json"), 'w') as f:
            json.dump(REPORT, f, indent=2, default=str)
    except OSError as e:
        print(f"(could not write the preflight report: {e})", flush=True)

    print("\n" + "=" * 72, flush=True)
    for s, n, d in RESULTS:
        print(f"  [{s:4s}] {n}{': ' + d if d else ''}", flush=True)
    print("=" * 72, flush=True)
    if failed:
        print(f"PREFLIGHT FAILED: {len(failed)} check(s) - fix these before the full sweep.",
              flush=True)
        return 1
    print("PREFLIGHT PASSED", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
