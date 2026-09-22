import src.determinism  # noqa: F401 — must be first (sets CUBLAS env before torch loads)
# Preflight for Slim Conformer B: everything a run needs, checked in a few minutes,
# no training run. scripts/local_test.py and submit_ls6.slurm run it first and stop
# if any check fails.
#
#   python -m scripts.preflight --allow-cpu --project-pairs 3 --n-gpus 1
#   python -m scripts.preflight --skip-timing
#
# Checks (each prints [ok] or [FAIL]):
#   packages     torch, numpy, scipy, sklearn, pandas import
#   device       a GPU is visible (or --allow-cpu), strict determinism settings apply
#   results      the results folder is writable
#   pairs        all 28 pairs parse to 28 distinct folders
#   data         EVERY subject file loads, decimates to 50 Hz, has 62 channels, finite
#                values, all 8 classes, enough trials per class for 5-fold CV
#   reference    Conformer A / B results to compare against exist (warning only)
#   model        every window (full config) builds with EXACTLY 25,752 parameters, runs
#                forward + backward + an Adam step on a batch of 32 under the strict
#                deterministic algorithms, finite loss and gradients
#   determinism  training twice from the same seed gives bit-identical weights
#   timing       full-mode time projection for --project-pairs pairs on --n-gpus GPUs;
#                fails only if --walltime-hours is given and the projection exceeds it

import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np

from src.config import (DATASET_ROOT_DEFAULT, RESULTS_ROOT_DEFAULT, REFERENCE_ROOT_DEFAULT,
                        REFERENCE_MODELS, MODELS, EXPECTED_PARAMS, MODE_OVERRIDES, PAIRS,
                        DEFAULT_PAIRS, make_config, mode_root, pair_slug, parse_pairs)

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
    summary, value = detail, detail
    if isinstance(detail, tuple):
        summary, value = detail
    summary = summary if isinstance(summary, str) else ''
    print(f"[ok]   {name}{': ' + summary if summary else ''}  ({time.time() - t0:.1f}s)",
          flush=True)
    RESULTS.append(('ok', name, summary))
    return value if value is not None else True


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
def check_packages():
    import torch, scipy, sklearn, pandas
    from scipy import stats  # noqa: F401 — compare_to_reference needs it
    v = (f"python {sys.version.split()[0]}, torch {torch.__version__}, numpy {np.__version__}, "
         f"scipy {scipy.__version__}, sklearn {sklearn.__version__}, pandas {pandas.__version__}")
    REPORT['packages'] = v
    return v


def check_device(allow_cpu):
    import torch
    src.determinism.apply_all(seed=42)          # the exact settings every run uses
    if not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("CUDA is not available (is this a GPU node, and is the venv's "
                               "torch a CUDA build?). Pass --allow-cpu to run on the CPU.")
        print("WARNING: no GPU visible - everything will run on the CPU (much slower).",
              flush=True)
        REPORT['device'] = 'cpu'
        return 'CPU (no CUDA device visible)', 'cpu'
    props = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    info = (f"{name}, {props.total_memory / 1e9:.1f} GB, cuda {torch.version.cuda}, "
            f"visible devices {torch.cuda.device_count()}")
    REPORT['device'] = info
    return info, 'cuda'


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
    """Loads every subject exactly as a run does (decimated, cached per process)."""
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
            problems.append(f"{tag}: classes {missing} missing")
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
        for pr in problems:
            print(f"  problem: {pr}", flush=True)
        raise RuntimeError(f"{len(problems)} data problem(s), see above")
    shapes = ', '.join(f"(C={c}, T={t} @ {cfg_full.sampling_rate} Hz, was {o})"
                       for c, t, o in sorted(lengths))
    REPORT['data'] = dict(n_subjects=len(subjects), shapes=sorted(lengths),
                          class_counts={str(k): v for k, v in counts.items()})
    lo = min(min(v.values()) for v in counts.values())
    hi = max(max(v.values()) for v in counts.values())
    return (f"{len(subjects)} subjects, {shapes}, trials/class {lo}-{hi}",
            dict(n_subjects=len(subjects), first_labels=subjects[0][2]))


def check_reference(reference_root, pairs):
    """Warn (never fail) if the Conformer A / B results to compare against are missing."""
    root = os.path.join(reference_root, 'full')
    missing = []
    for a, b in pairs:
        for m in REFERENCE_MODELS:
            if not os.path.isfile(os.path.join(root, m, pair_slug(a, b), 'wide_subject_x_window.csv')):
                missing.append(f"{pair_slug(a, b)} [{m}]")
    if missing:
        print(f"WARNING: no reference results for {', '.join(missing)} under {root}. "
              f"The comparison step will skip them.", flush=True)
        return f"{len(missing)} reference run(s) missing (comparison will skip them)"
    return f"Conformer A and B results found for {len(pairs)} pair(s) in {root}"


def check_model(cfg, device):
    import torch
    import torch.nn as nn
    from src.train import build_model, count_params
    from src.utils import init_weights
    rng = np.random.default_rng(1)
    want = EXPECTED_PARAMS[cfg.model]
    lines = []
    for t_ms, T in zip(cfg.test_time_windows_ms, cfg.window_samples()):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        model = build_model(cfg, n_channels=cfg.n_channels, n_timepoints=T).to(device)
        model.apply(init_weights)
        n_params = count_params(model)
        if n_params != want:
            raise AssertionError(f"{t_ms} ms: model has {n_params:,} parameters, expected {want:,}")
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        x = torch.as_tensor(rng.standard_normal((cfg.batch_size, cfg.n_channels, T)),
                            dtype=torch.float32, device=device)
        y = torch.as_tensor(rng.integers(0, 2, cfg.batch_size), device=device)
        model.train()
        out = model(x)
        if tuple(out.shape) != (cfg.batch_size, 2):
            raise AssertionError(f"{t_ms} ms: output {tuple(out.shape)}")
        loss = nn.CrossEntropyLoss()(out, y)
        opt.zero_grad()
        loss.backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
        if not torch.isfinite(loss) or bad:
            raise AssertionError(f"{t_ms} ms: loss {loss.item()} / bad grads {bad[:5]}")
        opt.step()
        model.eval()
        with torch.no_grad():
            if not torch.isfinite(model(x)).all():
                raise AssertionError(f"{t_ms} ms: non-finite eval output")
        peak = (torch.cuda.max_memory_allocated() / 1e9) if device.type == 'cuda' else float('nan')
        line = (f"{cfg.model} {t_ms:5d} ms  T={T:3d}  tokens {model.n_tokens:3d}  "
                f"params {n_params:,}  peak {peak:.2f} GB")
        print("  " + line, flush=True)
        lines.append(line)
        del model, opt, x, y, out, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    REPORT['model'] = lines
    return f"{want:,} parameters at every window; forward, backward and Adam step ok"


def check_determinism(cfg, device):
    import torch
    import torch.nn as nn
    from src.train import build_model
    from src.utils import init_weights

    def run_once(T):
        src.determinism.apply_all(seed=42)
        rng = np.random.default_rng(3)
        x = torch.as_tensor(rng.standard_normal((cfg.batch_size, cfg.n_channels, T)),
                            dtype=torch.float32, device=device)
        y = torch.as_tensor(rng.integers(0, 2, cfg.batch_size), device=device)
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        model = build_model(cfg, n_channels=cfg.n_channels, n_timepoints=T).to(device)
        model.apply(init_weights)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        model.train()
        for _ in range(3):                      # dropout active -> exercises the RNG
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            opt.step()
        return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])

    T = max(cfg.window_samples())
    a, b = run_once(T), run_once(T)
    if not torch.equal(a, b):
        raise AssertionError(f"two identical runs differ (max |diff| {(a - b).abs().max().item():.3e})")
    return "bit-identical weights after 3 training steps"


def fold_sizes(labels, cfg_full):
    """(n_train_final, n_val, n_test) of one full-mode fold, computed with the same
    splitters train_pair uses."""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    mask = (labels == 0) | (labels == 1)
    y = np.where(labels[mask] == 1, 1, 0)
    idx = np.arange(len(y))
    tr_val, te = next(StratifiedKFold(cfg_full.n_folds, shuffle=True, random_state=42)
                      .split(idx, y))
    tr, va = train_test_split(tr_val, test_size=cfg_full.val_size_of_train,
                              random_state=42, stratify=y[tr_val])
    return 2 * len(tr), len(va), len(te)          # training set is doubled by augmentation


def check_timing(cfg, device, n_subjects, sizes, n_pairs, n_gpus, walltime_h, reps):
    import torch.nn as nn
    import torch.optim as optim
    from src.train import build_model, make_loaders, train_epoch, validate
    from src.utils import init_weights

    n_tr, n_va, n_te = sizes
    rng = np.random.default_rng(5)
    rows, per_pair = [], 0.0
    for t_ms, T in zip(cfg.test_time_windows_ms, cfg.window_samples()):
        X_tr = rng.standard_normal((n_tr, cfg.n_channels, T))
        X_va = rng.standard_normal((n_va, cfg.n_channels, T))
        X_te = rng.standard_normal((n_te, cfg.n_channels, T))
        y_tr, y_va, y_te = (rng.integers(0, 2, n) for n in (n_tr, n_va, n_te))

        tl, vl, _ = make_loaders(cfg, X_tr, y_tr, X_va, y_va, X_te, y_te, loader_seed=1)
        model = build_model(cfg, n_channels=cfg.n_channels, n_timepoints=T).to(device)
        model.apply(init_weights)
        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

        def epoch():
            train_epoch(model, tl, crit, opt, device)
            vloss, *_ = validate(model, vl, crit, device)
            sched.step(vloss)

        epoch()                                          # warm-up (cuDNN, allocator)
        _sync()
        t0 = time.time()
        for _ in range(reps):
            epoch()
        _sync()
        t_epoch = (time.time() - t0) / reps

        fold_s = cfg.n_epochs * t_epoch * 1.05           # +5%: checkpoint copies, test pass
        window_pair_s = fold_s * cfg.n_folds * n_subjects
        per_pair += window_pair_s
        rows.append(dict(window_ms=t_ms, T=T, epoch_s=round(t_epoch, 4),
                         pair_min=round(window_pair_s / 60, 2)))
        print(f"  {t_ms:5d} ms: epoch {t_epoch * 1000:7.1f} ms -> {window_pair_s / 60:6.2f} min "
              f"per pair", flush=True)
        del model, opt, tl, vl

    pairs_per_gpu = math.ceil(n_pairs / max(1, n_gpus))
    load_s = 120                                          # start-up + data load
    projected_h = (per_pair * pairs_per_gpu + load_s) * 1.10 / 3600   # +10% margin
    REPORT['timing'] = dict(rows=rows, per_pair_min=round(per_pair / 60, 1), n_pairs=n_pairs,
                            n_gpus=n_gpus, pairs_per_gpu=pairs_per_gpu,
                            projected_hours=round(projected_h, 2), walltime_hours=walltime_h)
    msg = (f"about {per_pair / 60:.0f} min per pair x {pairs_per_gpu} pair(s) per device "
           f"-> about {projected_h:.1f} h for a full-mode run of {n_pairs} pair(s) on "
           f"{n_gpus} device(s)")
    if walltime_h and projected_h > walltime_h:
        raise RuntimeError(msg + f", over the {walltime_h:g} h walltime. Raise '#SBATCH -t' "
                           "and FULL_WALLTIME_H together, or plan to resubmit.")
    return msg


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Preflight checks for Slim Conformer B.")
    ap.add_argument('--dataset-root', default=DATASET_ROOT_DEFAULT)
    ap.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    ap.add_argument('--reference-root', default=REFERENCE_ROOT_DEFAULT,
                    help="Conformer_decimated_CWT/results, for the Conformer A / B comparison")
    ap.add_argument('--pairs', nargs='+', default=list(DEFAULT_PAIRS),
                    help="pairs whose reference results should exist (default: the 3 pilot pairs)")
    ap.add_argument('--mode', choices=list(MODE_OVERRIDES), default='smoke',
                    help="results/<mode>/ gets the write probe and the preflight report")
    ap.add_argument('--allow-cpu', action='store_true',
                    help="run on the CPU when no GPU is visible (local machines)")
    ap.add_argument('--skip-timing', action='store_true')
    ap.add_argument('--walltime-hours', type=float, default=0.0,
                    help="fail if the projection exceeds this (0 = report only)")
    ap.add_argument('--project-pairs', type=int, default=len(DEFAULT_PAIRS),
                    help="pairs in the full-mode run to project (default 3)")
    ap.add_argument('--n-gpus', type=int, default=1,
                    help="devices the full-mode run spreads pairs over (default 1)")
    ap.add_argument('--timing-epochs', type=int, default=5,
                    help="timed epochs per window (after one warm-up)")
    args = ap.parse_args()

    import torch
    model_name = MODELS[0]
    cfg = make_config(model_name, 'full', args.dataset_root, args.results_root)
    root = mode_root(args.results_root, args.mode)
    try:
        pairs = parse_pairs(args.pairs)
    except ValueError as e:
        ap.error(str(e))

    print("=" * 72, flush=True)
    print(f"PREFLIGHT  model={model_name}  dataset={args.dataset_root}", flush=True)
    print(f"  decim {cfg.decim}: {cfg.orig_sampling_rate} -> {cfg.sampling_rate} Hz | windows "
          f"{list(cfg.test_time_windows_ms)} ms = {cfg.window_samples()} samples | "
          f"full: {cfg.n_folds} folds x {cfg.n_epochs} epochs", flush=True)
    print("=" * 72, flush=True)

    run_check('packages', check_packages)
    dev_kind = run_check('device', check_device, args.allow_cpu)
    run_check('results folder writable', check_results_writable, root)
    run_check('pairs', check_pairs)
    data = run_check('data (all subjects)', check_data, args.dataset_root, cfg)
    run_check('reference results (Conformer A / B)', check_reference, args.reference_root, pairs)

    if dev_kind:
        device = torch.device(dev_kind)
        run_check(f'model: {EXPECTED_PARAMS[model_name]:,} parameters, every window', check_model,
                  cfg, device)
        run_check('determinism', check_determinism, cfg, device)
        if not args.skip_timing:
            if not isinstance(data, dict):
                RESULTS.append(('FAIL', 'timing', 'needs the data check to pass'))
                print("[FAIL] timing: needs the data check to pass", flush=True)
            else:
                sizes = fold_sizes(data['first_labels'], cfg)
                print(f"  full-mode fold: train {sizes[0]} (incl. augmentation), val {sizes[1]}, "
                      f"test {sizes[2]}; {data['n_subjects']} subjects x {cfg.n_folds} folds",
                      flush=True)
                run_check('timing projection (full mode)', check_timing, cfg, device,
                          data['n_subjects'], sizes, args.project_pairs, args.n_gpus,
                          args.walltime_hours, args.timing_epochs)
    else:
        print("\nSkipping model, determinism and timing checks: no usable device.", flush=True)
        RESULTS.append(('FAIL', 'model checks', 'skipped because no device passed'))

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
        print(f"PREFLIGHT FAILED: {len(failed)} check(s).", flush=True)
        return 1
    print("PREFLIGHT PASSED", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
