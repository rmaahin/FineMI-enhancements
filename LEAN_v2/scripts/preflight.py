import src.determinism  # noqa: F401 — must be first (sets CUBLAS env before torch loads)
# Preflight for LEAN_v2: everything a run needs, checked in a few minutes, no real
# training. run_vast.sh runs it first and stops if any check fails.
#
#   python -m scripts.preflight --models all --n-slots 6
#   python -m scripts.preflight --models ladder --skip-timing
#
# Checks (each prints [ok] or [FAIL]):
#   packages     torch, numpy, scipy, sklearn, pandas import
#   device       a GPU is visible (or --allow-cpu), strict determinism settings apply
#   results      the results folder is writable
#   data         EVERY subject file loads: 1000-sample (0-4 s) epochs at 250 Hz, 200 at
#                50 Hz, 62 channels, finite, all 8 classes, enough trials per class
#   band         most of the EEG power lies in 0.5-3 Hz
#   LEAN layout  lean_v2_r0 has exactly LEAN's 25,752 parameters in LEAN's layout
#                (=> same initial weights as the paper's LEAN for the same seed)
#   model        every selected preset at every window: EXACTLY the parameters
#                src/param_budget.py predicts; one real training epoch on random data
#                through the preset's own path (SupCon, crops, schedule, weight
#                averaging) under the strict deterministic algorithms; finite outputs
#   determinism  the same training steps twice give bit-identical weights
#   timing       projected hours for a screen run and a full run of the selected
#                presets on --n-slots worker slots (optimistic when several workers
#                share a GPU)

import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np

from src.config import (DATASET_ROOT_DEFAULT, RESULTS_ROOT_DEFAULT, EXPECTED_EPOCH_SAMPLES,
                        MODEL_LABELS, MODE_OVERRIDES, BAND_HZ, BAND_LABEL, SCREEN_PAIRS,
                        expand_models, make_config, mode_root)
from src.param_budget import expected_params, LEAN_LAYOUT

RESULTS = []          # (status, name, detail)
REPORT = {}           # written to results/<mode>/preflight/
MIN_BAND_POWER_SHARE = 0.5


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
    from scipy import stats, signal  # noqa: F401
    v = (f"python {sys.version.split()[0]}, torch {torch.__version__}, numpy {np.__version__}, "
         f"scipy {scipy.__version__}, sklearn {sklearn.__version__}, pandas {pandas.__version__}")
    REPORT['packages'] = v
    return v


def check_device(allow_cpu):
    import torch
    src.determinism.apply_all(seed=42)
    if not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("CUDA is not available. Pass --allow-cpu to run on the CPU.")
        REPORT['device'] = 'cpu'
        return 'CPU (no CUDA device visible)', 'cpu'
    props = torch.cuda.get_device_properties(0)
    info = (f"{torch.cuda.get_device_name(0)}, {props.total_memory / 1e9:.1f} GB, "
            f"cuda {torch.version.cuda}, visible devices {torch.cuda.device_count()}")
    REPORT['device'] = info
    return info, 'cuda'


def check_results_writable(root):
    os.makedirs(root, exist_ok=True)
    probe = os.path.join(root, f".write_probe_{os.getpid()}")
    with open(probe, 'w') as f:
        f.write('ok')
    os.remove(probe)
    return root


def check_data(dataset_root, cfg):
    from src.data import load_subjects
    subjects = load_subjects(dataset_root, cfg.band_tag, cfg.decim)
    max_win = max(cfg.window_samples())
    problems, warnings = [], []
    lengths, counts = set(), {}
    for sid, data, labels, orig_len in subjects:
        tag = f"subject {sid}"
        if data.ndim != 3:
            problems.append(f"{tag}: data is {data.ndim}-D")
            continue
        if orig_len != EXPECTED_EPOCH_SAMPLES:
            problems.append(f"{tag}: epochs have {orig_len} samples at 250 Hz, expected "
                            f"{EXPECTED_EPOCH_SAMPLES} (0 to 4 s). 1126 = the old -0.5 to "
                            f"4.0 s files - check --dataset-root")
        if data.shape[0] != 320:
            warnings.append(f"{tag}: {data.shape[0]} trials (the dataset paper has 320)")
        if data.shape[1] != cfg.n_channels:
            problems.append(f"{tag}: {data.shape[1]} channels, expected {cfg.n_channels}")
        if data.shape[2] < max_win:
            problems.append(f"{tag}: {data.shape[2]} samples, need {max_win}")
        if data.shape[2] != math.ceil(orig_len / cfg.decim):
            problems.append(f"{tag}: length {data.shape[2]} != ceil({orig_len}/{cfg.decim})")
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
        few = {k: n for k, n in c.items() if 0 <= k < 8 and n < cfg.n_folds}
        if few:
            problems.append(f"{tag}: classes {few} have fewer than {cfg.n_folds} trials")
        lengths.add((data.shape[1], data.shape[2], orig_len))
    if len(subjects) != 18:
        print(f"WARNING: found {len(subjects)} subject files, expected 18", flush=True)
    for w in warnings:
        print(f"  WARNING: {w}", flush=True)
    if problems:
        for pr in problems:
            print(f"  problem: {pr}", flush=True)
        raise RuntimeError(f"{len(problems)} data problem(s), see above")
    shapes = ', '.join(f"(C={c}, T={t} @ {cfg.sampling_rate} Hz, {o} @ 250 Hz)"
                       for c, t, o in sorted(lengths))
    REPORT['data'] = dict(n_subjects=len(subjects), shapes=sorted(lengths), warnings=warnings)
    lo = min(min(v.values()) for v in counts.values())
    hi = max(max(v.values()) for v in counts.values())
    return (f"{len(subjects)} subjects, {shapes}, trials/class {lo}-{hi}"
            + (f", {len(warnings)} warning(s)" if warnings else ""),
            dict(n_subjects=len(subjects), first_labels=subjects[0][2]))


def check_band(dataset_root, cfg):
    from scipy.signal import welch
    from src.data import load_subjects
    lo_hz, hi_hz = BAND_HZ
    shares, problems = {}, []
    for sid, data, _labels, _o in load_subjects(dataset_root, cfg.band_tag, cfg.decim,
                                                verbose=False):
        f, pxx = welch(data, fs=cfg.sampling_rate, nperseg=data.shape[-1], axis=-1)
        psd = pxx.mean(axis=(0, 1))
        sel = (f >= 0.1) & (f <= 25)
        total = psd[sel].sum()
        share = float(psd[(f >= lo_hz) & (f <= hi_hz)].sum() / total) if total > 0 else 0.0
        shares[sid] = share
        if share < MIN_BAND_POWER_SHARE:
            problems.append(f"subject {sid}: only {100 * share:.0f}% of the power is in {BAND_LABEL}")
    REPORT['band'] = dict(band_hz=list(BAND_HZ), share_by_subject=shares)
    if problems:
        for pr in problems:
            print(f"  problem: {pr}", flush=True)
        raise RuntimeError(f"{len(problems)} subject(s) do not look like {BAND_LABEL} data")
    return (f"{100 * min(shares.values()):.0f}-{100 * max(shares.values()):.0f}% of the "
            f"0.1-25 Hz power lies in {BAND_LABEL}")


def check_lean_layout(dataset_root, results_root):
    """lean_v2_r0 must be LEAN: 25,752 parameters with LEAN's exact layout."""
    from src.train import build_model, count_params
    cfg = make_config('lean_v2_r0', 'full', dataset_root, results_root)
    model = build_model(cfg, cfg.n_channels, max(cfg.window_samples()))
    n = count_params(model)
    shapes = [tuple(p.shape) for p in model.parameters()]
    if n != 25_752:
        raise AssertionError(f"lean_v2_r0 has {n:,} parameters, LEAN has 25,752")
    if shapes != [tuple(s) for s in LEAN_LAYOUT]:
        first = next(i for i, (a, b) in enumerate(zip(shapes + [None] * 99, LEAN_LAYOUT + [None] * 99))
                     if a != b)
        raise AssertionError(f"lean_v2_r0's parameter layout differs from LEAN's at tensor "
                             f"{first}: {shapes[first:first + 3]} vs {LEAN_LAYOUT[first:first + 3]}")
    return f"25,752 parameters in LEAN's layout ({len(shapes)} tensors)"


def _random_loaders(cfg, T, n_train, n_val, seed):
    from src.train import make_loaders
    rng = np.random.default_rng(seed)
    X_tr = rng.standard_normal((n_train, cfg.n_channels, T))
    X_va = rng.standard_normal((n_val, cfg.n_channels, T))
    y_tr = np.tile([0, 1], n_train // 2 + 1)[:n_train]
    y_va = np.tile([0, 1], n_val // 2 + 1)[:n_val]
    return make_loaders(cfg, X_tr, y_tr, X_va, y_va, X_va, y_va, loader_seed=1)


def _fresh(cfg, T, device, seed=1234):
    import torch
    from src.train import build_model, build_projection
    from src.utils import init_weights
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(cfg, cfg.n_channels, T).to(device)
    model.apply(init_weights)
    proj = build_projection(cfg, model)
    if proj is not None:
        proj = proj.to(device)
        proj.apply(init_weights)
    return model, proj


def check_model(cfg, device):
    """Exact parameter count at every window, then a real 2-epoch fit_fold (so the
    preset's schedule, averaging, SupCon and crops all run) and an evaluation."""
    import copy
    import torch
    import torch.nn as nn
    from src.train import count_params, fit_fold, validate
    lines = []
    for t_ms, T in zip(cfg.test_time_windows_ms, cfg.window_samples()):
        want = expected_params(cfg, T)
        model, proj = _fresh(cfg, T, device)
        n = count_params(model)
        if n != want:
            raise AssertionError(f"{t_ms} ms: model has {n:,} parameters, param_budget "
                                 f"predicts {want:,}")
        c2 = copy.deepcopy(cfg)
        c2.n_epochs = 2
        c2.train['avg_last'] = 2
        tl, vl, _ = _random_loaders(cfg, T, n_train=2 * cfg.batch_size, n_val=16, seed=t_ms)
        gen = torch.Generator()
        gen.manual_seed(7)
        fit_fold(c2, model, proj, tl, vl, device, log_prefix='', log_epochs=False, crop_gen=gen)
        bad = [k for k, p in model.state_dict().items()
               if p.is_floating_point() and not torch.isfinite(p).all()]
        if bad:
            raise AssertionError(f"{t_ms} ms: non-finite weights after training: {bad[:5]}")
        loss, acc, _, _ = validate(model, vl, nn.CrossEntropyLoss(), device, cfg)
        if not np.isfinite(loss):
            raise AssertionError(f"{t_ms} ms: non-finite eval loss")
        line = (f"{cfg.model:22s} {t_ms:5d} ms  T={T:4d} (model input "
                f"{cfg.model_samples(T)})  params {n:,}")
        print("  " + line, flush=True)
        lines.append(line)
        del model, proj
    REPORT.setdefault('model', []).extend(lines)
    counts = sorted({expected_params(cfg, T) for T in cfg.window_samples()})
    return (f"{' / '.join(f'{c:,}' for c in counts)} parameters as predicted; "
            f"{cfg.train['schedule']}"
            + (f", SupCon {cfg.train['supcon_weight']}" if cfg.train['supcon_weight'] else '')
            + (f", {cfg.train['n_crops']} crops" if cfg.train['n_crops'] else '')
            + " run at every window")


def check_determinism(cfg, device):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.train import train_epoch

    def run_once(T):
        src.determinism.apply_all(seed=42)
        model, proj = _fresh(cfg, T, device)
        tl, _, _ = _random_loaders(cfg, T, n_train=3 * cfg.batch_size, n_val=8, seed=3)
        params = list(model.parameters()) + (list(proj.parameters()) if proj else [])
        opt = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        gen = torch.Generator()
        gen.manual_seed(11)
        train_epoch(model, proj, tl, nn.CrossEntropyLoss(), opt, device, cfg, gen)
        return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])

    T = max(cfg.window_samples())
    a, b = run_once(T), run_once(T)
    if not torch.equal(a, b):
        raise AssertionError(f"two identical runs differ (max |diff| {(a - b).abs().max().item():.3e})")
    return "bit-identical weights after 3 training steps"


def fold_sizes(labels, cfg):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    mask = (labels == 0) | (labels == 1)
    y = np.where(labels[mask] == 1, 1, 0)
    idx = np.arange(len(y))
    tr_val, te = next(StratifiedKFold(cfg.n_folds, shuffle=True, random_state=42).split(idx, y))
    tr, va = train_test_split(tr_val, test_size=cfg.val_size_of_train, random_state=42,
                              stratify=y[tr_val])
    return 2 * len(tr), len(va), len(te)          # training set doubled by augmentation


def check_timing(models, dataset_root, results_root, device, n_subjects, sizes, n_slots,
                 walltime_h, reps):
    """Epoch times per preset and window, projected to a screen and a full run."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.train import train_epoch, validate

    n_tr, n_va, _ = sizes
    epoch_s = {}                                   # (model, T) -> seconds per epoch
    for m in models:
        cfg = make_config(m, 'full', dataset_root, results_root)
        for t_ms, T in zip(cfg.test_time_windows_ms, cfg.window_samples()):
            model, proj = _fresh(cfg, T, device)
            tl, vl, _ = _random_loaders(cfg, T, n_train=n_tr, n_val=n_va, seed=5)
            params = list(model.parameters()) + (list(proj.parameters()) if proj else [])
            opt = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
            crit = nn.CrossEntropyLoss()
            gen = torch.Generator()
            gen.manual_seed(5)
            per_epoch_val = cfg.train['schedule'] == 'best_val'

            def epoch():
                train_epoch(model, proj, tl, crit, opt, device, cfg, gen)
                if per_epoch_val:
                    validate(model, vl, crit, device, cfg)

            epoch()
            _sync()
            t0 = time.time()
            for _ in range(reps):
                epoch()
            _sync()
            epoch_s[(m, T)] = (time.time() - t0) / reps
            del model, proj, opt, tl, vl

    out, rows = [], []
    for mode in ('screen', 'full'):
        n_pairs = len(SCREEN_PAIRS) if mode == 'screen' else 28
        per_pair = 0.0
        for m in models:
            cfg = make_config(m, mode, dataset_root, results_root)
            m_pair = sum(cfg.n_epochs * epoch_s[(m, T)] * 1.05 * cfg.n_folds * n_subjects
                         for T in cfg.window_samples())
            per_pair += m_pair
            rows.append(dict(mode=mode, model=m, pair_min=round(m_pair / 60, 2)))
        pairs_per_slot = math.ceil(n_pairs / max(1, n_slots))
        hours = (per_pair * pairs_per_slot + 120) * 1.10 / 3600
        out.append(f"{mode}: {per_pair / 60:.1f} min per pair for {len(models)} preset(s) x "
                   f"{pairs_per_slot} pair(s) per slot -> about {hours:.1f} h on {n_slots} slot(s)")
        REPORT.setdefault('timing', {})[mode] = dict(per_pair_min=round(per_pair / 60, 1),
                                                     pairs_per_slot=pairs_per_slot,
                                                     projected_hours=round(hours, 2))
        if mode == 'full' and walltime_h and hours > walltime_h:
            raise RuntimeError(f"full run projected at {hours:.1f} h, over {walltime_h:g} h")
    for r in rows:
        print(f"  {r['mode']:6s} {r['model']:22s} {r['pair_min']:7.2f} min per pair", flush=True)
    REPORT['timing']['rows'] = rows
    return '; '.join(out)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Preflight checks for LEAN_v2.")
    ap.add_argument('--dataset-root', default=DATASET_ROOT_DEFAULT)
    ap.add_argument('--results-root', default=RESULTS_ROOT_DEFAULT)
    ap.add_argument('--models', nargs='+', default=['all'])
    ap.add_argument('--mode', choices=list(MODE_OVERRIDES), default='smoke',
                    help="results/<mode>/ gets the write probe and the preflight report")
    ap.add_argument('--allow-cpu', action='store_true')
    ap.add_argument('--skip-timing', action='store_true')
    ap.add_argument('--walltime-hours', type=float, default=0.0)
    ap.add_argument('--n-slots', type=int, default=1,
                    help="worker slots (GPUs x workers per GPU) for the time projection")
    ap.add_argument('--timing-epochs', type=int, default=3)
    args = ap.parse_args()

    import torch
    try:
        models = expand_models(args.models)
    except ValueError as e:
        ap.error(str(e))
    cfg_full = make_config(models[0], 'full', args.dataset_root, args.results_root)
    root = mode_root(args.results_root, args.mode)

    print("=" * 72, flush=True)
    print(f"PREFLIGHT  {len(models)} preset(s): {', '.join(models)}", flush=True)
    print(f"  dataset={args.dataset_root}  band={BAND_LABEL}  {cfg_full.sampling_rate} Hz",
          flush=True)
    print("=" * 72, flush=True)

    run_check('packages', check_packages)
    dev_kind = run_check('device', check_device, args.allow_cpu)
    run_check('results folder writable', check_results_writable, root)
    data = run_check(f'data (all subjects, {cfg_full.sampling_rate} Hz)', check_data,
                     args.dataset_root, cfg_full)
    if isinstance(data, dict):
        run_check(f'band ({BAND_LABEL})', check_band, args.dataset_root, cfg_full)

    if dev_kind:
        device = torch.device(dev_kind)
        run_check('LEAN layout (lean_v2_r0)', check_lean_layout, args.dataset_root,
                  args.results_root)
        for m in models:
            # model checks use the run's own windows (smoke/full: 4, screen: 2)
            c = make_config(m, 'full', args.dataset_root, args.results_root)
            run_check(f'model {m} ({MODEL_LABELS[m]})', check_model, c, device)
            run_check(f'determinism {m}', check_determinism, c, device)
        if not args.skip_timing:
            if not isinstance(data, dict):
                RESULTS.append(('FAIL', 'timing', 'needs the data check to pass'))
                print("[FAIL] timing: needs the data check to pass", flush=True)
            else:
                sizes = fold_sizes(data['first_labels'], cfg_full)
                print(f"  fold: train {sizes[0]} (incl. augmentation), val {sizes[1]}, "
                      f"test {sizes[2]}; {data['n_subjects']} subjects", flush=True)
                run_check('timing projection', check_timing, models, args.dataset_root,
                          args.results_root, device, data['n_subjects'], sizes, args.n_slots,
                          args.walltime_hours, args.timing_epochs)
    else:
        RESULTS.append(('FAIL', 'model checks', 'skipped because no device passed'))

    failed = [r for r in RESULTS if r[0] == 'FAIL']
    REPORT['checks'] = [dict(status=s, name=n, detail=d) for s, n, d in RESULTS]
    try:
        out_dir = os.path.join(root, 'preflight')
        os.makedirs(out_dir, exist_ok=True)
        tag = os.environ.get('RUN_TAG') or time.strftime('%Y%m%d_%H%M%S')
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
