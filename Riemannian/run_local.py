"""
Local batch driver for the FINE Riemannian (Technique 3) comparison.

Runs any subset of the 28 task pairs in any set of arms, sequentially, and SKIPS pairs
that are already saved - so it is safe to Ctrl+C and restart at any point.

Arms (see fine_mi.py): raw ts tsimg fbts fbtsimg mdrm tslda
  --mode accepts one arm, a comma list, 'riem' (every Riemannian arm), 'neural',
  'classical' or 'all'.

Examples
--------
  python run_local.py --selftest                      # geometry + model checks, no training
  python run_local.py --mode ts --pairs HOC/SPS --smoke
  python run_local.py --mode classical --pairs all    # MDRM + TSLDA, minutes, CPU only
  python run_local.py --mode riem --pairs all         # every Riemannian arm, 28 pairs
  python run_local.py --mode raw --pairs all          # the baseline arm
  python run_local.py --mode ts,tsimg --pairs HOC/SPS --time-probe

Results are skipped only if produced by the CURRENT pipeline and configuration
(fine_mi.PIPELINE_VERSION, cue alignment, channel set, shrinkage, bands, segments ...).

Diagnostics - give each its own --results-root so nothing is mixed:
  FINE_RIEM_CHANNELS=motor python run_local.py --mode riem --results-root diag_motor
  FINE_RIEM_BANDS="8-12,12-16,16-20,20-24,24-28" FINE_RIEM_SEGMENTS=2 \\
      python run_local.py --mode fbts,fbtsimg --results-root diag_tensorcsp

Progress is appended to <results-root>/run_log.txt.
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np

import fine_mi as F


def parse_pairs(spec):
    """'all' | 'HOC/SPS,SAA/SFE' | '0-5,6-7' -> [(a, b), ...]"""
    if spec.strip().lower() == 'all':
        return list(F.ALL_PAIRS)
    name_to_id = {v: k for k, v in F.JOINT_NAMES.items()}
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        sep = '/' if '/' in tok else '-'
        a, b = tok.split(sep)
        a = name_to_id[a.strip().upper()] if a.strip().upper() in name_to_id else int(a)
        b = name_to_id[b.strip().upper()] if b.strip().upper() in name_to_id else int(b)
        if a > b:
            a, b = b, a
        if (a, b) not in F.ALL_PAIRS:
            raise ValueError(f"not a valid pair: {tok}")
        out.append((a, b))
    return out


def parse_modes(spec):
    groups = {'all': F.ALL_MODES, 'riem': F.RIEM_MODES,
              'neural': F.NEURAL_MODES, 'classical': F.CLASSICAL_MODES}
    out = []
    for tok in spec.split(','):
        tok = tok.strip().lower()
        for m in groups.get(tok, (tok,)):
            if m not in F.ALL_MODES:
                raise ValueError(f"unknown mode '{m}' (choose from {F.ALL_MODES})")
            if m not in out:
                out.append(m)
    return out


def log(msg, path):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def _rand_spd(rng, n, p, cond=50.0):
    """n random SPD matrices with eigenvalues log-uniform in [1, cond]."""
    q, _ = np.linalg.qr(rng.standard_normal((n, p, p)))
    w = np.exp(rng.uniform(0, np.log(cond), (n, p)))
    return F._sym((q * w[:, None, :]) @ np.swapaxes(q, -1, -2))


def selftest(device, args):
    """Geometry, front-end and model checks. Nothing here trains."""
    import torch
    import torch.nn as nn
    from sklearn.covariance import ledoit_wolf

    print("=" * 68)
    print("SELF-TESTS")
    print("=" * 68)
    print(F.riem_config_str())
    rng = np.random.default_rng(0)

    # 1. SPD primitives ------------------------------------------------------
    a = _rand_spd(rng, 6, 8)
    assert np.abs(F.expm(F.logm(a)) - a).max() < 1e-9 * np.abs(a).max()
    assert np.abs(F.sqrtm(a) @ F.sqrtm(a) - a).max() < 1e-9 * np.abs(a).max()
    assert np.abs(F.invsqrtm(a) @ a @ F.invsqrtm(a) - np.eye(8)).max() < 1e-9
    print("[ok] expm(logm C) = C, sqrtm^2 = C, C^-1/2 C C^-1/2 = I")

    # affine invariance: delta(W C W^T, W M W^T) = delta(C, M) for any invertible W
    # (Congedo 2017 eq. 16/17 - why sensor space == source space for Riemannian methods)
    w_mix = rng.standard_normal((8, 8)) + 3 * np.eye(8)
    d0 = F.distance_riemann(a[1:], a[0])
    d1 = F.distance_riemann(w_mix @ a[1:] @ w_mix.T, w_mix @ a[0] @ w_mix.T)
    assert np.abs(d0 - d1).max() < 1e-8 * d0.max(), (d0, d1)
    print(f"[ok] affine invariance of delta_R (max rel. dev {np.abs(d0 - d1).max() / d0.max():.1e})")

    # Karcher mean: exact for commuting matrices (geometric mean of the eigenvalues);
    # at the true mean the tangent vectors average to zero
    diag = np.stack([np.diag(rng.uniform(0.5, 5, 5)) for _ in range(7)])
    gm = F.mean_riemann(diag)
    assert np.abs(np.diag(gm) - np.exp(np.log(np.diagonal(diag, axis1=1, axis2=2)).mean(0))
                  ).max() < 1e-10
    b = _rand_spd(rng, 40, 10)
    m = F.mean_riemann(b)
    grad = np.linalg.norm(F.log_map(b, m).mean(0))
    assert grad < 1e-7, f"Karcher gradient {grad:.1e}"
    m_perm = F.mean_riemann(b[::-1])
    assert np.abs(m - m_perm).max() < 1e-8 * np.abs(m).max()
    print(f"[ok] Karcher mean: commuting case exact, gradient at mean {grad:.1e}, "
          f"order invariant")

    # tangent vectorisation is an isometry: ||vec(log_M C)|| = delta_R(C, M)
    v = F.upper_vec(F.log_map(b, m))
    assert v.shape[1] == 10 * 11 // 2
    assert np.abs(np.linalg.norm(v, axis=1) - F.distance_riemann(b, m)).max() < 1e-9
    print(f"[ok] tangent vector norm == Riemannian distance to the reference "
          f"(dim {v.shape[1]} = P(P+1)/2)")

    # Ledoit-Wolf transcription == sklearn
    y = rng.standard_normal((3, 12, 200)) * rng.uniform(0.5, 3, (1, 12, 1))
    y[:, 1] += 0.8 * y[:, 0]
    y = y - y.mean(-1, keepdims=True)
    emp = y @ np.swapaxes(y, -1, -2) / y.shape[-1]
    alpha, _ = F.ledoit_wolf_shrinkage(y, emp)
    for i in range(3):
        _, s_ref = ledoit_wolf(y[i].T, assume_centered=True)
        assert abs(alpha[i] - s_ref) < 1e-10, (alpha[i], s_ref)
    print(f"[ok] Ledoit-Wolf shrinkage matches sklearn (alpha {alpha.round(4)})")

    # 2. real data: rank, conditioning, filter bank ---------------------------
    subs, n_channels = F.load_pair(0, 5, dataset_root=args.dataset_root,
                                   max_subjects=1, verbose=False)
    sid = sorted(subs)[0]
    x = subs[sid]['X']
    y_lab = subs[sid]['y']
    xz, _, _ = F.z_score_normalize(x, x[:2], x[:2])
    ch_idx = F.riem_channel_index(n_channels, args.dataset_root)
    xs = xz if ch_idx is None else xz[:, ch_idx]
    yc = xs - xs.mean(-1, keepdims=True)
    pooled_w = np.linalg.eigvalsh(np.einsum('nct,ndt->cd', yc, yc) / yc[..., 0].size)
    basis = F.subspace_basis(xs)
    print(f"[ok] subject {sid}: {xs.shape[1]} channels -> rank {basis.shape[1]} "
          f"(pooled eigenvalues / max: smallest {pooled_w[0] / pooled_w[-1]:.1e}, "
          f"next {pooled_w[1] / pooled_w[-1]:.1e}; tol {F.RANK_TOL:g})")
    assert np.abs(basis.T @ basis - np.eye(basis.shape[1])).max() < 1e-10

    for mode in ('ts', 'fbts'):
        fe = F.RiemannFrontend(mode, ch_idx)
        fe.basis = basis
        covs = fe.block_covs(xz[:20])
        w = np.linalg.eigvalsh(covs)
        assert (w > 0).all(), "non-SPD covariance"
        conds = np.median(w[..., -1] / w[..., 0], axis=0)
        labels = (['8-30 (full)'] if mode == 'ts' else
                  [f"{lo:g}-{hi:g}/s{s}" for lo, hi in F.BANDS for s in range(F.N_SEGMENTS)])
        print(f"[ok] {mode}: {covs.shape[1]} block(s) SPD, median condition number "
              + ", ".join(f"{l}: {c:.0f}" for l, c in zip(labels, conds)))

    # filter bank: a sine lands in its own band and survives unshifted
    t = np.arange(F.WINDOW_SAMPLES) / F.SAMPLING_RATE
    for f0 in (10.0, 16.0, 25.0):
        sig = np.tile(np.sin(2 * np.pi * f0 * t), (1, 2, 1))
        out = F.filter_bank(sig)
        power = (out ** 2).mean(axis=(0, 1, 3))
        got = F.BANDS[int(np.argmax(power))]
        assert got[0] <= f0 < got[1], f"{f0} Hz peaked in {got}"
        core = slice(F.WINDOW_SAMPLES // 4, 3 * F.WINDOW_SAMPLES // 4)
        err = np.abs(out[0, 0, int(np.argmax(power)), core] - sig[0, 0, core]).max()
        assert err < 2e-2, f"{f0} Hz: group delay / gain error {err:.2e}"
    print("[ok] filter bank: 10/16/25 Hz sines land in their band, unshifted, unit gain")

    # 3. rank handling under augmentation: the noisy copies are full rank, the clean
    #    trials are not - projecting both on the clean-data subspace keeps them together
    xa = xz + np.random.default_rng(1).normal(0, 0.15, xz.shape)
    fe = F.RiemannFrontend('ts', ch_idx).fit(xz)
    d_clean = F.distance_riemann(fe.block_covs(xz[:10])[:, 0], fe.refs[0])
    d_aug = F.distance_riemann(fe.block_covs(xa[:10])[:, 0], fe.refs[0])
    print(f"[ok] augmented copies stay near the clean manifold region: mean delta_R to "
          f"reference clean {d_clean.mean():.2f} vs augmented {d_aug.mean():.2f}")
    assert d_aug.mean() < 2 * d_clean.mean()

    # 4. every arm end to end on this subject: shapes, determinism, forward/backward
    for mode in F.ALL_MODES:
        if mode in F.CLASSICAL_MODES:
            n_tr = int(0.8 * len(y_lab))
            vp, tp, fe = F._fit_classical(mode, xz[:n_tr], y_lab[:n_tr],
                                          xz[n_tr:], xz[n_tr:], ch_idx)
            vp2, _, _ = F._fit_classical(mode, xz[:n_tr], y_lab[:n_tr],
                                         xz[n_tr:], xz[n_tr:], ch_idx)
            assert (vp == vp2).all(), f"{mode} not deterministic"
            print(f"[ok] {mode:8s}: fit/predict deterministic, rank {fe.rank}, "
                  f"held-out acc on the last 20% {100 * np.mean(tp == y_lab[n_tr:]):.1f}%")
            continue
        (feat, _, _), fe = F.apply_frontend(mode, xz[:16], xz[:4], xz[:4], xz,
                                            None if mode == 'raw' else ch_idx)
        (feat2, _, _), _ = F.apply_frontend(mode, xz[:16], xz[:4], xz[:4], xz,
                                            None if mode == 'raw' else ch_idx)
        assert np.array_equal(feat, feat2), f"{mode}: front-end not bit-reproducible"
        assert np.isfinite(feat).all(), f"{mode}: non-finite features"
        if fe is not None and fe.feat == 'img':
            sym = feat if feat.ndim == 3 else np.transpose(feat, (0, 2, 1, 3))
            assert np.abs(sym - np.swapaxes(sym, -1, -2)).max() < 1e-4, "image not symmetric"
        m_ = F.build_model(mode, feat.shape[1:]).to(device)
        m_.apply(F.init_weights)
        out = m_(torch.FloatTensor(feat[:4]).to(device))
        assert out.shape == (4, F.N_CLASSES)
        nn.CrossEntropyLoss()(out, torch.zeros(4, dtype=torch.long, device=device)).backward()
        nparam = sum(p.numel() for p in m_.parameters() if p.requires_grad)
        print(f"[ok] {mode:8s}: features {tuple(feat.shape[1:])} -> "
              f"{type(m_).__name__} | {nparam:,} params | fwd+bwd under determinism")
        del m_, out

    print("\nALL SELF-TESTS PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='riem',
                    help="arm(s): raw,ts,tsimg,fbts,fbtsimg,mdrm,tslda | riem | neural | "
                         "classical | all")
    ap.add_argument('--pairs', default='all',
                    help="'all', or e.g. 'HOC/SPS,SAA/SFE' or '0-5,6-7'")
    ap.add_argument('--epochs', type=int, default=F.N_EPOCHS)
    ap.add_argument('--smoke', action='store_true',
                    help='2 subjects, 5 epochs, results are not saved')
    ap.add_argument('--max-subjects', type=int, default=None,
                    help='use only the first N subjects (give it its own --results-root)')
    ap.add_argument('--redo', action='store_true', help='recompute pairs already saved')
    ap.add_argument('--selftest', action='store_true', help='run checks and exit')
    ap.add_argument('--time-probe', action='store_true',
                    help='stop after the first pair of each arm and report projected time')
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--results-root', default=None)
    ap.add_argument('--allow-nondeterministic', action='store_true',
                    help='downgrade determinism errors to warnings (NOT for final runs)')
    args = ap.parse_args()

    results_root = args.results_root or F.RESULTS_ROOT
    log_path = os.path.join(results_root, 'run_log.txt')

    device, gpu = F.setup_determinism(strict=not args.allow_nondeterministic)
    print(F.describe_environment(device, gpu))
    print(f"dataset: {args.dataset_root or F.DATASET_ROOT}")
    print(f"results: {results_root}\n")

    if args.selftest:
        selftest(device, args)
        return 0

    modes = parse_modes(args.mode)
    pairs = parse_pairs(args.pairs)
    n_epochs = 5 if args.smoke else args.epochs
    max_subjects = 2 if args.smoke else args.max_subjects

    jobs = [(m, a, b) for m in modes for (a, b) in pairs]
    todo = [j for j in jobs
            if args.redo or args.smoke
            or not F.is_done(j[1], j[2], j[0], results_root=results_root, verbose=True)]
    if args.time_probe:                       # one pair per arm is enough to time it
        seen, probe = set(), []
        for j in todo:
            if j[0] not in seen:
                seen.add(j[0])
                probe.append(j)
        todo = probe
    skipped = len(jobs) - len(todo)

    log(f"START modes={','.join(modes)} pairs={len(pairs)} jobs={len(jobs)} "
        f"todo={len(todo)} skipped={skipped} epochs={n_epochs} smoke={args.smoke} "
        f"max_subjects={max_subjects} gpu={gpu} | {F.riem_config_str()}", log_path)

    if not todo:
        log("nothing to do - every requested pair is already saved (use --redo to force)",
            log_path)
        return 0

    durations, failures, per_mode = [], [], {}
    sweep_start = time.time()

    for i, (mode, a, b) in enumerate(todo, 1):
        name = F.pair_name(a, b)
        eta = ""
        if durations:
            eta = f" | ETA {(np.mean(durations) * (len(todo) - i + 1)) / 3600:.1f} h"
        log(f"[{i}/{len(todo)}] {mode:7s} {name}{eta}", log_path)

        t0 = time.time()
        try:
            results = F.run_pair(a, b, mode, device, n_epochs=n_epochs,
                                 max_subjects=max_subjects,
                                 dataset_root=args.dataset_root, verbose=False,
                                 progress=lambda m: print("      " + m, flush=True))
        except Exception:
            failures.append((mode, name))
            log(f"      FAILED {mode} {name}\n{traceback.format_exc()}", log_path)
            continue
        dt = time.time() - t0
        durations.append(dt)
        per_mode.setdefault(mode, []).append(dt)

        if args.smoke:
            accs = [r['time_window_results'][F.WINDOW_MS]['test_accuracy'] for r in results]
            log(f"      smoke ok: {len(results)} subjects, "
                f"mean {np.mean(accs):.2f}% in {dt / 60:.1f} min (not saved)", log_path)
        else:
            _, wide = F.save_results(results, a, b, mode, gpu,
                                     results_root=results_root, verbose=False)
            log(f"      done in {dt / 60:.1f} min | mean "
                f"{wide.mean().iloc[0]:.2f}% | sd {wide.std(ddof=1).iloc[0]:.2f}", log_path)

    if args.time_probe:
        log("\nTIME PROBE (one pair per arm):", log_path)
        total = 0.0
        for mode, ds in per_mode.items():
            per_pair = np.mean(ds) / 60
            total += per_pair * 28
            log(f"  {mode:8s} {per_pair:5.1f} min/pair -> 28 pairs {per_pair * 28 / 60:.1f} h",
                log_path)
        log(f"  all probed arms, one worker: {total / 60:.1f} h", log_path)

    total = (time.time() - sweep_start) / 3600
    log(f"SWEEP COMPLETE: {len(durations)}/{len(todo)} jobs in {total:.2f} h", log_path)
    if failures:
        log(f"FAILURES ({len(failures)}): " + ", ".join(f"{m}:{n}" for m, n in failures),
            log_path)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
