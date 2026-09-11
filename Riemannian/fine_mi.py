"""
FINE fine-grained MI pipeline with a RIEMANNIAN (SPD covariance) feature front-end.

Technique 3 of "Proposed Feature Extraction Techniques": per-trial spatial covariance
matrices, handled with the affine-invariant Riemannian geometry of the SPD manifold
(Barachant et al. 2012; Congedo, Barachant & Bhatia 2017), fed to the FINE architecture
from EMBC_deterministic-3.ipynb.

Single source of truth for `run_local.py` (batch sweep) and `analyze_results.py`.

FEATURE_MODE (arms)
-------------------
  neural arms - train the FINE network, identical recipe to the original
    'raw'      8-30 Hz EEG (C, T) -> FINE 1D                      verbatim baseline
    'ts'       tangent-space vector (P(P+1)/2)  -> FINE MLP head    proposal fit (a)
    'tsimg'    tangent-space matrix as a (C, C) "spatial image"
                                                -> FINE 1D trunk    proposal fit (b)
    'fbts'     filter-bank block tangent vectors (K * P(P+1)/2)
                                                -> FINE MLP head    block version of (a)
    'fbtsimg'  filter-bank block tangent images (C, K, C)
                                                -> FINE 2D trunk    block version of (b),
                                                                    Tensor-CSPNet-style
  classical arms - no network, same folds; references for "use the geometry directly"
    'mdrm'     Minimum Distance to Riemannian Mean  (Barachant 2012, Congedo 2017)
    'tslda'    tangent space + shrinkage LDA        (Barachant 2012 TSLDA, regularised)

All arms share identical fold indices, seeds, channel z-scoring and augmentation draws;
the neural arms diverge only inside `apply_frontend` / `build_model`.

THE GEOMETRY (what the front-end does, per fold)
------------------------------------------------
 1. Rank handling. The FineMI epochs are common-average referenced, so every trial's
    62x62 sample covariance is SINGULAR (rank 61; verified: smallest eigenvalue ~1e-17
    of the largest). log / inverse-sqrt are undefined there. We therefore estimate the
    data subspace U (C x P, orthonormal) from the CLEAN training fold and do all geometry
    on U^T x (P = 61 for all 62 channels). Because the noise augmentation is added per
    channel it breaks the CAR constraint, which is exactly why U must come from the
    clean trials: the augmented copies are projected onto the same subspace, discarding
    the physically meaningless noise component along the null direction.
 2. Covariance. Sample covariance of the P-dim projected trial, Ledoit-Wolf shrinkage
    towards (tr/P) I (Congedo 2017: regularise when the ratio samples/channels is not
    large - it is not, for 62 channels and narrow bands, where a 5 Hz band over 4 s has
    only ~40 degrees of freedom per channel).
 3. Reference point. The Riemannian (Karcher / geometric) mean of the clean training
    covariances, fixed-point algorithm of Barachant 2012 / Congedo 2017, initialised at
    the log-Euclidean mean. Test/val trials are mapped with the TRAINING reference - no
    leakage. Because each subject is its own CV problem this is also per-subject
    re-centering (Congedo 2017, section 3.3; Zanini 2018).
 4. Tangent space. S_i = logm(M^-1/2 C_i M^-1/2). Vector form: upper triangle with the
    off-diagonal entries scaled by sqrt(2), so ||vec(S)||_2 = ||S||_F = the AIRM distance
    to M (Barachant 2012). Image form: U S U^T, i.e. the tangent matrix mapped back to
    electrode coordinates (C x C, symmetric). This is independent of the choice of basis
    inside the subspace, and preserves the Frobenius norm.
 5. Standardise every feature (vector entry / image pixel) with clean-training-fold
    statistics, exactly like the original per-channel z-score.

Blocks (fb* arms). A fixed linear-phase FIR filter bank (the FBCNet arm's filter design,
verbatim) splits the trial into B sub-bands, optionally into S equal temporal segments
(Tensor-CSPNet's tensor stacking), giving K = B*S SPD matrices per trial; each block
gets its own Riemannian reference mean.

DESIGN CONTRACT
---------------
The training loop, CV structure, seeding, normalisation, augmentation, optimiser,
scheduler, checkpoint behaviour and save format are the ORIGINAL pipeline, verbatim.
'raw' reproduces the FBCNet/Morlet raw arms bit-for-bit (same code path, same folds), so
their saved raw results can be reused and all arms are paired per subject and fold.

IMPORTANT: import this module BEFORE anything else imports torch - the determinism
bootstrap must set CUBLAS_WORKSPACE_CONFIG before CUDA initialises.
"""

# ===========================================================================
# [P1] DETERMINISM BOOTSTRAP - must run before torch/CUDA
# ===========================================================================
import os

SEED = 42
os.environ.setdefault('PYTHONHASHSEED', str(SEED))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import glob
import json
import pickle
import random
import re
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits          # ships with scikit-learn

warnings.filterwarnings('ignore')

# ===========================================================================
# PATHS - resolved relative to this file so cwd does not matter
# ===========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from google.colab import drive  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def _default_dataset_root():
    env = os.environ.get('FINE_DATASET_ROOT')
    if env:
        return env
    if IN_COLAB:
        return '/content/drive/MyDrive/multi_joint_mi_dataset/extracted/FineMI/FineMI'
    # FineMI/ either next to this repo's technique folders or next to the repo itself
    for cand in (os.path.join(HERE, '..', 'FineMI'), os.path.join(HERE, '..', '..', 'FineMI')):
        if glob.glob(os.path.join(cand, 'subject*_eeg_epochs_*.npz')):
            return os.path.normpath(cand)
    return os.path.normpath(os.path.join(HERE, '..', 'FineMI'))


DATASET_ROOT = _default_dataset_root()
RESULTS_ROOT = os.environ.get('FINE_RESULTS_ROOT', os.path.join(HERE, 'results'))

# ===========================================================================
# EPOCHING - identical to the FBCNet / Morlet arms
# ===========================================================================
SAMPLING_RATE = 250
# ORIGINAL slicing preserved: the notebook takes X[:, :, :t] from the epoch start, i.e.
# the "4000 ms" window spans -500 -> +3500 ms around the cue. FINE_CUE_SAMPLE=125 re-runs
# cue-aligned, as a diagnostic only.
CUE_SAMPLE = int(os.environ.get('FINE_CUE_SAMPLE', '0'))
WINDOW_MS = 4000
WINDOW_SAMPLES = int(WINDOW_MS * SAMPLING_RATE / 1000)      # 1000
TEST_TIME_WINDOWS_MS_FINAL = [WINDOW_MS]                    # list keeps the save format

# ===========================================================================
# ARMS
# ===========================================================================
# feat: 'vec' | 'img' | None ; fb: uses the filter-bank blocks ; clf: what classifies
MODE_SPEC = {
    'raw':     dict(fb=False, feat=None,  clf='fine1d'),
    'ts':      dict(fb=False, feat='vec', clf='mlp'),
    'tsimg':   dict(fb=False, feat='img', clf='fine'),
    'fbts':    dict(fb=True,  feat='vec', clf='mlp'),
    'fbtsimg': dict(fb=True,  feat='img', clf='fine'),
    'mdrm':    dict(fb=False, feat=None,  clf='mdrm'),
    'tslda':   dict(fb=False, feat='vec', clf='lda'),
}
ALL_MODES = tuple(MODE_SPEC)
RIEM_MODES = tuple(m for m in ALL_MODES if m != 'raw')
CLASSICAL_MODES = tuple(m for m in ALL_MODES if MODE_SPEC[m]['clf'] in ('mdrm', 'lda'))
NEURAL_MODES = tuple(m for m in ALL_MODES if m not in CLASSICAL_MODES)

# ===========================================================================
# RIEMANNIAN FRONT-END PARAMETERS  (env overrides are diagnostics; defaults = the sweep)
# ===========================================================================
def _parse_bands(spec):
    bands = []
    for tok in spec.split(','):
        lo, hi = tok.strip().split('-')
        bands.append((float(lo), float(hi)))
    return tuple(bands)


# Sub-bands for the block (fb*) arms: mu, low-beta, high-beta. Deliberately wider than
# Tensor-CSPNet's 4 Hz bank: with 62 channels a 4 Hz band over 4 s has ~32 degrees of
# freedom per channel, far below P = 61, so its covariance would be almost all shrinkage.
#   FINE_RIEM_BANDS="8-12,12-16,16-20,20-24,24-28" gives the Tensor-CSPNet-style bank.
BANDS = _parse_bands(os.environ.get('FINE_RIEM_BANDS', '8-13,13-20,20-30'))
N_BANDS = len(BANDS)
# Equal, non-overlapping temporal segments per band for the fb* arms (Tensor-CSPNet's
# temporal segmentation). 1 = whole window. Tensor-CSPNet's own ablation found no
# significant average gain from temporal segmentation in CV, hence the default.
N_SEGMENTS = int(os.environ.get('FINE_RIEM_SEGMENTS', '1'))
assert N_SEGMENTS >= 1 and WINDOW_SAMPLES // N_SEGMENTS >= 100, \
    f"FINE_RIEM_SEGMENTS={N_SEGMENTS} leaves segments shorter than 100 samples"
N_BLOCKS_FB = N_BANDS * N_SEGMENTS

# FIR transition width in Hz (same design as the FBCNet arm): 2 Hz -> 413 taps.
FB_TRANS_HZ = float(os.environ.get('FINE_FB_TRANS', '2.0'))

# Covariance shrinkage: 'lw' = Ledoit-Wolf (per trial, analytic), or a fixed float in
# [0, 1) (e.g. '0' = plain sample covariance - only valid on the full-rank subspace).
RIEM_SHRINK = os.environ.get('FINE_RIEM_SHRINK', 'lw')
if RIEM_SHRINK != 'lw':
    assert 0.0 <= float(RIEM_SHRINK) < 1.0, f"FINE_RIEM_SHRINK must be lw or [0,1)"

# Channels for the Riemannian arms ('raw' always uses all 62).
#   'all'   - all 62 channels (P = 61 after removing the CAR null direction)
#   'motor' - the 20 sensorimotor channels Tensor-CSPNet used on the (also 62-channel)
#             KU dataset: FC5/3/1/2/4/6, C5/3/1/z/2/4/6, CP5/3/1/z/2/4/6. Congedo 2017
#             (section 4): with N >= 32 electrodes the Riemannian distance is dominated
#             by task-irrelevant components, so a smaller N can help.
#   or a comma list of channel names from channel_location_64_neuroscan.locs
RIEM_CHANNELS = os.environ.get('FINE_RIEM_CHANNELS', 'all')
MOTOR_CHANNELS = ('FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6',
                  'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                  'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6')

# Eigenvalues below RANK_TOL * largest of the pooled clean-training covariance are
# treated as the null space. The CAR null eigenvalue sits ~1e-17 below the top one and
# the smallest genuine one ~1e-3, so anything in between separates them.
RANK_TOL = float(os.environ.get('FINE_RIEM_RANK_TOL', '1e-8'))

# Karcher-mean fixed point (Barachant 2012 / Congedo 2017 algorithm).
MEAN_TOL = 1e-9
MEAN_MAXITER = 100

# Trials per filtering/covariance chunk (CPU memory: chunk * C * B * n_fft * 16 bytes).
RIEM_CHUNK = int(os.environ.get('FINE_RIEM_CHUNK', '32'))

# Bump whenever a change alters the numbers a Riemannian arm produces. is_done() treats
# a mismatch as "not done" so a behaviour change cannot leave stale results in a sweep.
#   1 - initial Riemannian front-end
PIPELINE_VERSION = 1
# The raw arm's code path is the FBCNet/Morlet one, unchanged; its stamp stays 1 so their
# saved raw results are recognised as current and can be reused.
RAW_PIPELINE_VERSION = 1

# ===========================================================================
# TRAINING  (verbatim from the original notebook)
# ===========================================================================
NOISE_LEVEL = 0.15
BATCH_SIZE = 32
N_EPOCHS = 50
N_CLASSES = 2
N_FOLDS = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

JOINT_NAMES = {0: "HOC", 1: "WFE", 2: "WAA", 3: "EPS",
               4: "EFE", 5: "SPS", 6: "SAA", 7: "SFE"}
ALL_PAIRS = [(a, b) for a in range(8) for b in range(a + 1, 8)]   # 28 pairs


def pair_name(class_a, class_b):
    return f"{JOINT_NAMES[class_a]}/{JOINT_NAMES[class_b]}"


def pair_slug(class_a, class_b):
    return pair_name(class_a, class_b).replace("/", "_")


# ===========================================================================
# DEVICE + DETERMINISM
# ===========================================================================
def setup_determinism(seed=SEED, strict=True):
    """Seed everything and enable deterministic kernels. Returns (device, gpu_name)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)   # [P2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    return device, gpu


def describe_environment(device, gpu):
    lines = [f"device={device} | gpu={gpu} | torch={torch.__version__}"]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        lines.append(f"VRAM={props.total_memory / 1e9:.1f} GB | "
                     f"sm_{props.major}{props.minor} | cuda={torch.version.cuda}")
    else:
        lines.append("WARNING: no CUDA device. The neural arms train ~20-40x slower on CPU.")
    lines.append("NOTE: determinism holds per-GPU-model. Results are not bit-comparable "
                 "across different GPUs.")
    lines.append(riem_config_str())
    return "\n".join(lines)


def riem_config_str():
    return (f"riemann: channels={RIEM_CHANNELS} shrink={RIEM_SHRINK} "
            f"bands(fb*)={','.join(f'{lo:g}-{hi:g}' for lo, hi in BANDS)} "
            f"segments(fb*)={N_SEGMENTS} fir={FB_NUMTAPS} taps rank_tol={RANK_TOL:g}")


def seed_worker(worker_id):     # [P4]
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ===========================================================================
# SPD MANIFOLD PRIMITIVES  (float64 NumPy, batched over leading axes; CPU so the
# geometry is bit-reproducible and independent of the GPU model)
# ===========================================================================
def _sym(a):
    return 0.5 * (a + np.swapaxes(a, -1, -2))


def _eigfun(c, fn):
    """V diag(fn(w)) V^T for symmetric c (..., P, P)."""
    w, v = np.linalg.eigh(_sym(c))
    return _sym((v * fn(w)[..., None, :]) @ np.swapaxes(v, -1, -2))


def _pos(w):
    # eigenvalues of a shrunk covariance are strictly positive; this only guards
    # round-off on a matrix that is SPD in exact arithmetic
    return np.maximum(w, np.finfo(np.float64).tiny)


def sqrtm(c):
    return _eigfun(c, lambda w: np.sqrt(_pos(w)))


def invsqrtm(c):
    return _eigfun(c, lambda w: 1.0 / np.sqrt(_pos(w)))


def logm(c):
    return _eigfun(c, lambda w: np.log(_pos(w)))


def expm(s):
    return _eigfun(s, np.exp)


def congruence(a, c):
    """a c a^T with a broadcast over c's leading axes (a symmetric here)."""
    return _sym(a @ c @ np.swapaxes(a, -1, -2))


def distance_riemann(c, m):
    """Affine-invariant Riemannian distance delta_R(c_i, m) for a batch c (n, P, P):
    ||logm(m^-1/2 c m^-1/2)||_F = sqrt(sum log^2 lambda_i(m^-1 c))   (Barachant 2012)."""
    w = np.linalg.eigvalsh(congruence(invsqrtm(m), c))
    return np.sqrt(np.sum(np.log(_pos(w)) ** 2, axis=-1))


def mean_riemann(c, tol=MEAN_TOL, maxiter=MEAN_MAXITER):
    """Karcher (geometric) mean of SPD matrices c (n, P, P).

    Fixed-point / gradient-descent iteration of Barachant 2012 and Congedo 2017:
        M <- M^1/2 expm( mean_i logm(M^-1/2 C_i M^-1/2) ) M^1/2
    until ||mean_i logm(...)||_F < tol. Initialised at the log-Euclidean mean, which is
    already close (it is exact when the C_i commute), so few iterations are needed.
    """
    m = expm(logm(c).mean(axis=0))
    for _ in range(maxiter):
        m_sqrt, m_isqrt = sqrtm(m), invsqrtm(m)
        j = logm(congruence(m_isqrt, c)).mean(axis=0)
        m = congruence(m_sqrt, expm(j))
        if np.linalg.norm(j) < tol:
            break
    return m


def log_map(c, m):
    """Tangent vectors at m, as symmetric matrices: logm(m^-1/2 c m^-1/2)."""
    return logm(congruence(invsqrtm(m), c))


def upper_vec(s):
    """(..., P, P) symmetric -> (..., P(P+1)/2). Off-diagonals scaled by sqrt(2) so the
    Euclidean norm of the vector equals the Frobenius norm of the matrix, i.e. the
    tangent-space vector's norm is the Riemannian distance to the reference point."""
    p = s.shape[-1]
    iu = np.triu_indices(p)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return s[..., iu[0], iu[1]] * coef


def ledoit_wolf_shrinkage(y, emp):
    """Ledoit-Wolf shrinkage intensity per trial. y (n, P, T) zero-mean, emp = y y^T / T.
    Vectorised transcription of sklearn.covariance.ledoit_wolf_shrinkage
    (assume_centered=True); checked against it in `run_local.py --selftest`."""
    n_feat, n_samp = y.shape[-2], y.shape[-1]
    x2 = y ** 2
    emp_trace = x2.sum(axis=-1) / n_samp                         # (n, P)
    mu = emp_trace.sum(axis=-1) / n_feat                         # (n,)
    beta_ = (x2.sum(axis=-2) ** 2).sum(axis=-1)                  # sum(X2^T X2)
    delta_ = (emp ** 2).sum(axis=(-2, -1))                       # sum((X^T X)^2) / T^2
    beta = (beta_ / n_samp - delta_) / (n_feat * n_samp)
    delta = (delta_ - 2.0 * mu * emp_trace.sum(axis=-1) + n_feat * mu ** 2) / n_feat
    beta = np.minimum(beta, delta)
    return np.where(beta == 0, 0.0, beta / np.where(delta == 0, 1.0, delta)), mu


def covariances(x, basis):
    """(n, C, T) -> (n, P, P) shrunk covariance of basis^T x (P = basis.shape[1])."""
    y = np.einsum('cp,nct->npt', basis, x)
    y = y - y.mean(axis=-1, keepdims=True)
    emp = _sym(y @ np.swapaxes(y, -1, -2) / y.shape[-1])
    p = emp.shape[-1]
    if RIEM_SHRINK == 'lw':
        alpha, mu = ledoit_wolf_shrinkage(y, emp)
    else:
        alpha = np.full(emp.shape[0], float(RIEM_SHRINK))
        mu = np.trace(emp, axis1=-2, axis2=-1) / p
    eye = np.eye(p)
    return (1.0 - alpha)[:, None, None] * emp + (alpha * mu)[:, None, None] * eye


def subspace_basis(x):
    """Orthonormal basis (C, P) of the signal subspace of clean trials x (n, C, T).

    The pooled covariance of CAR data has one (numerically) zero eigenvalue along the
    reference's null direction; dropping it makes every trial covariance full rank.
    Any further rank loss (e.g. interpolated channels) is handled the same way.
    """
    y = x - x.mean(axis=-1, keepdims=True)
    pooled = _sym(np.einsum('nct,ndt->cd', y, y) / (y.shape[0] * y.shape[-1]))
    w, v = np.linalg.eigh(pooled)
    keep = w > RANK_TOL * w[-1]
    return v[:, keep]


# ===========================================================================
# FILTER BANK (fb* arms). FIR design verbatim from the FBCNet arm.
# ===========================================================================
def _fir_numtaps(trans_hz, fs):
    n = int(np.ceil(3.3 * fs / float(trans_hz)))
    return n + 1 if n % 2 == 0 else n


def make_bandpass_fir(f_lo, f_hi, fs, numtaps):
    """Linear-phase bandpass FIR: difference of Hamming-windowed sincs, unit gain at the
    band centre. Identical to ../FBCNet/fine_mi.py."""
    assert numtaps % 2 == 1, "numtaps must be odd"
    assert 0 < f_lo < f_hi < fs / 2, f"invalid band {f_lo}-{f_hi} Hz at fs={fs}"
    n = np.arange(numtaps, dtype=np.float64) - (numtaps - 1) / 2.0

    def sinc_lowpass(fc):
        return (2.0 * fc / fs) * np.sinc(2.0 * fc * n / fs)

    h = sinc_lowpass(f_hi) - sinc_lowpass(f_lo)
    h *= np.hamming(numtaps)
    f_c = 0.5 * (f_lo + f_hi)
    gain = np.abs(np.sum(h * np.exp(-2j * np.pi * f_c * n / fs)))
    return h / gain


FB_NUMTAPS = _fir_numtaps(FB_TRANS_HZ, SAMPLING_RATE)
FB_PAD = FB_NUMTAPS // 2
FB_H_NP = np.stack([make_bandpass_fir(lo, hi, SAMPLING_RATE, FB_NUMTAPS)
                    for lo, hi in BANDS])                            # (B, numtaps)
assert FB_PAD < WINDOW_SAMPLES, "FIR longer than the window; raise FINE_FB_TRANS"
_FB_NFFT = 1 << int(np.ceil(np.log2(WINDOW_SAMPLES + 2 * FB_PAD + FB_NUMTAPS - 1)))
_FB_HF = np.fft.rfft(FB_H_NP, n=_FB_NFFT)                            # (B, nfft/2+1)


def filter_bank(x):
    """(n, C, T) float64 -> (n, C, B, T). FFT convolution, reflect padding on both edges
    (the window is the whole slice, so no sample outside it is ever read), group delay
    removed. CPU NumPy: deterministic and GPU-independent."""
    t = x.shape[-1]
    xp = np.pad(x, ((0, 0), (0, 0), (FB_PAD, FB_PAD)), mode='reflect')
    xf = np.fft.rfft(xp, n=_FB_NFFT)[:, :, None, :]
    y = np.fft.irfft(xf * _FB_HF, n=_FB_NFFT)
    lo = 2 * FB_PAD                        # FB_PAD of padding + FB_PAD of group delay
    return y[..., lo:lo + t]


# ===========================================================================
# CHANNELS
# ===========================================================================
def channel_names(dataset_root=None):
    root = DATASET_ROOT if dataset_root is None else dataset_root
    path = os.path.join(root, 'channel_location_64_neuroscan.locs')
    names = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                names.append(parts[3].upper())
    return names


def riem_channel_index(n_channels, dataset_root=None):
    """Indices of the channels the Riemannian arms use, or None for all."""
    if RIEM_CHANNELS.lower() == 'all':
        return None
    names = channel_names(dataset_root)[:n_channels]   # .locs lists HEO/VEO last
    want = MOTOR_CHANNELS if RIEM_CHANNELS.lower() == 'motor' else \
        tuple(c.strip().upper() for c in RIEM_CHANNELS.split(','))
    missing = [c for c in want if c not in names]
    if missing:
        raise ValueError(f"channels not in the montage: {missing}")
    return np.array([names.index(c) for c in want])


# ===========================================================================
# RIEMANNIAN FRONT-END (fit on the clean training fold, then transform anything)
# ===========================================================================
def _single_thread_blas(fn):
    """Run with one BLAS thread. The geometry is thousands of batched 61x61 eigh calls,
    where multithreaded BLAS is ~7x SLOWER (thread start-up dominates), and a fixed
    thread count also makes the float64 results independent of the machine's cores."""
    def wrapped(*args, **kwargs):
        with threadpool_limits(limits=1, user_api='blas'):
            return fn(*args, **kwargs)
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    return wrapped


class RiemannFrontend:
    """Per-fold Riemannian feature extractor.

    fit(x_clean, y) learns, from CLEAN (non-augmented) z-scored training trials only:
      the signal subspace U, one Riemannian reference mean per block, feature
      standardisation statistics and, for MDRM, per-class Riemannian means.
    transform(x) returns model-ready float32 features.
    """

    def __init__(self, mode, ch_idx=None):
        spec = MODE_SPEC[mode]
        self.mode, self.feat, self.fb = mode, spec['feat'], spec['fb']
        self.ch_idx = ch_idx
        self.n_bands = N_BANDS if self.fb else 1
        self.n_seg = N_SEGMENTS if self.fb else 1
        self.n_blocks = self.n_bands * self.n_seg

    # -- covariances -------------------------------------------------------
    def _select(self, x):
        x = x if self.ch_idx is None else x[:, self.ch_idx, :]
        return x.astype(np.float64, copy=False)

    def block_covs(self, x):
        """(n, C, T) -> (n, K, P, P)."""
        x = self._select(x)
        n, t = x.shape[0], x.shape[-1]
        seg = t // self.n_seg
        p = self.basis.shape[1]
        out = np.empty((n, self.n_blocks, p, p))
        for s in range(0, n, RIEM_CHUNK):
            xb = x[s:s + RIEM_CHUNK]
            views = filter_bank(xb) if self.fb else xb[:, :, None, :]
            k = 0
            for b in range(self.n_bands):
                for si in range(self.n_seg):
                    out[s:s + RIEM_CHUNK, k] = covariances(
                        views[:, :, b, si * seg:(si + 1) * seg], self.basis)
                    k += 1
        return out

    # -- features ----------------------------------------------------------
    def _raw_features(self, covs):
        s = np.stack([log_map(covs[:, k], self.refs[k]) for k in range(self.n_blocks)],
                     axis=1)                                          # (n, K, P, P)
        if self.feat == 'vec':
            return upper_vec(s).reshape(s.shape[0], -1)               # (n, K*P(P+1)/2)
        img = congruence(self.basis, s)                               # U S U^T (n,K,C,C)
        if self.n_blocks == 1:
            return img[:, 0]                                          # (n, C, C)
        return np.transpose(img, (0, 2, 1, 3))                        # (n, C, K, C)

    @_single_thread_blas
    def fit(self, x_clean, y=None):
        self.basis = subspace_basis(self._select(x_clean))
        covs = self.block_covs(x_clean)
        self.refs = [mean_riemann(covs[:, k]) for k in range(self.n_blocks)]
        if self.mode == 'mdrm':
            self.classes = np.unique(y)
            self.class_means = {c: [mean_riemann(covs[y == c, k])
                                    for k in range(self.n_blocks)] for c in self.classes}
        if self.feat is not None:
            f = self._raw_features(covs)
            self.mu = f.mean(axis=0)
            sd = f.std(axis=0)
            self.sd = np.where(sd == 0, 1.0, sd)
        return self

    @_single_thread_blas
    def transform(self, x):
        f = self._raw_features(self.block_covs(x))
        return ((f - self.mu) / self.sd).astype(np.float32)

    # -- MDRM --------------------------------------------------------------
    @_single_thread_blas
    def mdrm_predict(self, x):
        """argmin_c sum_k delta_R^2(C_k, M_{c,k}); with several blocks the squared
        distances add, as Congedo 2017 (Appendix I) prescribes for filter banks."""
        covs = self.block_covs(x)
        d2 = np.stack([sum(distance_riemann(covs[:, k], self.class_means[c][k]) ** 2
                           for k in range(self.n_blocks)) for c in self.classes], axis=1)
        return self.classes[np.argmin(d2, axis=1)]

    @property
    def rank(self):
        return int(self.basis.shape[1])


def apply_frontend(mode, x_train_final, x_val, x_test, x_train_clean, ch_idx=None):
    """Z-scored trials -> model-ready tensors. The front-end is FIT on x_train_clean."""
    if mode == 'raw':
        return (x_train_final.astype(np.float32), x_val.astype(np.float32),
                x_test.astype(np.float32)), None
    fe = RiemannFrontend(mode, ch_idx).fit(x_train_clean)
    return (fe.transform(x_train_final), fe.transform(x_val), fe.transform(x_test)), fe


# ===========================================================================
# PREPROCESSING HELPERS  (verbatim)
# ===========================================================================
def z_score_normalize(x_train, x_val, x_test):
    """Per-channel z-score using training-fold statistics only."""
    means = np.mean(x_train, axis=(0, 2))
    stds = np.std(x_train, axis=(0, 2))
    stds = np.where(stds == 0, 1.0, stds)
    means = means[np.newaxis, :, np.newaxis]
    stds = stds[np.newaxis, :, np.newaxis]
    return (x_train - means) / stds, (x_val - means) / stds, (x_test - means) / stds


def add_gaussian_noise_augmentation(x, noise_level=0.1, random_seed=None):
    """Add Gaussian noise to EEG data for augmentation.  DA = DR + NL*GN"""
    if random_seed is not None:
        np.random.seed(int(random_seed))            # [P5] int cast
    channel_stds = np.std(x, axis=(0, 2), keepdims=True)
    return x + np.random.normal(loc=0.0, scale=noise_level * channel_stds, size=x.shape)


class EEGDataset(Dataset):
    """Dataset for any feature tensor: (C, T), (D,), (C, C) or (C, K, C) per sample."""

    def __init__(self, eeg_data, labels):
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'eeg': self.eeg_data[idx], 'label': self.labels[idx]}


# ===========================================================================
# MODELS
# ===========================================================================
def init_weights(m):
    """Initialize model weights deterministically."""
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)


class EfficientChannelAttention(nn.Module):
    """ECA block. [P3] amax instead of AdaptiveMaxPool1d (nondeterministic backward)."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(x.amax(dim=-1, keepdim=True))
        return self.sigmoid(avg_out + max_out) * x


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride,
                                   padding=padding, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        return self.relu(self.bn2(self.pointwise(x)))


class FeatureFusionModule(nn.Module):
    def __init__(self, channels, kernel_size=(1, 5), groups=None):
        super().__init__()
        groups = channels if groups is None else groups
        k = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size
        self.ds_conv = DepthwiseSeparableConv1d(channels, channels, k,
                                                groups=groups, padding=k // 2)
        self.eca = EfficientChannelAttention(channels)

    def forward(self, x):
        return self.eca(self.ds_conv(x))


class MultiscaleTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels_per_branch=32):
        super().__init__()

        def branch(k):
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(out_channels_per_branch), nn.ReLU())

        self.branch1, self.branch2, self.branch3 = branch(7), branch(15), branch(31)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class SpatialFeatureExtraction(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU())
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class CNNEarlyClassificationModel(nn.Module):
    """CNN model for early EEG classification. Unchanged from the original notebook.

    'raw' feeds it (C, T). 'tsimg' feeds it the (C, C) tangent-space image: rows are the
    Conv1d in-channels exactly as electrodes are for raw EEG, and the MTC kernels
    (7/15/31) slide along the partner-electrode axis instead of time.
    """

    def __init__(self, n_channels, n_timepoints, n_classes=2, embedding_dim=128):
        super().__init__()
        self.multiscale_temporal = MultiscaleTemporalBlock(n_channels, 32)
        self.spatial_extraction = SpatialFeatureExtraction(32 * 3, 64)
        self.feature_fusion = FeatureFusionModule(64, kernel_size=(1, 5), groups=64)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.cnn_projection = nn.Sequential(
            nn.Linear(64, embedding_dim), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes))

    def forward(self, x_eeg):
        x = self.multiscale_temporal(x_eeg)
        x = self.spatial_extraction(x)
        x = self.feature_fusion(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(self.cnn_projection(x))


class TangentSpaceMLP(nn.Module):
    """Proposal fit (a): tangent-space vector -> the FINE head.

    The CNN trunk is replaced by the Riemannian features; what follows is FINE's own
    projection (Linear -> ReLU -> Dropout 0.3 to the 128-d embedding) and its original
    two-layer classifier, unchanged.
    """

    def __init__(self, in_dim, n_classes=2, embedding_dim=128):
        super().__init__()
        self.cnn_projection = nn.Sequential(
            nn.Linear(in_dim, embedding_dim), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.classifier(self.cnn_projection(x))


# --------------------------------------------------------------------------
# 2D lift for block images (C, K, C). As in the FBCNet arm: AdaptiveAvg/MaxPool2d have
# nondeterministic CUDA backwards and RAISE under the determinism flags - use mean/amax.
# --------------------------------------------------------------------------
class EfficientChannelAttention2d(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(x.mean(dim=(-2, -1), keepdim=True))
        max_out = self.fc(x.amax(dim=(-2, -1), keepdim=True))
        return self.sigmoid(avg_out + max_out) * x


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        return self.relu(self.bn2(self.pointwise(x)))


class MultiscaleTemporalBlock2d(nn.Module):
    """Kernels (1, k): each block (band x segment) is convolved on its own; blocks first
    mix in the 3x3 spatial conv."""

    def __init__(self, in_channels, out_channels_per_branch=32, kernels=(7, 15, 31)):
        super().__init__()

        def branch(k):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=(1, k),
                          padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(out_channels_per_branch), nn.ReLU())

        self.branch1, self.branch2, self.branch3 = (branch(k) for k in kernels)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class FINEBlock2D(nn.Module):
    """FINE over block tangent images. Input (N, C, K, C); rows are conv in-channels.

    Node-for-node the FBCNet arm's FINEFilterBank2D (the original 1D graph lifted to
    2D): MTC (1,k) per block -> 3x3 spatial conv (blocks mix here) -> 2x2 maxpool ->
    depthwise-separable fusion -> ECA -> global mean pool -> the original head.
    """

    def __init__(self, n_channels, n_blocks, n_classes=2, embedding_dim=128):
        super().__init__()
        assert n_blocks >= 2, "use CNNEarlyClassificationModel for a single block"
        self.multiscale_temporal = MultiscaleTemporalBlock2d(n_channels, 32)
        self.spatial_extraction = nn.Sequential(
            nn.Conv2d(32 * 3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ds_conv = DepthwiseSeparableConv2d(64, 64, (3, 5), groups=64, padding=(1, 2))
        self.eca = EfficientChannelAttention2d(64)
        self.cnn_projection = nn.Sequential(
            nn.Linear(64, embedding_dim), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.multiscale_temporal(x)
        x = self.maxpool(self.spatial_extraction(x))
        x = self.eca(self.ds_conv(x))
        x = x.mean(dim=(-2, -1))
        return self.classifier(self.cnn_projection(x))


def build_model(mode, feat_shape):
    """feat_shape = per-sample feature shape (without the batch axis)."""
    spec = MODE_SPEC[mode]
    if spec['clf'] == 'fine1d' or (spec['clf'] == 'fine' and len(feat_shape) == 2):
        return CNNEarlyClassificationModel(n_channels=feat_shape[0],
                                           n_timepoints=feat_shape[-1],
                                           n_classes=N_CLASSES, embedding_dim=128)
    if spec['clf'] == 'fine':
        return FINEBlock2D(n_channels=feat_shape[0], n_blocks=feat_shape[1],
                           n_classes=N_CLASSES, embedding_dim=128)
    if spec['clf'] == 'mlp':
        return TangentSpaceMLP(in_dim=int(np.prod(feat_shape)), n_classes=N_CLASSES)
    raise ValueError(f"mode {mode} has no neural model")


# ===========================================================================
# TRAIN / VALIDATE  (verbatim from the original notebook)
# ===========================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in train_loader:
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(eeg)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)
            outputs = model(eeg)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / len(val_loader), 100.0 * correct / total, all_preds, all_labels


# ===========================================================================
# DATA LOADING
# ===========================================================================
def get_subject_number(path):
    match = re.search(r"subject(\d+)", os.path.basename(path), re.IGNORECASE)
    return int(match.group(1)) if match else 0


def list_subject_files(dataset_root=None):
    root = DATASET_ROOT if dataset_root is None else dataset_root
    files = sorted(glob.glob(os.path.join(root, "subject*_eeg_epochs_*.npz")),
                   key=get_subject_number)
    if not files:
        raise FileNotFoundError(f"no subject*_eeg_epochs_*.npz under {root}")
    return files


def load_pair(class_a, class_b, dataset_root=None, max_subjects=None, verbose=True):
    """-> dict {subject_id: {'X': (n, C, WINDOW_SAMPLES), 'y': (n,)}}, n_channels

    Filters to the two classes and crops the decision window per file. Same slicing as
    the original notebook (and the FBCNet/Morlet raw arms)."""
    files = list_subject_files(dataset_root)
    if verbose:
        print(f"Found {len(files)} subject files")
        print(f"\nBINARY CLASSIFICATION: Class {class_a} vs Class {class_b}  "
              f"({pair_name(class_a, class_b)})")

    per_subject, n_channels = {}, None
    lo, hi = CUE_SAMPLE, CUE_SAMPLE + WINDOW_SAMPLES
    for path in files:
        sid = get_subject_number(path)
        if max_subjects is not None and len(per_subject) >= max_subjects:
            break
        with np.load(path, allow_pickle=True) as z:
            lab = z["labels"]
            m = (lab == class_a) | (lab == class_b)
            if not m.any():
                continue
            dat = z["data"]
            if hi > dat.shape[2]:
                raise ValueError(f"window end {hi} exceeds epoch length "
                                 f"{dat.shape[2]} in {path}")
            per_subject[sid] = {'X': dat[m][:, :, lo:hi],
                                'y': np.where(lab[m] == class_b, 1, 0)}
            n_channels = per_subject[sid]['X'].shape[1]
            del dat

    if verbose:
        n0 = sum(int((d['y'] == 0).sum()) for d in per_subject.values())
        n1 = sum(int((d['y'] == 1).sum()) for d in per_subject.values())
        print(f"  Class 0 (original {class_a}): {n0} trials")
        print(f"  Class 1 (original {class_b}): {n1} trials")
        print(f"\nWindow per subject: (n, {n_channels}, {WINDOW_SAMPLES})  samples {lo}:{hi}")
    return per_subject, n_channels


# ===========================================================================
# ONE PAIR, ONE MODE
# ===========================================================================
def _fit_classical(mode, x_tr, y_tr, x_va, x_te, ch_idx):
    """-> (val_preds, test_preds). Fit on the clean training fold only."""
    fe = RiemannFrontend(mode, ch_idx).fit(x_tr, y_tr)
    if mode == 'mdrm':
        return fe.mdrm_predict(x_va), fe.mdrm_predict(x_te), fe
    # tslda: Barachant 2012's TSLDA with the ANOVA feature-selection step replaced by
    # Ledoit-Wolf shrinkage inside the LDA, which handles P(P+1)/2 >> n_trials without a
    # tuning parameter (the regularised-LDA form Congedo 2017 section 6 cites).
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(fe.transform(x_tr), y_tr)
    return clf.predict(fe.transform(x_va)), clf.predict(fe.transform(x_te)), fe


def run_pair(class_a, class_b, mode, device, n_epochs=N_EPOCHS, max_subjects=None,
             dataset_root=None, verbose=True, progress=None):
    """Within-subject 5-fold CV for one task pair. Returns all_subject_results."""
    assert mode in ALL_MODES, f"unknown mode {mode}"
    subjects, n_channels = load_pair(class_a, class_b, dataset_root=dataset_root,
                                     max_subjects=max_subjects, verbose=verbose)
    ch_idx = None if mode == 'raw' else riem_channel_index(n_channels, dataset_root)
    unique_subjects = sorted(subjects)
    if verbose:
        print(f"\n  Mode: {mode} | Subjects: {len(unique_subjects)} | "
              f"Batch {BATCH_SIZE} | Epochs {n_epochs} | Folds {N_FOLDS}")

    all_subject_results = []
    t_start = time.time()

    for subj_id in unique_subjects:
        subj_id = int(subj_id)
        x_subj = subjects[subj_id]['X']
        y_subj = subjects[subj_id]['y']

        if len(np.unique(y_subj)) < 2 or len(y_subj) < 10:
            print(f"Skipping Subject {subj_id}: insufficient data")
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=int(42 + subj_id))       # [P5]
        cv_splits = list(skf.split(np.arange(len(y_subj)), y_subj))

        tw_idx, t_ms = 0, WINDOW_MS
        fold_results_tw = []
        feat_desc = None

        for fold_idx, (train_val_indices, test_indices) in enumerate(cv_splits):
            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=0.25, random_state=int(42 + subj_id),
                stratify=y_subj[train_val_indices])

            x_train, y_train = x_subj[train_indices], y_subj[train_indices]
            x_val, y_val = x_subj[val_indices], y_subj[val_indices]
            x_test, y_test = x_subj[test_indices], y_subj[test_indices]

            # --- identical in every arm: per-channel z-score, training-fold stats ---
            x_train_norm, x_val_norm, x_test_norm = z_score_normalize(x_train, x_val, x_test)

            if mode in CLASSICAL_MODES:
                val_preds, test_preds, fe = _fit_classical(
                    mode, x_train_norm, y_train, x_val_norm, x_test_norm, ch_idx)
                feat_desc = f"rank {fe.rank}"
                best_val_acc = 100.0 * float(np.mean(val_preds == y_val))
                test_preds, test_labels = np.asarray(test_preds), np.asarray(y_test)
                test_acc = 100.0 * float(np.mean(test_preds == test_labels))
                test_loss = float('nan')
            else:
                aug_seed = int(SEED + int(subj_id) + int(fold_idx) + int(tw_idx) * 100)
                x_train_aug = add_gaussian_noise_augmentation(
                    x_train_norm, noise_level=NOISE_LEVEL, random_seed=aug_seed)
                x_train_final = np.concatenate([x_train_norm, x_train_aug], axis=0)
                y_train_final = np.concatenate([y_train, y_train], axis=0)

                # --- the ONLY place the neural arms differ ---
                (x_tr_f, x_va_f, x_te_f), fe = apply_frontend(
                    mode, x_train_final, x_val_norm, x_test_norm, x_train_norm, ch_idx)
                feat_desc = (f"feat {tuple(x_tr_f.shape[1:])}"
                             + (f" rank {fe.rank}" if fe is not None else ""))

                loader_seed = int(SEED + int(subj_id) * 1000 + int(fold_idx) * 100
                                  + int(tw_idx))
                train_generator = torch.Generator()
                train_generator.manual_seed(loader_seed)

                train_loader = DataLoader(EEGDataset(x_tr_f, y_train_final),
                                          batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=0, generator=train_generator,
                                          worker_init_fn=seed_worker)
                val_loader = DataLoader(EEGDataset(x_va_f, y_val), batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=0,
                                        worker_init_fn=seed_worker)
                test_loader = DataLoader(EEGDataset(x_te_f, y_test), batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0,
                                         worker_init_fn=seed_worker)

                model_init_seed = loader_seed
                torch.manual_seed(model_init_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(model_init_seed)

                model = build_model(mode, x_tr_f.shape[1:]).to(device)
                model.apply(init_weights)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                       weight_decay=WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                 factor=0.5, patience=5)

                best_val_acc, best_model_state = 0.0, None
                for _epoch in range(n_epochs):
                    train_epoch(model, train_loader, criterion, optimizer, device)
                    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
                    scheduler.step(val_loss)
                    if val_acc > best_val_acc:                 # original selection rule
                        best_val_acc = val_acc
                        best_model_state = model.state_dict().copy()

                # PRESERVED FROM THE ORIGINAL, DO NOT "FIX" (same as the FBCNet/Morlet
                # arms): state_dict().copy() is shallow, so this restore is a no-op and the
                # FINAL epoch's weights are tested. Kept so every arm matches the paper.
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)

                test_loss, test_acc, test_preds, test_labels = validate(
                    model, test_loader, criterion, device)
                test_preds, test_labels = np.array(test_preds), np.array(test_labels)

                del model, best_model_state, x_tr_f, x_va_f, x_te_f
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            fold_results_tw.append({
                'fold': fold_idx + 1, 'test_accuracy': test_acc, 'test_loss': test_loss,
                'test_preds': test_preds, 'test_labels': test_labels,
                'test_indices': np.asarray(test_indices),
                'n_test_samples': len(test_labels), 'val_accuracy': best_val_acc})

        preds = np.concatenate([r['test_preds'] for r in fold_results_tw])
        labels = np.concatenate([r['test_labels'] for r in fold_results_tw])
        twr = {t_ms: {'test_accuracy': accuracy_score(labels, preds) * 100,
                      'test_preds': preds, 'test_labels': labels,
                      'confusion_matrix': confusion_matrix(labels, preds),
                      'n_test_samples': len(labels), 'fold_results': fold_results_tw}}

        all_subject_results.append({'subject_id': subj_id, 'n_folds': N_FOLDS,
                                    'time_window_results': twr})
        msg = (f"S{subj_id:>2}: {t_ms}ms={twr[t_ms]['test_accuracy']:6.2f}%"
               f"   [{feat_desc}]   [{time.time() - t_start:6.1f}s]")
        if progress is not None:
            progress(msg)
        elif verbose:
            print(msg)

    if verbose:
        print(f"\nALL SUBJECTS COMPLETED - {pair_name(class_a, class_b)} [{mode}] "
              f"in {(time.time() - t_start) / 60:.1f} min")
    return all_subject_results


# ===========================================================================
# [P7] SAVE
# ===========================================================================
def frontend_meta(mode):
    if mode == 'raw':
        return dict(kind='raw')
    spec = MODE_SPEC[mode]
    meta = dict(kind='riemannian', feat=spec['feat'], clf=spec['clf'], fb=spec['fb'],
                channels=RIEM_CHANNELS, shrink=RIEM_SHRINK, rank_tol=RANK_TOL,
                metric='affine-invariant', reference='karcher-mean-of-clean-train')
    if spec['fb']:
        meta.update(bands=[[float(lo), float(hi)] for lo, hi in BANDS],
                    n_segments=int(N_SEGMENTS), trans_hz=FB_TRANS_HZ,
                    numtaps=int(FB_NUMTAPS))
    return meta


def save_results(all_subject_results, class_a, class_b, mode, gpu,
                 results_root=None, verbose=True):
    root = RESULTS_ROOT if results_root is None else results_root
    name, slug = pair_name(class_a, class_b), pair_slug(class_a, class_b)
    pair_dir = os.path.join(root, mode, slug)
    os.makedirs(pair_dir, exist_ok=True)

    subj_rows, fold_rows, pred_rows = [], [], []
    for r in sorted(all_subject_results, key=lambda x: x['subject_id']):
        sid = int(r['subject_id'])
        for w in TEST_TIME_WINDOWS_MS_FINAL:
            twr = r['time_window_results'][w]
            subj_rows.append(dict(pair=name, mode=mode, subject=sid, window_ms=w,
                                  accuracy=float(twr['test_accuracy']),
                                  n_test=int(twr['n_test_samples'])))
            for f in twr['fold_results']:
                fold_rows.append(dict(pair=name, mode=mode, subject=sid, window_ms=w,
                                      fold=int(f['fold']),
                                      accuracy=float(f['test_accuracy']),
                                      val_accuracy=float(f['val_accuracy']),
                                      n_test=int(f['n_test_samples'])))
                for t, yt, yp in zip(f['test_indices'], f['test_labels'], f['test_preds']):
                    pred_rows.append((sid, int(f['fold']), int(t), int(yt), int(yp)))

    df_subj = pd.DataFrame(subj_rows)
    df_fold = pd.DataFrame(fold_rows)
    df_pred = pd.DataFrame(pred_rows,
                           columns=['subject', 'fold', 'trial', 'y_true', 'y_pred'])
    df_pred.insert(0, 'mode', mode)
    df_pred.insert(0, 'pair', name)
    df_pred.to_csv(os.path.join(pair_dir, "predictions.csv"), index=False)
    wide = (df_subj.pivot(index='subject', columns='window_ms', values='accuracy')
            .reindex(columns=TEST_TIME_WINDOWS_MS_FINAL).sort_index())

    df_subj.to_csv(os.path.join(pair_dir, "per_subject.csv"), index=False)
    df_fold.to_csv(os.path.join(pair_dir, "per_fold.csv"), index=False)
    wide.to_csv(os.path.join(pair_dir, "wide_subject_x_window.csv"))

    payload = dict(pair=name, mode=mode, class_a=int(class_a), class_b=int(class_b),
                   pipeline_version=(RAW_PIPELINE_VERSION if mode == 'raw'
                                     else PIPELINE_VERSION),
                   windows=TEST_TIME_WINDOWS_MS_FINAL, seed=SEED, gpu=gpu,
                   torch=torch.__version__, cue_sample=CUE_SAMPLE,
                   frontend=frontend_meta(mode),
                   subjects=[int(s) for s in wide.index],
                   matrix=wide.to_numpy().tolist())
    with open(os.path.join(pair_dir, "results.pkl"), "wb") as f:
        pickle.dump(payload, f)
    with open(os.path.join(pair_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    if verbose:
        print("=" * 68)
        print(f"SAVED {name} [{mode}] -> {pair_dir}")
        print("=" * 68)
        print(wide.round(2).to_string())
        print("\nmean:", wide.mean().round(2).to_dict())
        print("sd  :", wide.std(ddof=1).round(2).to_dict())
        if wide.isna().any().any():
            print("WARNING: missing subject/window cells - a subject was skipped.")
    return pair_dir, wide


def is_done(class_a, class_b, mode, results_root=None, verbose=False):
    """True only if this pair was produced by the CURRENT pipeline and configuration."""
    root = RESULTS_ROOT if results_root is None else results_root
    pair_dir = os.path.join(root, mode, pair_slug(class_a, class_b))
    meta_path = os.path.join(pair_dir, "results.json")
    if not (os.path.isfile(os.path.join(pair_dir, "wide_subject_x_window.csv"))
            and os.path.isfile(meta_path)):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False
    stale = []
    want_v = RAW_PIPELINE_VERSION if mode == 'raw' else PIPELINE_VERSION
    if meta.get('pipeline_version') != want_v:
        stale.append(f"pipeline_version {meta.get('pipeline_version')} != {want_v}")
    if meta.get('cue_sample') != CUE_SAMPLE:
        stale.append(f"cue_sample {meta.get('cue_sample')} != {CUE_SAMPLE}")
    if not os.path.isfile(os.path.join(pair_dir, "predictions.csv")):
        stale.append("no predictions.csv")
    if mode != 'raw':
        got, want = meta.get('frontend', {}), frontend_meta(mode)
        for key in want:
            if got.get(key) != want[key]:
                stale.append(f"{key} {got.get(key)} != {want[key]}")
    if stale and verbose:
        print(f"  stale {mode}/{pair_slug(class_a, class_b)}: {'; '.join(stale)}")
    return not stale
