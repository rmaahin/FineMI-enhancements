"""
FINE fine-grained MI pipeline, 0.5-3 Hz band, with 5x DECIMATION and a switchable
feature-extraction front-end.

This is the morlet-scalogram/fine_mi.py pipeline re-targeted at the 0.5-3 Hz FineMI
data (subject{N}_eeg_epochs_0.5_3hz_250hz.npz) with ONE methodological change:

    Each trial is decimated by a factor D = 5, retaining every fifth sample,

        X[n, c, t] = Xbar[n, c, s] |_{s = D t},     t = 0, ..., T-1,

    so the effective sampling rate is 50 Hz and T = 200 time points per 4 s window.
    After the 0.5-3 Hz filter the signal only holds slow activity, and 50 Hz
    (Nyquist 25 Hz) is far above the filtered band, so no anti-alias filter is needed
    beyond the band-pass already applied. Decimation also stretches the receptive
    field of the (sample-count-fixed) network fivefold: the FINE 1D network spans 42
    samples = 168 ms at 250 Hz but 840 ms at 50 Hz, i.e. most of one 1 Hz cycle.

FEATURE_MODE:
    'raw'  - decimated 0.5-3 Hz EEG,   (C, T),    FINE 1D  (the paper-modified baseline)
    'cwt'  - Morlet scalograms at 50 Hz, (C, F, T),  FINE 2D  (Technique 2)

Both arms share identical fold indices, seeds, normalisation and augmentation draws;
they diverge only inside `apply_frontend`. Both arms see the SAME decimated trials, so
the {7, 15, 31}-sample temporal kernels cover the same 140 / 300 / 620 ms in either arm
and the front-end is the only difference.

What changed vs morlet-scalogram/fine_mi.py (everything else is verbatim):
  * DECIM = 5 applied in load_pair() straight after the class filter (Eq. 1 above);
    all window / cue / context arithmetic is in DECIMATED samples at FS = 50 Hz.
  * Wavelet bank moved from 8-30 Hz to 0.5-3 Hz; the CWT runs on the decimated
    50 Hz signal (equivalent to a 250 Hz CWT sub-sampled by 5, since a <=3 Hz Morlet
    has no energy anywhere near 25 Hz) and no extra time decimation is applied.
  * reflect padding is applied iteratively: a 0.5 Hz wavelet is longer than the
    200-sample window and torch's 'reflect' mode requires pad < length.
  * Dataset discovery looks for FineMI_0.5_3hz/ instead of FineMI/.
  * FINE_BEST_CKPT=1 (optional, default 0) makes best-val checkpointing actually work
    via deepcopy; the default preserves the original no-op restore (final-epoch weights).
  * FINE_DECIM=1 (optional, default 5) turns decimation off: the control arm at 250 Hz,
    T = 1000, same 0.5-3 Hz wavelet bank rebuilt for 250 Hz. Its results default to
    results_decim1/ so they never mix with the decimated run in results/.

IMPORTANT: import this module BEFORE anything else imports torch - the
determinism bootstrap must set CUBLAS_WORKSPACE_CONFIG before CUDA initialises.
"""

# ===========================================================================
# [P1] DETERMINISM BOOTSTRAP - must run before torch/CUDA
# ===========================================================================
import os

SEED = 42
os.environ.setdefault('PYTHONHASHSEED', str(SEED))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import copy
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

warnings.filterwarnings('ignore')

# ===========================================================================
# PATHS - resolved relative to this file so cwd does not matter
# ===========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
BAND_TAG = '0.5_3hz'
DATASET_DIRNAME = 'FineMI_0.5_3hz'
FILE_GLOB = f"subject*_eeg_epochs_{BAND_TAG}_*.npz"

#   FINE_DATASET_ROOT=$SCRATCH/finemi-dataset/FineMI_0.5_3hz   FINE_RESULTS_ROOT=/data/out
if os.environ.get('FINE_DATASET_ROOT'):
    DATASET_ROOT = os.path.abspath(os.path.expanduser(os.environ['FINE_DATASET_ROOT']))
else:
    # search upwards for a FineMI_0.5_3hz/ directory holding the subject files
    DATASET_ROOT = os.path.join(HERE, DATASET_DIRNAME)
    for up in ('', '..', '../..', '../../..'):
        cand = os.path.normpath(os.path.join(HERE, up, DATASET_DIRNAME))
        if glob.glob(os.path.join(cand, FILE_GLOB)):
            DATASET_ROOT = cand
            break

_DECIM_FOR_ROOT = int(os.environ.get('FINE_DECIM', '5'))
if os.environ.get('FINE_RESULTS_ROOT'):
    RESULTS_ROOT = os.path.abspath(os.path.expanduser(os.environ['FINE_RESULTS_ROOT']))
elif _DECIM_FOR_ROOT == 5:
    RESULTS_ROOT = os.path.join(HERE, 'results')
else:
    RESULTS_ROOT = os.path.join(HERE, f'results_decim{_DECIM_FOR_ROOT}')

# ===========================================================================
# DECIMATION + EPOCHING
# Epochs run -0.5 -> 4.0 s at 250 Hz (1126 samples); the cue is at sample 125.
# Every fifth sample is retained (Eq. 1), so at 50 Hz the epoch is 226 samples and the
# cue sits at decimated sample 25. The original slices from sample 0 (the window spans
# -500 -> +3500 ms rather than 0 -> 4000 ms); that is preserved, as in the parent.
#   FINE_CUE_SAMPLE=125 (in ORIGINAL 250 Hz samples) re-runs cue-aligned, diagnostic only.
# ===========================================================================
ORIG_SAMPLING_RATE = 250
# D in Eq. (1). FINE_DECIM=1 disables decimation entirely and gives the "vanilla"
# 0.5-3 Hz pipeline at 250 Hz (T = 1000, wavelet bank rebuilt for 250 Hz) - the control
# arm for isolating the effect of decimation. Results of the two settings never mix:
# is_done() checks the stamp and the default results root is suffixed for D != 5.
DECIM = int(os.environ.get('FINE_DECIM', '5'))
assert DECIM >= 1 and ORIG_SAMPLING_RATE % DECIM == 0, f"bad FINE_DECIM={DECIM}"
SAMPLING_RATE = ORIG_SAMPLING_RATE // DECIM                 # 50 Hz effective (D=5)

CUE_SAMPLE_ORIG = int(os.environ.get('FINE_CUE_SAMPLE', '0'))
assert CUE_SAMPLE_ORIG % DECIM == 0, "FINE_CUE_SAMPLE must be a multiple of DECIM"
CUE_SAMPLE = CUE_SAMPLE_ORIG // DECIM                       # decimated units
WINDOW_MS = 4000
WINDOW_SAMPLES = int(WINDOW_MS * SAMPLING_RATE / 1000)      # 200  (= T in the paper)
TEST_TIME_WINDOWS_MS_FINAL = [WINDOW_MS]                    # list keeps the save format

# ===========================================================================
# MORLET CWT FRONT-END PARAMETERS (0.5-3 Hz bank, evaluated at 50 Hz)
# ===========================================================================
N_FREQS = 24
FREQS = np.logspace(np.log10(0.5), np.log10(3.0), N_FREQS)
N_CYCLES = np.linspace(3.0, 7.0, N_FREQS)   # ramp; FREQS/2 would give a Gabor transform
CWT_TRUNC = 5.0                             # kernel support in sigma_t
LOG_EPS = 1e-10                             # absolute floor; see morlet_scalogram contract

# CWT batch size. Peak VRAM is roughly chunk * C * F * n_fft * 12 bytes; at 50 Hz the
# FFT length is 2048 (chunk=64 is well under 1 GB), at 250 Hz (FINE_DECIM=1) it is 8192,
# so ~2.3 GB at chunk=16. Lower FINE_CWT_CHUNK / --cwt-chunk if VRAM is tight.
def _default_cwt_chunk():
    env = os.environ.get('FINE_CWT_CHUNK')
    if env:
        return int(env)
    try:
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return 64 if gb >= 16 else (32 if gb >= 11 else 16)
    except Exception:
        pass
    return 16


CWT_CHUNK = _default_cwt_chunk()

# Temporal kernel sizes for the 2D MTC block. Both arms now run at 50 Hz, so the
# default {7,15,31} covers 140 / 300 / 620 ms in BOTH arms - receptive fields are matched
# by construction and the env override is only kept for ablations. Kernels must be odd
# so that padding=k//2 preserves the time length.
MTC_KERNELS_2D = tuple(int(k) for k in
                       os.environ.get('FINE_MTC_KERNELS_2D', '7,15,31').split(','))
assert len(MTC_KERNELS_2D) == 3 and all(k % 2 == 1 for k in MTC_KERNELS_2D), \
    f"need three odd kernels, got {MTC_KERNELS_2D}"

# Best-validation checkpoint. 0 (default) = the ORIGINAL no-op restore, i.e. the final
# epoch's weights are tested (see run_pair). 1 = deepcopy so best-val selection works.
BEST_CKPT = int(os.environ.get('FINE_BEST_CKPT', '0'))

# Bump whenever a change alters the numbers a run produces. Stamped into results.json;
# is_done() treats a mismatched or missing stamp as "not done".
#   1-3 - morlet-scalogram lineage (8-30 Hz, 250 Hz)
#   4   - CWT-modified: 0.5-3 Hz data, DECIM=5 (50 Hz), 0.5-3 Hz wavelet bank
PIPELINE_VERSION = 4

# ===========================================================================
# TRAINING
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
        lines.append(f"CWT_CHUNK={CWT_CHUNK}")
    else:
        lines.append("WARNING: no CUDA device. Training on CPU is roughly 20-40x slower.")
    lines.append(f"band={BAND_TAG} | decim={DECIM} ({ORIG_SAMPLING_RATE} -> {SAMPLING_RATE} Hz)"
                 f" | window={WINDOW_SAMPLES} samples | cue={CUE_SAMPLE} | "
                 f"best_ckpt={BEST_CKPT} | pipeline_version={PIPELINE_VERSION}")
    lines.append(f"NOTE: determinism holds per-GPU-model. Run BOTH arms on this same "
                 f"GPU model ({gpu}); results from a different card are not "
                 f"bit-comparable.")
    return "\n".join(lines)


def seed_worker(worker_id):     # [P4]
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ===========================================================================
# MORLET CWT FRONT-END (fixed, non-learned, zero trainable parameters)
# ===========================================================================
def make_morlet_kernels(freqs, n_cycles, fs, trunc=CWT_TRUNC):
    """Complex Morlet kernels, zero-mean and L1-normalised.

    L1 (not L2) normalisation gives a flat amplitude response across frequency; with L2
    the response decays as sqrt(n_cycles/f), which biases the peak row downwards.
    Returns (psi_real, psi_imag), each (F, K) float64 with K odd.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    n_cycles = np.broadcast_to(np.asarray(n_cycles, dtype=np.float64), freqs.shape)
    sigma_t = n_cycles / (2.0 * np.pi * freqs)
    half = int(np.ceil(trunc * sigma_t.max() * fs))
    t = np.arange(-half, half + 1) / fs
    psi = np.zeros((freqs.size, t.size), dtype=np.complex128)
    for i, (f, s) in enumerate(zip(freqs, sigma_t)):
        w = np.exp(-(t ** 2) / (2.0 * s ** 2)) * np.exp(2j * np.pi * f * t)
        w = w - w.mean()                    # admissibility: zero mean
        w = w / np.sum(np.abs(w))           # L1 normalise
        psi[i] = w
    return psi.real.copy(), psi.imag.copy()


# Kernels are built for the DECIMATED rate: the CWT runs on the 50 Hz signal.
PSI_R_NP, PSI_I_NP = make_morlet_kernels(FREQS, N_CYCLES, SAMPLING_RATE)
CWT_K = PSI_R_NP.shape[1]                   # 479 samples = 9.6 s at 50 Hz
CWT_PAD = CWT_K // 2                        # 239
SIGMA_T = N_CYCLES / (2 * np.pi * FREQS)
# per-row half-support in samples (the top row is the only one that is local on a 4 s window)
ROW_HALF_SUPPORT = np.ceil(CWT_TRUNC * SIGMA_T * SAMPLING_RATE).astype(int)

# Context window fed to the CWT (DECIMATED samples). The left side uses REAL pre-cue
# data where available; the right side is reflect-padded, so no post-window sample can
# ever influence the output.
CTX_START = max(0, CUE_SAMPLE - CWT_PAD)
CTX_END = CUE_SAMPLE + WINDOW_SAMPLES
CTX_LEN = CTX_END - CTX_START
LEFT_CTX = CUE_SAMPLE - CTX_START           # real pre-cue samples we actually have
LEFT_REFLECT = CWT_PAD - LEFT_CTX           # shortfall made up by reflection
WIN_SLICE = slice(LEFT_CTX, LEFT_CTX + WINDOW_SAMPLES)
# offset of the decision window inside the full 'valid' convolution output
WIN_OFFSET = CWT_PAD + LEFT_REFLECT + WIN_SLICE.start
T_OUT = WINDOW_SAMPLES                      # no further time decimation: T = 200

assert WIN_SLICE.stop == CTX_LEN, "window must end exactly at the context end (causality)"
assert LEFT_REFLECT >= 0

_KERNEL_CACHE = {}


def _kernels_on(device):
    if device not in _KERNEL_CACHE:
        _KERNEL_CACHE[device] = (torch.from_numpy(PSI_R_NP).float().to(device),
                                 torch.from_numpy(PSI_I_NP).float().to(device))
    return _KERNEL_CACHE[device]


def reflect_pad(x, left, right):
    """Reflect-pad the last axis by (left, right), iterating when a pad exceeds the
    signal length (torch's 'reflect' requires pad < length). The 0.5 Hz wavelet is
    239 samples each side of a 200-sample window, so this case is the norm here."""
    while left > 0 or right > 0:
        n = x.shape[-1] - 1
        l, r = min(left, n), min(right, n)
        x = torch.nn.functional.pad(x, (l, r), mode='reflect')
        left, right = left - l, right - r
    return x


def morlet_scalogram(x_np, device, chunk=None):
    """(N, C, CTX_LEN) -> (N, C, F, T_OUT) float32.

    magnitude -> crop pad -> log. Runs under no_grad; the front-end is a fixed
    transform, nothing here is trainable or seeded.

    CONTRACT: input must already be per-channel z-scored. LOG_EPS is an ABSOLUTE floor,
    so on volt-scale data (sigma ~ 3e-6) it would dominate the log and destroy the
    output. On z-scored input the magnitudes are orders of magnitude above LOG_EPS.
    """
    assert x_np.ndim == 3 and x_np.shape[2] == CTX_LEN, \
        f"expected (N, C, {CTX_LEN}), got {x_np.shape}"
    chunk = CWT_CHUNK if chunk is None else chunk
    psi_r, psi_i = _kernels_on(device)

    padded_len = CTX_LEN + LEFT_REFLECT + CWT_PAD
    n_fft = 1
    while n_fft < padded_len + CWT_K - 1:
        n_fft *= 2

    out = np.empty((x_np.shape[0], x_np.shape[1], N_FREQS, T_OUT), dtype=np.float32)
    with torch.no_grad():
        kr = torch.fft.rfft(psi_r, n=n_fft)
        ki = torch.fft.rfft(psi_i, n=n_fft)
        for s in range(0, x_np.shape[0], chunk):
            xb = torch.from_numpy(np.ascontiguousarray(x_np[s:s + chunk])).float().to(device)
            # Right edge is always reflected (never reads post-window data). The left is
            # reflected only where real pre-cue context is missing.
            xb = reflect_pad(xb, LEFT_REFLECT, CWT_PAD)
            xf = torch.fft.rfft(xb, n=n_fft).unsqueeze(-2)          # (b, C, 1, n/2+1)
            lo, hi = WIN_OFFSET, WIN_OFFSET + WINDOW_SAMPLES
            re = torch.fft.irfft(xf * kr, n=n_fft)[..., lo:hi]
            im = torch.fft.irfft(xf * ki, n=n_fft)[..., lo:hi]
            mag = torch.sqrt(re * re + im * im)
            mag = torch.log(mag + LOG_EPS)
            out[s:s + chunk] = mag.cpu().numpy()
            del xb, xf, re, im, mag
    return out


def scalogram_normalize(s_train, s_val, s_test):
    """Per-(channel, frequency) z-score, statistics from the TRAINING fold only.

    log() plus this centring makes the front-end invariant to any global amplitude
    scaling of the input, which log1p() would not be.
    """
    mu = s_train.mean(axis=(0, 3), keepdims=True)
    sd = s_train.std(axis=(0, 3), keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (s_train - mu) / sd, (s_val - mu) / sd, (s_test - mu) / sd


def apply_frontend(mode, x_train, x_val, x_test, device):
    """Context tensors (N, C, CTX_LEN) -> model-ready tensors."""
    if mode == 'raw':
        return (x_train[:, :, WIN_SLICE].astype(np.float32),
                x_val[:, :, WIN_SLICE].astype(np.float32),
                x_test[:, :, WIN_SLICE].astype(np.float32))
    return scalogram_normalize(morlet_scalogram(x_train, device),
                               morlet_scalogram(x_val, device),
                               morlet_scalogram(x_test, device))


# ===========================================================================
# PREPROCESSING HELPERS
# ===========================================================================
def z_score_normalize_ctx(x_train, x_val, x_test, win_slice):
    """Per-channel z-score. Stats come from the TRAINING fold and, within it, only from
    the decision window - so the CWT context padding cannot shift the statistics."""
    stat_src = x_train[:, :, win_slice]
    means = np.mean(stat_src, axis=(0, 2))
    stds = np.std(stat_src, axis=(0, 2))
    stds = np.where(stds == 0, 1.0, stds)
    means = means[np.newaxis, :, np.newaxis]
    stds = stds[np.newaxis, :, np.newaxis]
    return ((x_train - means) / stds, (x_val - means) / stds, (x_test - means) / stds)


def add_gaussian_noise_augmentation(x, noise_level=0.1, random_seed=None):
    """Add Gaussian noise to EEG data for augmentation.  DA = DR + NL*GN"""
    if random_seed is not None:
        np.random.seed(int(random_seed))            # [P5] int cast
    channel_stds = np.std(x, axis=(0, 2), keepdims=True)
    return x + np.random.normal(loc=0.0, scale=noise_level * channel_stds, size=x.shape)


class EEGDataset(Dataset):
    """Dataset for EEG data. Works for both (C, T) and (C, F, T) samples."""

    def __init__(self, eeg_data, labels):
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'eeg': self.eeg_data[idx], 'label': self.labels[idx]}


# ===========================================================================
# MODELS (unchanged from the parent; only the time length they see differs)
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

    Receptive field: MTC 31 -> spatial 33 -> maxpool(2) 34 -> DS-conv(5, stride 2) 42
    samples = 168 ms at 250 Hz, 840 ms at 50 Hz (the paper's numbers).
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


# --------------------------------------------------------------------------
# 2D lift. NOTE: AdaptiveAvgPool2d / AdaptiveMaxPool2d have nondeterministic CUDA
# backwards and RAISE under use_deterministic_algorithms(True). Use mean/amax.
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
    """Kernels are (1, kt): purely temporal, frequency untouched until the spatial conv."""

    def __init__(self, in_channels, out_channels_per_branch=32, kernels=None):
        super().__init__()
        k1, k2, k3 = MTC_KERNELS_2D if kernels is None else kernels

        def branch(k):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=(1, k),
                          padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(out_channels_per_branch), nn.ReLU())

        self.branch1, self.branch2, self.branch3 = branch(k1), branch(k2), branch(k3)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class FINEScalogram2D(nn.Module):
    """FINE over (F, T) scalograms. Input (B, C, F, T); C are conv in-channels."""

    def __init__(self, n_channels, n_freqs, n_timepoints, n_classes=2, embedding_dim=128):
        super().__init__()
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
        x = x.mean(dim=(-2, -1))              # global pool, deterministic
        return self.classifier(self.cnn_projection(x))


def build_model(mode, n_channels):
    if mode == 'raw':
        return CNNEarlyClassificationModel(n_channels=n_channels,
                                           n_timepoints=WINDOW_SAMPLES,
                                           n_classes=N_CLASSES, embedding_dim=128)
    return FINEScalogram2D(n_channels=n_channels, n_freqs=N_FREQS, n_timepoints=T_OUT,
                           n_classes=N_CLASSES, embedding_dim=128)


# ===========================================================================
# TRAIN / VALIDATE
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
    files = sorted(glob.glob(os.path.join(root, FILE_GLOB)), key=get_subject_number)
    if not files:
        raise FileNotFoundError(f"no {FILE_GLOB} under {root}")
    return files


def decimate(x):
    """Eq. (1): keep every DECIM-th sample of the last axis, starting at s = 0."""
    return x[..., ::DECIM]


def load_pair(class_a, class_b, dataset_root=None, max_subjects=None, verbose=True):
    """-> dict {subject_id: {'X': (n, C, CTX_LEN), 'y': (n,)}}, n_channels

    Per file: class filter -> DECIMATE (250 -> 50 Hz) -> crop the context window.
    Decimation runs on the full epoch from sample 0, exactly as Eq. (1) states, and the
    context is then cut in decimated units.
    """
    files = list_subject_files(dataset_root)
    if verbose:
        print(f"Found {len(files)} subject files ({BAND_TAG})")
        print(f"\nBINARY CLASSIFICATION: Class {class_a} vs Class {class_b}  "
              f"({pair_name(class_a, class_b)})")

    per_subject, n_channels = {}, None
    for path in files:
        sid = get_subject_number(path)
        if max_subjects is not None and len(per_subject) >= max_subjects:
            break
        with np.load(path, allow_pickle=True) as z:
            lab = z["labels"]
            m = (lab == class_a) | (lab == class_b)
            if not m.any():
                continue
            dat = decimate(z["data"][m])            # (n, C, 226) at 50 Hz
            if CTX_END > dat.shape[2]:
                raise ValueError(f"context end {CTX_END} exceeds decimated epoch length "
                                 f"{dat.shape[2]} in {path}")
            per_subject[sid] = {'X': dat[:, :, CTX_START:CTX_END],
                                'y': np.where(lab[m] == class_b, 1, 0)}
            n_channels = per_subject[sid]['X'].shape[1]
            del dat

    if verbose:
        n0 = sum(int((d['y'] == 0).sum()) for d in per_subject.values())
        n1 = sum(int((d['y'] == 1).sum()) for d in per_subject.values())
        print(f"  Class 0 (original {class_a}): {n0} trials")
        print(f"  Class 1 (original {class_b}): {n1} trials")
        print(f"\nContext tensor per subject: (n, {n_channels}, {CTX_LEN}) @ {SAMPLING_RATE} Hz "
              f"(decimated samples {CTX_START}:{CTX_END})")
        print(f"  decision window = context[{WIN_SLICE.start}:{WIN_SLICE.stop}] "
              f"= {WINDOW_SAMPLES} samples = {WINDOW_MS} ms")
    return per_subject, n_channels


# ===========================================================================
# ONE PAIR, ONE MODE
# ===========================================================================
def run_pair(class_a, class_b, mode, device, n_epochs=N_EPOCHS, max_subjects=None,
             dataset_root=None, verbose=True, progress=None):
    """Within-subject 5-fold CV for one task pair. Returns all_subject_results."""
    subjects, n_channels = load_pair(class_a, class_b, dataset_root=dataset_root,
                                     max_subjects=max_subjects, verbose=verbose)
    unique_subjects = sorted(subjects)
    if verbose:
        print(f"\n  Mode: {mode} | Subjects: {len(unique_subjects)} | "
              f"Model input: "
              + (f"({n_channels}, {WINDOW_SAMPLES})" if mode == 'raw'
                 else f"({n_channels}, {N_FREQS}, {T_OUT})")
              + f" | Batch {BATCH_SIZE} | Epochs {n_epochs} | Folds {N_FOLDS}")

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

        for fold_idx, (train_val_indices, test_indices) in enumerate(cv_splits):
            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=0.25, random_state=int(42 + subj_id),
                stratify=y_subj[train_val_indices])

            x_train, y_train = x_subj[train_indices], y_subj[train_indices]
            x_val, y_val = x_subj[val_indices], y_subj[val_indices]
            x_test, y_test = x_subj[test_indices], y_subj[test_indices]

            # --- identical in both arms: z-score then augment, on the CONTEXT tensor ---
            x_train_norm, x_val_norm, x_test_norm = z_score_normalize_ctx(
                x_train, x_val, x_test, WIN_SLICE)

            aug_seed = int(SEED + int(subj_id) + int(fold_idx) + int(tw_idx) * 100)
            x_train_aug = add_gaussian_noise_augmentation(
                x_train_norm, noise_level=NOISE_LEVEL, random_seed=aug_seed)
            x_train_final = np.concatenate([x_train_norm, x_train_aug], axis=0)
            y_train_final = np.concatenate([y_train, y_train], axis=0)

            # --- the ONLY place the two arms differ ---
            x_tr_f, x_va_f, x_te_f = apply_frontend(
                mode, x_train_final, x_val_norm, x_test_norm, device)

            loader_seed = int(SEED + int(subj_id) * 1000 + int(fold_idx) * 100 + int(tw_idx))
            train_generator = torch.Generator()
            train_generator.manual_seed(loader_seed)

            train_loader = DataLoader(EEGDataset(x_tr_f, y_train_final),
                                      batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                      generator=train_generator, worker_init_fn=seed_worker)
            val_loader = DataLoader(EEGDataset(x_va_f, y_val), batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=0, worker_init_fn=seed_worker)
            test_loader = DataLoader(EEGDataset(x_te_f, y_test), batch_size=BATCH_SIZE,
                                     shuffle=False, num_workers=0, worker_init_fn=seed_worker)

            model_init_seed = loader_seed
            torch.manual_seed(model_init_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(model_init_seed)

            model = build_model(mode, n_channels).to(device)
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
                    if BEST_CKPT:
                        best_model_state = copy.deepcopy(model.state_dict())
                    else:
                        best_model_state = model.state_dict().copy()

            # DEFAULT (FINE_BEST_CKPT=0) PRESERVES THE ORIGINAL BEHAVIOUR:
            # state_dict() hands back references to the live parameters, so .copy()
            # duplicates the dict but not the tensors; optimizer.step() then mutates them
            # in place and this restore is a no-op - the FINAL epoch's weights are what
            # get tested. That is the behaviour behind the paper's Table III.
            # FINE_BEST_CKPT=1 deep-copies instead so best-val selection really works
            # (the Conformer_0.5_3hz project runs that way).
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            test_loss, test_acc, test_preds, test_labels = validate(
                model, test_loader, criterion, device)

            fold_results_tw.append({
                'fold': fold_idx + 1, 'test_accuracy': test_acc, 'test_loss': test_loss,
                'test_preds': np.array(test_preds), 'test_labels': np.array(test_labels),
                # trial ids let every downstream metric be recomputed without re-running
                'test_indices': np.asarray(test_indices),
                'n_test_samples': len(test_labels), 'val_accuracy': best_val_acc})

            del model, best_model_state, x_tr_f, x_va_f, x_te_f
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        preds = np.concatenate([r['test_preds'] for r in fold_results_tw])
        labels = np.concatenate([r['test_labels'] for r in fold_results_tw])
        twr = {t_ms: {'test_accuracy': accuracy_score(labels, preds) * 100,
                      'test_preds': preds, 'test_labels': labels,
                      'confusion_matrix': confusion_matrix(labels, preds),
                      'n_test_samples': len(labels), 'fold_results': fold_results_tw}}

        all_subject_results.append({'subject_id': subj_id, 'n_folds': N_FOLDS,
                                    'time_window_results': twr})
        msg = (f"S{subj_id:>2}: {t_ms}ms={twr[t_ms]['test_accuracy']:6.2f}%"
               f"   [{time.time() - t_start:6.1f}s]")
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
                idx = f.get('test_indices')
                if idx is None:
                    idx = np.full(len(f['test_labels']), -1)
                for t, yt, yp in zip(idx, f['test_labels'], f['test_preds']):
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
                   pipeline_version=PIPELINE_VERSION, band=BAND_TAG,
                   decim=DECIM, sampling_rate=SAMPLING_RATE,
                   window_samples=WINDOW_SAMPLES, best_ckpt=BEST_CKPT,
                   windows=TEST_TIME_WINDOWS_MS_FINAL, seed=SEED, gpu=gpu,
                   torch=torch.__version__, cue_sample=CUE_SAMPLE_ORIG,
                   frontend=dict(n_freqs=N_FREQS, fmin=float(FREQS[0]),
                                 fmax=float(FREQS[-1]),
                                 n_cycles=[float(c) for c in N_CYCLES],
                                 cwt_fs=SAMPLING_RATE, kernel_len=int(CWT_K),
                                 pad=int(CWT_PAD), log_eps=LOG_EPS,
                                 mtc_kernels_2d=list(MTC_KERNELS_2D)),
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
    """True only if this pair was produced by the CURRENT pipeline settings.

    A results directory written by older code, another decimation, another cue
    alignment or another checkpoint rule reports False, so a sweep recomputes it
    rather than silently mixing pipelines.
    """
    root = RESULTS_ROOT if results_root is None else results_root
    pair_dir = os.path.join(root, mode, pair_slug(class_a, class_b))
    if not os.path.isfile(os.path.join(pair_dir, "wide_subject_x_window.csv")):
        return False
    meta_path = os.path.join(pair_dir, "results.json")
    if not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False
    stale = []
    for key, want in (('pipeline_version', PIPELINE_VERSION), ('band', BAND_TAG),
                      ('decim', DECIM), ('cue_sample', CUE_SAMPLE_ORIG),
                      ('best_ckpt', BEST_CKPT)):
        if meta.get(key) != want:
            stale.append(f"{key} {meta.get(key)} != {want}")
    if not os.path.isfile(os.path.join(pair_dir, "predictions.csv")):
        stale.append("no predictions.csv")
    if mode == 'cwt':
        got = tuple(meta.get('frontend', {}).get('mtc_kernels_2d', (7, 15, 31)))
        if got != MTC_KERNELS_2D:
            stale.append(f"mtc_kernels_2d {got} != {MTC_KERNELS_2D}")
    if stale and verbose:
        print(f"  stale {mode}/{pair_slug(class_a, class_b)}: {'; '.join(stale)}")
    return not stale
