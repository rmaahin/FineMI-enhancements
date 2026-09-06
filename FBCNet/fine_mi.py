"""
FINE fine-grained MI pipeline with a FILTER-BANK feature-extraction front-end.

Technique 1 of "Proposed Feature Extraction Techniques": FBCNet-style spectral
filtering. A fixed (non-learned) FIR filter bank splits each trial into B narrow
sub-bands spanning 8-30 Hz, turning the (C, T) trial into a multi-view (C, B, T)
tensor that the Multiscale Temporal Convolution (MTC) block processes per band.

Single source of truth for `run_local.py` (batch sweep) and any notebook driver.

FEATURE_MODE:
    'raw'  - 8-30 Hz EEG as the original slices it, (C, T),     FINE 1D  (baseline)
    'fbc'  - filter-bank sub-bands,              (C, B, T),     FINE 2D  (Technique 1)

Both arms share identical fold indices, seeds, normalisation and augmentation
draws; they diverge only inside `apply_frontend`.

DESIGN CONTRACT
---------------
The original pipeline (EMBC_deterministic-3.ipynb) is reproduced VERBATIM. Nothing in
the training loop, CV structure, seeding, normalisation, augmentation, optimiser,
scheduler, checkpoint behaviour or save format was changed. FEATURE_MODE 'raw' is
bit-for-bit the original method and exists so the filter bank can be A/B'd against it
under identical folds.

Only what the filter-bank front-end genuinely requires was added:
  * make_bandpass_fir() / filter_bank()         - the front-end itself
  * FINEFilterBank2D                            - the 1D FINE graph lifted to 2D over
        (B, T). The MTC kernels are (1, k): purely temporal, so no band ever mixes with
        another until the spatial conv - which is exactly FBCNet's "let the spatial
        filters specialise per band".
  * mean/amax instead of AdaptiveAvg/MaxPool2d  - the 2D adaptive pools have
        nondeterministic CUDA backwards and RAISE under the determinism flags the
        original pipeline already sets
  * reflect-padding around the CWT-style context window - the decision window starts at
        the epoch edge, so there is no earlier data for the FIR taps to use

The filter bank adds ZERO trainable parameters (it is a fixed transform applied on the
data side, before the DataLoader), so the parameter count of the trunk is unchanged
apart from the 1D -> 2D lift.

Deliberate mirror of ../morlet-scalogram/fine_mi.py (Technique 2): same module layout,
same function names, same result files, same `is_done` staleness stamping, so the two
techniques can be aggregated side by side.

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
try:
    from google.colab import drive  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    DATASET_ROOT = '/content/drive/MyDrive/multi_joint_mi_dataset/extracted/FineMI/FineMI'
    RESULTS_ROOT = '/content/drive/MyDrive/mi_results_fbc'
else:
    DATASET_ROOT = os.path.join(HERE, '../FineMI')
    RESULTS_ROOT = os.path.join(HERE, 'mi_results_fbc')

# ===========================================================================
# EPOCHING - epochs run -0.5 -> 4.0 s at 250 Hz (1126 samples); the cue is at
# sample 125. The original slices from sample 0; that is preserved (see below).
# ===========================================================================
SAMPLING_RATE = 250
# ORIGINAL slicing, preserved deliberately: the notebook takes X[:, :, :t], i.e. the
# window starts at the epoch start. Since epochs run -0.5 -> 4.0 s, that means each
# window actually spans -500 -> +3500 ms rather than 0 -> 4000 ms. This is NOT changed
# here - reproducing the paper's pipeline takes priority.
#   FINE_CUE_SAMPLE=125 re-runs everything cue-aligned, as a diagnostic only.
CUE_SAMPLE = int(os.environ.get('FINE_CUE_SAMPLE', '0'))
WINDOW_MS = 4000
WINDOW_SAMPLES = int(WINDOW_MS * SAMPLING_RATE / 1000)      # 1000
TEST_TIME_WINDOWS_MS_FINAL = [WINDOW_MS]                    # list keeps the save format

# ===========================================================================
# FILTER BANK FRONT-END PARAMETERS
# ===========================================================================
# Six 4 Hz-wide sub-bands over 8-30 Hz, exactly as written in the proposal:
#   8-12, 12-16, 16-20, 20-24, 24-28, 28-32.
# The data is already band-limited to 8-30 Hz by preprocessing, so the 28-32 band
# effectively carries 28-30; it is kept so the bank matches the proposal verbatim.
#   FINE_FB_BANDS="8-12,12-16,..." overrides the layout (diagnostic).
def _parse_bands(spec):
    bands = []
    for tok in spec.split(','):
        lo, hi = tok.strip().split('-')
        bands.append((float(lo), float(hi)))
    return tuple(bands)


BANDS = _parse_bands(os.environ.get(
    'FINE_FB_BANDS', '8-12,12-16,16-20,20-24,24-28,28-32'))
N_BANDS = len(BANDS)

# Transition width of each windowed-sinc bandpass, in Hz. Sets the filter length:
# a Hamming-windowed sinc needs ~3.3/(trans/fs) taps. 2 Hz -> 413 taps (1.65 s), which
# is the usual trade-off for 4 Hz-wide MI sub-bands: narrower transitions separate the
# bands better but need more context around the window.
FB_TRANS_HZ = float(os.environ.get('FINE_FB_TRANS', '2.0'))

# Per-(channel, band) z-score of the filtered signals, statistics from the TRAINING
# fold only. Sub-band amplitude falls off steeply with frequency (1/f), so without this
# the 24-32 Hz views arrive an order of magnitude smaller than the mu band and the
# shared MTC kernels effectively only see the low bands.
#   FINE_FB_NORM=none  keeps the raw sub-band amplitudes (diagnostic).
FB_NORM = os.environ.get('FINE_FB_NORM', 'bandz')
assert FB_NORM in ('bandz', 'none'), f"FINE_FB_NORM must be bandz|none, got {FB_NORM}"

# Filter batch size. Peak VRAM is roughly chunk * C * B * n_fft * 8 bytes.
def _default_fb_chunk():
    env = os.environ.get('FINE_FB_CHUNK')
    if env:
        return int(env)
    try:
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return 64 if gb >= 16 else (32 if gb >= 11 else 16)
    except Exception:
        pass
    return 16


FB_CHUNK = _default_fb_chunk()

# Temporal kernel sizes for the 2D MTC block. Unlike the CWT arm, the filter bank does
# NOT decimate time, so the original 1D values cover exactly the same spans here as in
# the raw arm:
#     raw {7,15,31} @ 250 Hz  =  28 / 60 / 124 ms
#     fbc {7,15,31} @ 250 Hz  =  28 / 60 / 124 ms   <- identical receptive fields
# i.e. the only thing that changes between the two arms is the front-end. Kernels must
# be odd so that padding=k//2 preserves the time length.
MTC_KERNELS_2D = tuple(int(k) for k in
                       os.environ.get('FINE_MTC_KERNELS_2D', '7,15,31').split(','))
assert len(MTC_KERNELS_2D) == 3 and all(k % 2 == 1 for k in MTC_KERNELS_2D), \
    f"need three odd kernels, got {MTC_KERNELS_2D}"

# Bump whenever a change alters the numbers a run produces. Stamped into results.json;
# is_done() treats a mismatched or missing stamp as "not done", so a behaviour change
# cannot silently leave stale results in a sweep.
#   1 - original paper pipeline preserved verbatim + filter-bank front-end only
PIPELINE_VERSION = 1

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
        lines.append(f"FB_CHUNK={FB_CHUNK}")
    else:
        lines.append("WARNING: no CUDA device. Training on CPU is roughly 20-40x slower; "
                     "a full 28-pair sweep would take days.")
    lines.append("NOTE: determinism holds per-GPU-model. Results are not bit-comparable "
                 "across different GPUs (e.g. these will not match the A100 runs).")
    return "\n".join(lines)


def seed_worker(worker_id):     # [P4]
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ===========================================================================
# FILTER BANK FRONT-END (fixed, non-learned, zero trainable parameters)
# ===========================================================================
def _fir_numtaps(trans_hz, fs):
    """Hamming-windowed sinc needs ~3.3 / (trans/fs) taps. Forced odd (Type I, so the
    filter is exactly linear phase and the group delay is an integer numtaps//2)."""
    n = int(np.ceil(3.3 * fs / float(trans_hz)))
    return n + 1 if n % 2 == 0 else n


def make_bandpass_fir(f_lo, f_hi, fs, numtaps):
    """One linear-phase bandpass FIR as a difference of Hamming-windowed sincs.

    Normalised to unit gain at the band centre, so the bank does not re-weight the
    sub-bands relative to each other. Hand-rolled rather than scipy.signal.firwin to
    keep the front-end dependency-free and bit-identical across environments (the
    reference FBCNet bank uses zero-phase Chebyshev-II IIR filters; a linear-phase FIR
    is the same idea with a deterministic, GPU-friendly, purely convolutional form).
    """
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
FB_PAD = FB_NUMTAPS // 2                      # group delay == kernel half-width
FB_H_NP = np.stack([make_bandpass_fir(lo, hi, SAMPLING_RATE, FB_NUMTAPS)
                    for lo, hi in BANDS]).copy()          # (B, numtaps) float64

# Context window fed to the filter bank. The left side uses REAL pre-cue data where
# available; the right side is reflect-padded, so no post-window sample can ever
# influence the output. Identical machinery to the CWT arm.
CTX_START = max(0, CUE_SAMPLE - FB_PAD)
CTX_END = CUE_SAMPLE + WINDOW_SAMPLES
CTX_LEN = CTX_END - CTX_START
LEFT_CTX = CUE_SAMPLE - CTX_START           # real pre-cue samples we actually have
LEFT_REFLECT = FB_PAD - LEFT_CTX            # shortfall made up by reflection
WIN_SLICE = slice(LEFT_CTX, LEFT_CTX + WINDOW_SAMPLES)
# offset of the decision window inside the full 'valid' convolution output
WIN_OFFSET = FB_PAD + LEFT_REFLECT + WIN_SLICE.start
T_OUT = WINDOW_SAMPLES                      # the filter bank does not decimate time

assert WIN_SLICE.stop == CTX_LEN, "window must end exactly at the context end (causality)"
assert LEFT_REFLECT >= 0
assert max(LEFT_REFLECT, FB_PAD) < CTX_LEN, \
    (f"FIR is longer than the context ({FB_NUMTAPS} taps vs {CTX_LEN} samples); "
     f"raise FINE_FB_TRANS")

_KERNEL_CACHE = {}


def _kernels_on(device):
    if device not in _KERNEL_CACHE:
        _KERNEL_CACHE[device] = torch.from_numpy(FB_H_NP).float().to(device)
    return _KERNEL_CACHE[device]


def filter_bank(x_np, device, chunk=None):
    """(N, C, CTX_LEN) -> (N, C, B, WINDOW_SAMPLES) float32.

    FFT convolution with the fixed FIR bank, then crop the group delay and the padding
    so the output is aligned sample-for-sample with the decision window. Runs under
    no_grad; the front-end is a fixed transform, nothing here is trainable or seeded.

    The band axis is placed at dim 2 (not dim 1 as the proposal's N x B x C x T) so that
    C stays the conv in-channel axis and (B, T) is the 2D feature map - same convention
    the Morlet arm uses for (F, T). It is the same tensor, transposed.
    """
    assert x_np.ndim == 3 and x_np.shape[2] == CTX_LEN, \
        f"expected (N, C, {CTX_LEN}), got {x_np.shape}"
    chunk = FB_CHUNK if chunk is None else chunk
    h = _kernels_on(device)

    padded_len = CTX_LEN + LEFT_REFLECT + FB_PAD
    n_fft = 1
    while n_fft < padded_len + FB_NUMTAPS - 1:
        n_fft *= 2

    out = np.empty((x_np.shape[0], x_np.shape[1], N_BANDS, T_OUT), dtype=np.float32)
    with torch.no_grad():
        hf = torch.fft.rfft(h, n=n_fft)                            # (B, n/2+1)
        for s in range(0, x_np.shape[0], chunk):
            xb = torch.from_numpy(np.ascontiguousarray(x_np[s:s + chunk])).float().to(device)
            # Right edge is always reflected (never reads post-window data). The left is
            # reflected only where there was not enough real pre-cue context.
            xb = torch.nn.functional.pad(xb, (LEFT_REFLECT, FB_PAD), mode='reflect')
            xf = torch.fft.rfft(xb, n=n_fft).unsqueeze(-2)          # (b, C, 1, n/2+1)
            lo, hi = WIN_OFFSET, WIN_OFFSET + WINDOW_SAMPLES
            y = torch.fft.irfft(xf * hf, n=n_fft)[..., lo:hi]       # (b, C, B, T)
            out[s:s + chunk] = y.cpu().numpy()
            del xb, xf, y
    return out


def band_normalize(b_train, b_val, b_test):
    """Per-(channel, band) z-score, statistics from the TRAINING fold only.

    Equalises the 1/f amplitude fall-off across sub-bands so every view reaches the
    shared MTC kernels on the same scale. FINE_FB_NORM=none skips it.
    """
    if FB_NORM == 'none':
        return b_train, b_val, b_test
    mu = b_train.mean(axis=(0, 3), keepdims=True)
    sd = b_train.std(axis=(0, 3), keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (b_train - mu) / sd, (b_val - mu) / sd, (b_test - mu) / sd


def apply_frontend(mode, x_train, x_val, x_test, device):
    """Context tensors (N, C, CTX_LEN) -> model-ready tensors."""
    if mode == 'raw':
        return (x_train[:, :, WIN_SLICE].astype(np.float32),
                x_val[:, :, WIN_SLICE].astype(np.float32),
                x_test[:, :, WIN_SLICE].astype(np.float32))
    return band_normalize(filter_bank(x_train, device),
                          filter_bank(x_val, device),
                          filter_bank(x_test, device))


# ===========================================================================
# PREPROCESSING HELPERS
# ===========================================================================
def z_score_normalize_ctx(x_train, x_val, x_test, win_slice):
    """Per-channel z-score. Stats come from the TRAINING fold and, within it, only from
    the decision window - so the filter-bank context padding cannot shift the statistics."""
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
    """Dataset for EEG data. Works for both (C, T) and (C, B, T) samples."""

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
    """CNN model for early EEG classification. Unchanged from the original notebook."""

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
    """Kernels are (1, kt): purely temporal, so each sub-band is convolved on its own.
    No band mixes with another until the spatial conv - the multi-view property."""

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


class FINEFilterBank2D(nn.Module):
    """FINE over (B, T) filter-bank views. Input (N, C, B, T); C are conv in-channels.

    Node-for-node the original 1D graph with Conv1d -> Conv2d and the band axis added:
      MTC (1,k) per band -> spatial 3x3 conv (this is where bands are allowed to mix,
      i.e. FBCNet's per-band spatial filters) -> 2x2 maxpool -> depthwise-separable
      fusion -> ECA -> global mean pool -> the original two-layer head.
    """

    def __init__(self, n_channels, n_bands, n_timepoints, n_classes=2, embedding_dim=128):
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
    return FINEFilterBank2D(n_channels=n_channels, n_bands=N_BANDS, n_timepoints=T_OUT,
                            n_classes=N_CLASSES, embedding_dim=128)


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
    """-> dict {subject_id: {'X': (n, C, CTX_LEN), 'y': (n,)}}, n_channels

    Filters to the two classes and crops the context window PER FILE before
    concatenating. Equivalent to the original concat-then-mask (subject order is
    preserved) but holds a fraction of the memory.
    """
    files = list_subject_files(dataset_root)
    if verbose:
        print(f"Found {len(files)} subject files")
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
            dat = z["data"]                 # decompress once
            if CTX_END > dat.shape[2]:
                raise ValueError(f"context end {CTX_END} exceeds epoch length "
                                 f"{dat.shape[2]} in {path}")
            per_subject[sid] = {'X': dat[m][:, :, CTX_START:CTX_END],
                                'y': np.where(lab[m] == class_b, 1, 0)}
            n_channels = per_subject[sid]['X'].shape[1]
            del dat

    if verbose:
        n0 = sum(int((d['y'] == 0).sum()) for d in per_subject.values())
        n1 = sum(int((d['y'] == 1).sum()) for d in per_subject.values())
        print(f"  Class 0 (original {class_a}): {n0} trials")
        print(f"  Class 1 (original {class_b}): {n1} trials")
        print(f"\nContext tensor per subject: (n, {n_channels}, {CTX_LEN})  "
              f"samples {CTX_START}:{CTX_END}")
        print(f"  decision window = context[{WIN_SLICE.start}:{WIN_SLICE.stop}], "
              f"FIR pad {FB_PAD} ({LEFT_CTX} real + {LEFT_REFLECT} reflected)")
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
                 else f"({n_channels}, {N_BANDS}, {T_OUT})")
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
                    best_model_state = model.state_dict().copy()

            # PRESERVED FROM THE ORIGINAL, DO NOT "FIX":
            # state_dict() hands back references to the live parameters, so .copy()
            # duplicates the dict but not the tensors; optimizer.step() then mutates them
            # in place and this restore is a no-op - the FINAL epoch's weights are what
            # get tested. That is the behaviour behind the paper's Table III, so it stays.
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
                # per-trial predictions: everything else (kappa, F1, confusion matrices,
                # sensitivity/specificity) is recoverable from these without re-running
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
                   pipeline_version=PIPELINE_VERSION,
                   windows=TEST_TIME_WINDOWS_MS_FINAL, seed=SEED, gpu=gpu,
                   torch=torch.__version__, cue_sample=CUE_SAMPLE,
                   frontend=dict(kind='filter_bank',
                                 bands=[[float(lo), float(hi)] for lo, hi in BANDS],
                                 n_bands=int(N_BANDS),
                                 trans_hz=FB_TRANS_HZ, numtaps=int(FB_NUMTAPS),
                                 pad=int(FB_PAD), norm=FB_NORM, decim=1,
                                 # recorded, not staleness-checked: cuFFT picks a
                                 # batch-dependent plan, so the chunk size moves the
                                 # output by ~1e-7 relative (float32 noise)
                                 chunk=int(FB_CHUNK),
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
    """True only if this pair was produced by the CURRENT pipeline.

    A results directory written by older code (or with a different band layout / cue
    alignment) reports False, so a sweep recomputes it rather than silently mixing
    pipelines.
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
    if meta.get('pipeline_version') != PIPELINE_VERSION:
        stale.append(f"pipeline_version {meta.get('pipeline_version')} != {PIPELINE_VERSION}")
    if meta.get('cue_sample') != CUE_SAMPLE:
        stale.append(f"cue_sample {meta.get('cue_sample')} != {CUE_SAMPLE}")
    if not os.path.isfile(os.path.join(pair_dir, "predictions.csv")):
        stale.append("no predictions.csv")
    if mode == 'fbc':
        fe = meta.get('frontend', {})
        got_bands = tuple(tuple(b) for b in fe.get('bands', ()))
        if got_bands != BANDS:
            stale.append(f"bands {got_bands} != {BANDS}")
        if fe.get('numtaps') != FB_NUMTAPS:
            stale.append(f"numtaps {fe.get('numtaps')} != {FB_NUMTAPS}")
        if fe.get('norm') != FB_NORM:
            stale.append(f"norm {fe.get('norm')} != {FB_NORM}")
        got_k = tuple(fe.get('mtc_kernels_2d', (7, 15, 31)))
        if got_k != MTC_KERNELS_2D:
            stale.append(f"mtc_kernels_2d {got_k} != {MTC_KERNELS_2D}")
    if stale and verbose:
        print(f"  stale {mode}/{pair_slug(class_a, class_b)}: {'; '.join(stale)}")
    return not stale
