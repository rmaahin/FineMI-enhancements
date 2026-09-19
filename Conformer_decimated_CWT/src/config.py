"""Experiment config: Conformer A / Conformer B / Conformer B + CWT on the 0.5-3 Hz
FineMI data, DECIMATED 250 -> 50 Hz (D = 5).

Same training hyperparameters as Conformer_0.5_3hz/src/config.py. Differences:
  - three models: conformer_a, conformer_b, conformer_b_cwt (no FINE baseline)
  - every trial is decimated by D = 5 (keep every 5th sample) right after loading,
    so the model sees 50 Hz data: the 1126-sample epoch becomes 226 samples and the
    800 / 1500 / 3000 / 4000 ms windows become 40 / 75 / 150 / 200 samples
  - convolution kernels stay FIXED IN SAMPLES (the decimation rationale: a fixed
    receptive field spans 5x more time at 50 Hz). Only the pooling in front of the
    transformer is divided by D, because Option A's 75-sample pool does not fit the
    40-sample 800 ms window. With the pool divided by D, every window gives exactly
    the same number of transformer tokens as the 250 Hz runs:
        Option A: pool 75/15 -> 15/3   tokens  9 / 21 / 46 / 62
        Option B: pool 15/15 ->  3/3   tokens 13 / 25 / 50 / 66
  - conformer_b_cwt = Option B with a Morlet-scalogram front-end (src/cwt.py):
    input (C, F, T) = (62, 24, T) and a 2D version of the MTC + spatial blocks
  - results go to Conformer_decimated_CWT/results/<mode>/<model>/<PAIR_SLUG>/
"""
from __future__ import annotations   # `int | None` field annotations on Python < 3.10

from dataclasses import dataclass, field, asdict
from typing import Tuple
from copy import deepcopy
from itertools import combinations
import json, os

# Bump whenever a change alters the numbers a run produces. It is stamped into
# results.json, and a results folder with another stamp is treated as not done.
PIPELINE_VERSION = 1

DECIM = 5                      # D in Eq. (1): keep every 5th sample
ORIG_SAMPLING_RATE = 250

PAIRS = list(combinations(range(8), 2))   # 28 pairs, deterministic order
JOINT_NAMES = ('HOC', 'WFE', 'WAA', 'EPS', 'EFE', 'SPS', 'SAA', 'SFE')

MODELS = ('conformer_a', 'conformer_b', 'conformer_b_cwt')
MODEL_LABELS = {
    'conformer_a': 'Conformer A',
    'conformer_b': 'Conformer B',
    'conformer_b_cwt': 'Conformer B + CWT',
}
CWT_MODELS = ('conformer_b_cwt',)          # models that get the Morlet front-end

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Conformer_decimated_CWT/
# Local layout (Fine MI/FineMI_0.5_3hz). On TACC, submit_ls6.slurm passes
# --dataset-root $SCRATCH/finemi-dataset/FineMI_0.5_3hz instead.
DATASET_ROOT_DEFAULT = os.path.normpath(
    os.path.join(PROJECT_ROOT, '..', '..', 'FineMI_0.5_3hz'))
RESULTS_ROOT_DEFAULT = os.path.join(PROJECT_ROOT, 'results')


def _option_b_defaults() -> dict:
    return dict(
        mtc_kernels=(7, 15, 31), mtc_filters=32,
        spatial_filters=64, spatial_kernel=3,
        pre_att_pool_kernel=15 // DECIM, pre_att_pool_stride=15 // DECIM,   # 3 / 3
        n_layers=2, n_heads=8, d_ff=256, p_att=0.3, norm_first=True,
        fusion_kernel=5, eca_reduction=4,
        embedding_dim=128, p_head=0.4)


@dataclass
class ExperimentConfig:
    # Identity
    model:              str

    # Data
    dataset_root:       str
    band_tag:           str  = '0.5_3hz'   # only subject{N}_eeg_epochs_<band_tag>_*.npz are loaded
    orig_sampling_rate: int  = ORIG_SAMPLING_RATE
    decim:              int  = DECIM
    sampling_rate:      int  = field(init=False)   # orig_sampling_rate // decim = 50
    n_channels:         int  = 62
    test_time_windows_ms: Tuple[int, ...] = (800, 1500, 3000, 4000)

    # Class labels (index -> abbreviation)
    joint_names: Tuple[str, ...] = JOINT_NAMES

    # CV
    n_folds:           int   = 5
    val_size_of_train: float = 0.25

    # Training
    batch_size:   int   = 32
    n_epochs:     int   = 50
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    noise_level:  float = 0.15
    seed:         int   = 42

    # Option A hyperparameters (temporal kernel 25 samples = 500 ms at 50 Hz)
    conformer_a: dict = field(default_factory=lambda: dict(
        F1=40, temporal_kernel=25,
        pool_kernel=75 // DECIM, pool_stride=15 // DECIM,                  # 15 / 3
        p_conv=0.5, n_layers=2, n_heads=10, d_ff=160,
        p_att=0.5, norm_first=True, proj_dim=32, p_head=0.3))

    # Option B hyperparameters (MTC kernels 7/15/31 samples = 140/300/620 ms at 50 Hz)
    conformer_b: dict = field(default_factory=_option_b_defaults)

    # Option B + CWT: same numbers; the MTC and spatial convs become 2D over (F, T)
    conformer_b_cwt: dict = field(default_factory=_option_b_defaults)

    # Morlet front-end (conformer_b_cwt only), evaluated on the 50 Hz signal.
    # Same bank as CWT-modified/fine_mi.py.
    cwt: dict = field(default_factory=lambda: dict(
        fmin=0.5, fmax=3.0, n_freqs=24,
        n_cycles_min=3.0, n_cycles_max=7.0,   # linear ramp over the 24 rows
        trunc=5.0,                            # kernel support in sigma_t
        log_eps=1e-10,                        # absolute floor; input must be z-scored
        chunk=64))                            # trials per FFT batch (VRAM knob)

    # Output: <results_root>/<PAIR_SLUG>/ (already namespaced by mode and model)
    results_root: str = ""

    # Execution mode (set by apply_mode)
    _max_subjects: int | None = None
    _mode: str = 'full'

    def __post_init__(self):
        if self.decim < 1 or self.orig_sampling_rate % self.decim != 0:
            raise ValueError(f"bad decim={self.decim} for {self.orig_sampling_rate} Hz")
        self.sampling_rate = self.orig_sampling_rate // self.decim

    def window_samples(self) -> list[int]:
        """Window lengths in DECIMATED samples (40 / 75 / 150 / 200 at 50 Hz)."""
        return [int(t * self.sampling_rate / 1000) for t in self.test_time_windows_ms]

    def to_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # write-then-rename: parallel SLURM workers may write the same config at once
        tmp_path = f"{path}.{os.getpid()}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp_path, path)


# smoke runs the SAME code path as full (K-fold CV, all four windows, table, stats),
# only smaller: 2 subjects, 2 folds, 2 epochs.
MODE_OVERRIDES = {
    'smoke': dict(
        max_subjects=2, n_folds=2, n_epochs=2,
        test_time_windows_ms=(800, 1500, 3000, 4000)),
    'full': dict(
        max_subjects=None, n_folds=5, n_epochs=50,
        test_time_windows_ms=(800, 1500, 3000, 4000)),
}


def apply_mode(cfg: ExperimentConfig, mode: str) -> ExperimentConfig:
    if mode not in MODE_OVERRIDES:
        raise ValueError(f"unknown mode: {mode}")
    cfg = deepcopy(cfg)
    ov = MODE_OVERRIDES[mode]
    cfg.n_folds = ov['n_folds']
    cfg.n_epochs = ov['n_epochs']
    cfg.test_time_windows_ms = ov['test_time_windows_ms']
    cfg._max_subjects = ov['max_subjects']            # None = all
    cfg._mode = mode
    return cfg


def mode_root(results_root: str, mode: str) -> str:
    """<results_root>/<mode> — holds one dir per model plus comparison/."""
    return os.path.join(results_root, mode)


def make_config(model: str, mode: str,
                dataset_root: str = DATASET_ROOT_DEFAULT,
                results_root: str = RESULTS_ROOT_DEFAULT) -> ExperimentConfig:
    """Config for `model` with `mode` overrides; results under <results_root>/<mode>/<model>."""
    if model not in MODELS:
        raise ValueError(f"unknown model: {model} (expected one of {MODELS})")
    cfg = apply_mode(ExperimentConfig(model=model, dataset_root=dataset_root), mode)
    cfg.results_root = os.path.join(mode_root(results_root, mode), model)
    return cfg


def pair_slug(class_a: int, class_b: int, joint_names=JOINT_NAMES) -> str:
    return f"{joint_names[class_a]}_{joint_names[class_b]}"


def parse_pair(token: str, joint_names=JOINT_NAMES) -> tuple[int, int]:
    """Parse one --pairs entry into (class_a, class_b) with class_a < class_b.

    Accepted forms (case-insensitive):
      WAA_SAA, WAA/SAA, WAA-SAA, WAA,SAA   joint abbreviations
      2_6, 2-6, 2/6, 2,6                   class ids 0-7
      13                                   pair index 0-27 into PAIRS
    Reversed order (SAA_WAA) is normalized to PAIRS order (WAA_SAA), so each
    pair always lands in the same results folder.
    """
    t = token.strip()
    for sep in ('/', '-', ','):
        t = t.replace(sep, '_')
    parts = [p for p in t.split('_') if p]

    if len(parts) == 1 and parts[0].isdigit():
        idx = int(parts[0])
        if not 0 <= idx < len(PAIRS):
            raise ValueError(f"pair index must be in [0, {len(PAIRS) - 1}], got {token!r}")
        return PAIRS[idx]

    if len(parts) != 2:
        raise ValueError(f"cannot parse pair {token!r} (use e.g. WAA_SAA, 2_6 or a pair index)")

    upper = [j.upper() for j in joint_names]
    classes = []
    for p in parts:
        if p.isdigit() and 0 <= int(p) < len(joint_names):
            classes.append(int(p))
        elif p.upper() in upper:
            classes.append(upper.index(p.upper()))
        else:
            raise ValueError(f"unknown joint/class {p!r} in pair {token!r} "
                             f"(joints: {', '.join(joint_names)}; classes 0-{len(joint_names) - 1})")
    a, b = sorted(classes)
    if a == b:
        raise ValueError(f"pair {token!r} uses the same class twice")
    return a, b


def parse_pairs(tokens) -> list[tuple[int, int]]:
    """Parse --pairs entries into a de-duplicated list, in the order given.

    The token 'all' (case-insensitive) expands to all 28 pairs in PAIRS order.
    Raises ValueError on a bad token.
    """
    pairs = []
    for token in tokens:
        expanded = PAIRS if token.strip().lower() == 'all' else [parse_pair(token)]
        for pair in expanded:
            if pair not in pairs:
                pairs.append(pair)
    return pairs
