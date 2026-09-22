"""Experiment config: Slim Conformer B on the 0.5-3 Hz FineMI data, decimated 250 -> 50 Hz.

Same data pipeline and training loop as Conformer_decimated_CWT (decimation D = 5,
windows 800 / 1500 / 3000 / 4000 ms = 40 / 75 / 150 / 200 samples from cue onset,
5-fold within-subject CV, 50 epochs, deepcopy best-validation checkpoint, seed 42),
so its results pair subject-by-subject with that project's Conformer A and B.

The model is Conformer B cut to 25,752 parameters (step 4 of the slimming ladder):
  - temporal branches: one 7 / 15 / 31-sample filter per electrode (depthwise), then a
    1x1 conv to 16 filters per branch                               6,358
  - spatial conv 48 -> 32 channels, kernel 3                        4,672
  - positional embedding, 66 tokens x 32                            2,112
  - 1 transformer layer, width 32, 4 heads, feed-forward 64         8,544
  - final LayerNorm                                                    64
  - depthwise-separable fusion conv + channel attention (ECA)       1,824
  - head: 32 -> 32 projection, 32 -> 32 -> 2 classifier             2,178
                                                          total    25,752
The pool in front of the transformer stays 3/3, so every window gives the same
13 / 25 / 50 / 66 tokens as Conformer B. The count does not depend on the window.
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

DECIM = 5                      # keep every 5th sample: 250 -> 50 Hz
ORIG_SAMPLING_RATE = 250

PAIRS = list(combinations(range(8), 2))   # 28 pairs, deterministic order
JOINT_NAMES = ('HOC', 'WFE', 'WAA', 'EPS', 'EFE', 'SPS', 'SAA', 'SFE')

MODELS = ('conformer_b_slim',)
MODEL_LABELS = {'conformer_b_slim': 'Slim Conformer B'}

# The preflight fails if the built model does not have exactly this many parameters.
EXPECTED_PARAMS = {'conformer_b_slim': 25_752}

# The 3-pair pilot used for Conformer_0.5_3hz; the local test runs these by default.
DEFAULT_PAIRS = ('WAA_SAA', 'HOC_WFE', 'EPS_SPS')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Slim_conformer_fine/
# Local layout (Fine MI/FineMI_0.5_3hz). On TACC, submit_ls6.slurm passes
# --dataset-root $SCRATCH/finemi-dataset/FineMI_0.5_3hz instead.
DATASET_ROOT_DEFAULT = os.path.normpath(
    os.path.join(PROJECT_ROOT, '..', '..', 'FineMI_0.5_3hz'))
RESULTS_ROOT_DEFAULT = os.path.join(PROJECT_ROOT, 'results')

# Conformer A and B at 50 Hz, same folds and seeds: the reference for "on par".
REFERENCE_ROOT_DEFAULT = os.path.normpath(
    os.path.join(PROJECT_ROOT, '..', 'Conformer_decimated_CWT', 'results'))
REFERENCE_MODELS = ('conformer_a', 'conformer_b')
REFERENCE_LABELS = {'conformer_a': 'Conformer A', 'conformer_b': 'Conformer B'}
# Measured by that project's preflight on TACC (A's head grows with the window).
REFERENCE_PARAMS = {
    'conformer_a': {800: 153_978, 1500: 169_338, 3000: 201_338, 4000: 221_818},
    'conformer_b': {800: 260_034, 1500: 260_034, 3000: 260_034, 4000: 260_034},
}


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

    # Training (identical to Conformer_decimated_CWT)
    batch_size:   int   = 32
    n_epochs:     int   = 50
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    noise_level:  float = 0.15
    seed:         int   = 42

    # Slim Conformer B hyperparameters (25,752 parameters)
    conformer_b_slim: dict = field(default_factory=lambda: dict(
        mtc_kernels=(7, 15, 31),          # samples: 140 / 300 / 620 ms at 50 Hz
        mtc_filters=16,                   # per branch, after the per-electrode filters
        spatial_filters=32, spatial_kernel=3,     # = transformer width (d_model)
        pre_att_pool_kernel=3, pre_att_pool_stride=3,
        n_layers=1, n_heads=4, d_ff=64, p_att=0.3, norm_first=True,
        fusion_kernel=5, eca_reduction=4,
        embedding_dim=32, p_proj=0.3,     # projection width and its dropout
        classifier_hidden=32, p_head=0.4))

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
        # write-then-rename: parallel workers may write the same config at once
        tmp_path = f"{path}.{os.getpid()}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp_path, path)


# smoke runs the SAME code path as full (K-fold CV, all four windows), only smaller:
# 2 subjects, 2 folds, 2 epochs.
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
    Reversed order (SAA_WAA) is normalized to PAIRS order (WAA_SAA).
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
    """
    pairs = []
    for token in tokens:
        expanded = PAIRS if token.strip().lower() == 'all' else [parse_pair(token)]
        for pair in expanded:
            if pair not in pairs:
                pairs.append(pair)
    return pairs
