"""Experiment config: ShallowConvNet (Schirrmeister et al. 2017) on the fixation-cross-
baseline, motor-imagery-only FineMI data at the full 250 Hz.

The data is the same kind as in 'Fine Conformer Fixation Cross Fix': each epoch is 0 to
4 s after the trigger ("Start Imagining"), 1000 samples at 250 Hz, baseline-corrected on
the fixation cross (-4 to -2 s). Windows are the first 800 / 1500 / 3000 / 4000 ms after
imagery onset (200 / 375 / 750 / 1000 samples).

The training loop is the one LEAN and EEG Conformer used (same CV splits, z-score,
noise augmentation, seeds, Adam, 50 epochs, best-validation checkpoint), so results
pair with theirs subject by subject and fold by fold. The one difference is that there
is NO decimation: ShallowConvNet keeps its published 250 Hz kernel and pool sizes.

ShallowConvNet_0.5-3 and ShallowConvNet_8-30 differ ONLY in the band block below.
"""
from __future__ import annotations   # `int | None` field annotations on Python < 3.10

from dataclasses import dataclass, field, asdict
from typing import Tuple
from copy import deepcopy
from itertools import combinations
import json, os

# ---- Band (the only block that differs between the two ShallowConvNet folders) ----
BAND_TAG = '0.5_3hz'          # only subject{N}_eeg_epochs_<BAND_TAG>_*.npz are loaded
BAND_LABEL = '0.5-3 Hz'
BAND_HZ = (0.5, 3.0)          # the preflight checks that most EEG power lies in here
DATASET_DIRNAME = 'fixationcrossbaseline_0.5_3hz_MIonly'
# Earlier runs that summarize.py compares against, paired over subjects (same subjects,
# same labels, so the same CV folds). Paths are relative to this project folder and
# {mode} is filled in; a missing folder is reported and skipped.
REFERENCES = {
    'LEAN': os.path.join('..', 'Fine Conformer Fixation Cross Fix', 'results', '{mode}',
                         'conformer_b_slim'),
    'EEG Conformer': os.path.join('..', 'Fine Conformer Fixation Cross Fix', 'results',
                                  '{mode}', 'conformer_a'),
}
# -----------------------------------------------------------------------------------

# Bump whenever a change alters the numbers a run produces. It is stamped into
# results.json, and a results folder with another stamp is treated as not done.
PIPELINE_VERSION = 1

DECIM = 1                      # no decimation: ShallowConvNet runs at 250 Hz
ORIG_SAMPLING_RATE = 250
# 0 to 4 s after the trigger at 250 Hz. The preflight fails on any other length, so a
# run can't silently use the old -0.5 to 4.0 s (1126-sample) epochs.
EXPECTED_EPOCH_SAMPLES = 1000

PAIRS = list(combinations(range(8), 2))   # 28 pairs, deterministic order
JOINT_NAMES = ('HOC', 'WFE', 'WAA', 'EPS', 'EFE', 'SPS', 'SAA', 'SFE')

MODELS = ('shallowconvnet',)
MODEL_LABELS = {'shallowconvnet': 'ShallowConvNet'}

# Exact parameter counts per window (ms), asserted by the preflight and check_results.
# 100,320 in the conv/BN layers plus a classifier that grows with the pooled length
# (7 / 19 / 44 / 61 steps): 40 * steps * 2 + 2.
EXPECTED_PARAMS = {
    'shallowconvnet': {800: 100_882, 1500: 101_842, 3000: 103_842, 4000: 105_202},
}

# Three pairs for the smoke test.
DEFAULT_PAIRS = ('WAA_SAA', 'HOC_WFE', 'EPS_SPS')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Local layout: Fine MI/<DATASET_DIRNAME>. On Vast.ai, run_vast.sh passes
# --dataset-root explicitly.
DATASET_ROOT_DEFAULT = os.path.normpath(
    os.path.join(PROJECT_ROOT, '..', '..', DATASET_DIRNAME))
RESULTS_ROOT_DEFAULT = os.path.join(PROJECT_ROOT, 'results')


@dataclass
class ExperimentConfig:
    # Identity
    model:              str

    # Data
    dataset_root:       str
    band_tag:           str  = BAND_TAG
    orig_sampling_rate: int  = ORIG_SAMPLING_RATE
    decim:              int  = DECIM
    sampling_rate:      int  = field(init=False)   # orig_sampling_rate // decim = 250
    n_channels:         int  = 62
    test_time_windows_ms: Tuple[int, ...] = (800, 1500, 3000, 4000)

    # Class labels (index -> abbreviation)
    joint_names: Tuple[str, ...] = JOINT_NAMES

    # CV
    n_folds:           int   = 5
    val_size_of_train: float = 0.25

    # Training (identical to LEAN / EEG Conformer in 'Fine Conformer Fixation Cross Fix')
    batch_size:   int   = 32
    n_epochs:     int   = 50
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    noise_level:  float = 0.15
    seed:         int   = 42

    # ShallowConvNet as in the FineMI authors' EEG_decoding.py (Yi et al. 2025) at 250 Hz:
    # 25-sample (100 ms) temporal conv, full-scalp spatial conv, square, 75-sample
    # (300 ms) mean pool every 15 samples (60 ms), log, dropout, linear classifier,
    # softmax; max-norm 2.0 / 2.0 / 0.5 on the temporal conv / spatial conv / classifier.
    shallowconvnet: dict = field(default_factory=lambda: dict(
        n_filters_time=40, filter_time_length=25,
        n_filters_spat=40,
        pool_time_length=75, pool_time_stride=15,
        bn_momentum=0.1, drop_prob=0.5,
        max_norm_temporal_conv=2.0, max_norm_spatial_conv=2.0, max_norm_linear=0.5))

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
        """Window lengths in samples (200 / 375 / 750 / 1000 at 250 Hz)."""
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
    """<results_root>/<mode> — holds one dir per model plus summary/."""
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
