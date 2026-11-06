"""Experiment config: EEGNet-8,2 (Lawhern et al. 2018) on the fixation-cross-baseline,
motor-imagery-only FineMI data, 0.5-3 Hz, at two sampling rates:

  eegnet_250   250 Hz, no decimation (as the ShallowConvNet runs)
  eegnet_50    decimated by D = 5 to 50 Hz (as LEAN and EEG Conformer)

The data is the same as in 'Fine Conformer Fixation Cross Fix': each epoch is 0 to 4 s
after the trigger ("Start Imagining"), 1000 samples at 250 Hz, baseline-corrected on
the fixation cross (-4 to -2 s). Windows are the first 800 / 1500 / 3000 / 4000 ms after
imagery onset: 200 / 375 / 750 / 1000 samples at 250 Hz, 40 / 75 / 150 / 200 at 50 Hz.

The training loop is the one LEAN, EEG Conformer and ShallowConvNet used (same CV
splits, z-score, noise augmentation, seeds, Adam, 50 epochs, best-validation
checkpoint), so results pair with theirs subject by subject and fold by fold.
"""
from __future__ import annotations   # `int | None` field annotations on Python < 3.10

from dataclasses import dataclass, field, asdict
from typing import Tuple
from copy import deepcopy
from itertools import combinations
import json, os

# ---- Band ---------------------------------------------------------------------------
BAND_TAG = '0.5_3hz'          # only subject{N}_eeg_epochs_<BAND_TAG>_*.npz are loaded
BAND_LABEL = '0.5-3 Hz'
BAND_HZ = (0.5, 3.0)          # the preflight checks that most EEG power lies in here
DATASET_DIRNAME = 'fixationcrossbaseline_0.5_3hz_MIonly'
# Earlier runs on the same data that summarize.py compares against, paired over
# subjects (same subjects, same labels, so the same CV folds). Paths are relative to
# this project folder and {mode} is filled in; a missing folder is reported and skipped.
REFERENCES = {
    'LEAN': os.path.join('..', 'Fine Conformer Fixation Cross Fix', 'results', '{mode}',
                         'conformer_b_slim'),
    'EEG Conformer': os.path.join('..', 'Fine Conformer Fixation Cross Fix', 'results',
                                  '{mode}', 'conformer_a'),
    'ShallowConvNet': os.path.join('..', 'ShallowConvNet_0.5-3', 'results', '{mode}',
                                   'shallowconvnet'),
}
# -----------------------------------------------------------------------------------

# Bump whenever a change alters the numbers a run produces. It is stamped into
# results.json, and a results folder with another stamp is treated as not done.
PIPELINE_VERSION = 1

ORIG_SAMPLING_RATE = 250
# 0 to 4 s after the trigger at 250 Hz. The preflight fails on any other length, so a
# run can't silently use the old -0.5 to 4.0 s (1126-sample) epochs.
EXPECTED_EPOCH_SAMPLES = 1000

PAIRS = list(combinations(range(8), 2))   # 28 pairs, deterministic order
JOINT_NAMES = ('HOC', 'WFE', 'WAA', 'EPS', 'EFE', 'SPS', 'SAA', 'SFE')

MODELS = ('eegnet_250', 'eegnet_50')
MODEL_LABELS = {'eegnet_250': 'EEGNet 250 Hz', 'eegnet_50': 'EEGNet 50 Hz'}

# Decimation factor per model: keep every D-th sample of the 250 Hz epoch.
MODEL_DECIM = {'eegnet_250': 1, 'eegnet_50': 5}

# EEGNet-8,2 hyperparameters per model. Shared: F1 8, D 2, F2 16, dropout 0.5, max-norm
# 1.0 on the depthwise spatial conv and 0.25 on the classifier (Lawhern et al. 2018).
# The temporal kernel is half the sampling rate (0.5 s) at both rates; pools and the
# separable kernel are the paper's 4 / 16 / 8 at 250 Hz and halved at 50 Hz.
# BatchNorm: eps 1e-3 as in Keras; momentum 0.1 (PyTorch default, as the ShallowConvNet
# runs) rather than Keras' 0.99 (= 0.01 here), which would leave the running statistics
# far from converged after the ~150 updates a 50-epoch fold gets.
_EEGNET_SHARED = dict(F1=8, D=2, F2=16, drop_prob=0.5, bn_momentum=0.1, bn_eps=1e-3,
                      max_norm_depthwise=1.0, max_norm_linear=0.25)
EEGNET_HPARAMS = {
    'eegnet_250': dict(_EEGNET_SHARED, kern_length=125, pool1=4, sep_kern_length=16, pool2=8),
    'eegnet_50':  dict(_EEGNET_SHARED, kern_length=25, pool1=2, sep_kern_length=8, pool2=4),
}

# Exact parameter counts per window (ms), asserted by the preflight and check_results.
# Conv/BN layers: 8*K1 + 16 + 16*62 + 32 + 16*K2 + 16*16 + 32 (2,584 at 250 Hz, 1,656
# at 50 Hz), plus a classifier 16 * n_pool_out * 2 + 2, where n_pool_out =
# (T // P1) // P2 = 6 / 11 / 23 / 31 at 250 Hz and 5 / 9 / 18 / 25 at 50 Hz.
EXPECTED_PARAMS = {
    'eegnet_250': {800: 2_778, 1500: 2_938, 3000: 3_322, 4000: 3_578},
    'eegnet_50':  {800: 1_818, 1500: 1_946, 3000: 2_234, 4000: 2_458},
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
    decim:              int  = 1                    # set from MODEL_DECIM by make_config
    sampling_rate:      int  = field(init=False)   # orig_sampling_rate // decim
    n_channels:         int  = 62
    test_time_windows_ms: Tuple[int, ...] = (800, 1500, 3000, 4000)

    # Class labels (index -> abbreviation)
    joint_names: Tuple[str, ...] = JOINT_NAMES

    # CV
    n_folds:           int   = 5
    val_size_of_train: float = 0.25

    # Training (identical to LEAN / EEG Conformer / ShallowConvNet)
    batch_size:   int   = 32
    n_epochs:     int   = 50
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    noise_level:  float = 0.15
    seed:         int   = 42

    # EEGNet hyperparameters for this model (EEGNET_HPARAMS[model], set by make_config)
    eegnet: dict = field(default_factory=dict)

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
        """Window lengths in samples (200 / 375 / 750 / 1000 at 250 Hz, 40 / 75 / 150 /
        200 at 50 Hz)."""
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
    base = ExperimentConfig(model=model, dataset_root=dataset_root,
                            decim=MODEL_DECIM[model], eegnet=dict(EEGNET_HPARAMS[model]))
    cfg = apply_mode(base, mode)
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
