"""Experiment config: the two ICASSP 2027 models on the OFNER motor-imagery data
(Ofner et al. 2017; preprocess/preprocess_ofner.py), in two bands.

Models (the paper's names in brackets):
  - conformer_b_slim  [LEAN]           25,139 parameters at 50 Hz, 38,071 at 250 Hz
  - conformer_a       [EEG Conformer]  151,738 / 167,098 / 199,098 parameters at
                                        800 / 1500 / 3000 ms (the head grows with the
                                        number of tokens), in both bands
Both are copied unchanged from Fine Conformer Fixation Cross Fix and trained by the
same loop with the same CV splits, normalization, augmentation, seeds, epochs and
checkpoint rule. The counts differ from FineMI's 25,752 / 153,978 ... only because
OFNER has 61 channels (not 62) and the longest window is 3000 ms (not 4000).

DATA: 15 subjects, 360 trials (6 movements x 60), 61 channels; each epoch is 0 to 3 s
after the cue (750 samples at 250 Hz), baseline-corrected on the fixation cross
(-2 to 0 s). Windows are the first 800 / 1500 / 3000 ms after the cue.

BANDS (BAND_PROFILES):
  0.5_3hz  decimated by 5 to 50 Hz, the rate both models were designed at.
  8_30hz   NOT decimated: 50 Hz has a 25 Hz Nyquist, so 25-30 Hz would alias. At
           250 Hz every LEAN layer that acts on time before the pool is scaled by 5 so
           it covers the same milliseconds (temporal kernels 7/15/31 -> 35/75/155,
           pre-attention pool 3/3 -> 15/15); the token counts stay the same. The
           spatial kernel (3) and the fusion kernel (5, on tokens) are unchanged.
           EEG Conformer uses its original 250 Hz config (kernel 25, pool 75/15).
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

ORIG_SAMPLING_RATE = 250
DESIGN_SAMPLING_RATE = 50      # the rate LEAN's kernel sizes were chosen at
# 0 to 3 s after the cue at 250 Hz. The preflight fails on any other length.
EXPECTED_EPOCH_SAMPLES = 750
EXPECTED_TRIALS = 360          # 6 classes x 60
EXPECTED_SUBJECTS = 15

# band tag -> decimation. The tag is also the file tag the preprocessing writes
# (subject{N}_eeg_epochs_<band>_250hz.npz in ofner_mi_<band>/).
BAND_PROFILES = {
    '0.5_3hz': dict(decim=5, label='0.5-3 Hz'),
    '8_30hz': dict(decim=1, label='8-30 Hz'),
}
DEFAULT_BAND = '0.5_3hz'

CLASS_NAMES = ('EF', 'EE', 'SUP', 'PRO', 'HC', 'HO')
CLASS_LONG = ('elbow flexion', 'elbow extension', 'forearm supination',
              'forearm pronation', 'hand close', 'hand open')
PAIRS = list(combinations(range(len(CLASS_NAMES)), 2))   # 15 pairs, deterministic order

# The proposed model first: compare_models reports "LEAN minus EEG Conformer".
MODELS = ('conformer_b_slim', 'conformer_a')
MODEL_LABELS = {'conformer_b_slim': 'LEAN', 'conformer_a': 'EEG Conformer'}

# Exact parameter counts per band and window (ms), asserted by the preflight and
# check_results.
EXPECTED_PARAMS = {
    '0.5_3hz': {
        'conformer_b_slim': {800: 25_139, 1500: 25_139, 3000: 25_139},
        'conformer_a': {800: 151_738, 1500: 167_098, 3000: 199_098},
    },
    '8_30hz': {
        'conformer_b_slim': {800: 38_071, 1500: 38_071, 3000: 38_071},
        'conformer_a': {800: 151_738, 1500: 167_098, 3000: 199_098},
    },
}

# One antagonist pair per joint, for the smoke test.
DEFAULT_PAIRS = ('EF_EE', 'SUP_PRO', 'HC_HO')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_ROOT = (os.path.join(os.environ['SCRATCH'], 'ofner') if os.environ.get('SCRATCH')
              else '/workspace/ofner')
RESULTS_ROOT_DEFAULT = os.path.join(PROJECT_ROOT, 'results')


def normalize_band(band: str) -> str:
    """'0.5_3hz', '0.5_3', '0.5-3', '8_30', '8-30hz' ... -> a BAND_PROFILES key."""
    b = band.strip().lower().replace('-', '_')
    if not b.endswith('hz'):
        b += 'hz'
    if b not in BAND_PROFILES:
        raise ValueError(f"unknown band {band!r} (expected one of {', '.join(BAND_PROFILES)})")
    return b


def dataset_root_default(band: str) -> str:
    """$SCRATCH/ofner/ofner_mi_<band> (TACC) or /workspace/ofner/ofner_mi_<band> (Vast)."""
    return os.path.join(_DATA_ROOT, f"ofner_mi_{normalize_band(band)}")


@dataclass
class ExperimentConfig:
    # Identity
    model:              str

    # Data
    dataset_root:       str
    band_tag:           str  = DEFAULT_BAND   # only subject{N}_eeg_epochs_<band_tag>_*.npz are loaded
    orig_sampling_rate: int  = ORIG_SAMPLING_RATE
    decim:              int  = field(init=False)   # from BAND_PROFILES
    sampling_rate:      int  = field(init=False)   # orig_sampling_rate // decim
    n_channels:         int  = 61
    test_time_windows_ms: Tuple[int, ...] = (800, 1500, 3000)

    # Class labels (index -> abbreviation)
    class_names: Tuple[str, ...] = CLASS_NAMES

    # CV
    n_folds:           int   = 5
    val_size_of_train: float = 0.25

    # Training (identical to Fine Conformer Fixation Cross Fix)
    batch_size:   int   = 32
    n_epochs:     int   = 50
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    noise_level:  float = 0.15
    seed:         int   = 42

    # LEAN = Slim Conformer B. Sizes are given at 50 Hz; __post_init__ scales the
    # time-axis ones (mtc_kernels, pre_att_pool_*) to the band's sampling rate.
    conformer_b_slim: dict = field(default_factory=lambda: dict(
        mtc_kernels=(7, 15, 31),          # samples at 50 Hz: 140 / 300 / 620 ms
        mtc_filters=16,                   # per branch, after the per-electrode filters
        spatial_filters=32, spatial_kernel=3,     # = transformer width (d_model)
        pre_att_pool_kernel=3, pre_att_pool_stride=3,
        n_layers=1, n_heads=4, d_ff=64, p_att=0.3, norm_first=True,
        fusion_kernel=5, eca_reduction=4,
        embedding_dim=32, p_proj=0.3,     # projection width and its dropout
        classifier_hidden=32, p_head=0.4))

    # EEG Conformer = Conformer A. The pool is 75/15 at 250 Hz, divided by the
    # decimation (__post_init__): 15/3 at 50 Hz, the original 75/15 at 250 Hz.
    conformer_a: dict = field(default_factory=lambda: dict(
        F1=40, temporal_kernel=25,
        pool_kernel=75, pool_stride=15,
        p_conv=0.5, n_layers=2, n_heads=10, d_ff=160,
        p_att=0.5, norm_first=True, proj_dim=32, p_head=0.3))

    # Output: <results_root>/<PAIR_SLUG>/ (already namespaced by band, mode and model)
    results_root: str = ""

    # Execution mode (set by apply_mode)
    _max_subjects: int | None = None
    _mode: str = 'full'

    def __post_init__(self):
        self.band_tag = normalize_band(self.band_tag)
        self.decim = BAND_PROFILES[self.band_tag]['decim']
        if self.orig_sampling_rate % self.decim != 0:
            raise ValueError(f"bad decim={self.decim} for {self.orig_sampling_rate} Hz")
        self.sampling_rate = self.orig_sampling_rate // self.decim
        if self.sampling_rate % DESIGN_SAMPLING_RATE != 0:
            raise ValueError(f"{self.sampling_rate} Hz is not a multiple of "
                             f"{DESIGN_SAMPLING_RATE} Hz; LEAN's kernels cannot be scaled")

        scale = self.sampling_rate // DESIGN_SAMPLING_RATE      # 1 at 50 Hz, 5 at 250 Hz
        b = self.conformer_b_slim
        b['mtc_kernels'] = tuple(k * scale for k in b['mtc_kernels'])   # odd x odd = odd
        b['pre_att_pool_kernel'] *= scale
        b['pre_att_pool_stride'] *= scale

        a = self.conformer_a
        a['pool_kernel'] //= self.decim
        a['pool_stride'] //= self.decim

    def window_samples(self) -> list[int]:
        """Window lengths in DECIMATED samples (40 / 75 / 150 at 50 Hz,
        200 / 375 / 750 at 250 Hz)."""
        return [int(t * self.sampling_rate / 1000) for t in self.test_time_windows_ms]

    def to_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # write-then-rename: parallel workers may write the same config at once
        tmp_path = f"{path}.{os.getpid()}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp_path, path)


# smoke runs the SAME code path as full (K-fold CV, all three windows), only smaller:
# 2 subjects, 2 folds, 2 epochs.
MODE_OVERRIDES = {
    'smoke': dict(
        max_subjects=2, n_folds=2, n_epochs=2,
        test_time_windows_ms=(800, 1500, 3000)),
    'full': dict(
        max_subjects=None, n_folds=5, n_epochs=50,
        test_time_windows_ms=(800, 1500, 3000)),
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


def mode_root(results_root: str, band: str, mode: str) -> str:
    """<results_root>/<band>/<mode> — holds one dir per model plus comparison/."""
    return os.path.join(results_root, normalize_band(band), mode)


def make_config(model: str, mode: str, band: str = DEFAULT_BAND,
                dataset_root: str | None = None,
                results_root: str = RESULTS_ROOT_DEFAULT) -> ExperimentConfig:
    """Config for `model` on `band` with `mode` overrides; results under
    <results_root>/<band>/<mode>/<model>."""
    if model not in MODELS:
        raise ValueError(f"unknown model: {model} (expected one of {MODELS})")
    band = normalize_band(band)
    if dataset_root is None:
        dataset_root = dataset_root_default(band)
    cfg = apply_mode(ExperimentConfig(model=model, dataset_root=dataset_root, band_tag=band),
                     mode)
    cfg.results_root = os.path.join(mode_root(results_root, band, mode), model)
    return cfg


def pair_slug(class_a: int, class_b: int, class_names=CLASS_NAMES) -> str:
    return f"{class_names[class_a]}_{class_names[class_b]}"


def parse_pair(token: str, class_names=CLASS_NAMES) -> tuple[int, int]:
    """Parse one --pairs entry into (class_a, class_b) with class_a < class_b.

    Accepted forms (case-insensitive):
      EF_EE, EF/EE, EF-EE, EF,EE           class abbreviations
      0_1, 0-1, 0/1, 0,1                   class ids 0-5
      7                                    pair index 0-14 into PAIRS
    Reversed order (EE_EF) is normalized to PAIRS order (EF_EE).
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
        raise ValueError(f"cannot parse pair {token!r} (use e.g. EF_EE, 0_1 or a pair index)")

    upper = [c.upper() for c in class_names]
    classes = []
    for p in parts:
        if p.isdigit() and 0 <= int(p) < len(class_names):
            classes.append(int(p))
        elif p.upper() in upper:
            classes.append(upper.index(p.upper()))
        else:
            raise ValueError(f"unknown class {p!r} in pair {token!r} "
                             f"(classes: {', '.join(class_names)}; ids 0-{len(class_names) - 1})")
    a, b = sorted(classes)
    if a == b:
        raise ValueError(f"pair {token!r} uses the same class twice")
    return a, b


def parse_pairs(tokens) -> list[tuple[int, int]]:
    """Parse --pairs entries into a de-duplicated list, in the order given.

    The token 'all' (case-insensitive) expands to all 15 pairs in PAIRS order.
    """
    pairs = []
    for token in tokens:
        expanded = PAIRS if token.strip().lower() == 'all' else [parse_pair(token)]
        for pair in expanded:
            if pair not in pairs:
                pairs.append(pair)
    return pairs
