"""Experiment config: FINE + GRU and FINE + GRU + CWT on the 0.5-3 Hz FineMI data,
DECIMATED 250 -> 50 Hz (D = 5).

Built from Conformer_decimated_CWT/src/config.py, so data, decimation, windows, CV
folds, seeds and training hyperparameters are identical and every subject pairs up
with that project's Conformer A / B / B + CWT results. Differences:
  - two models, each Conformer B with its 2-layer transformer replaced by a 2-layer
    bidirectional GRU (src/models/gru_tail.py); everything else is unchanged:
        fine_gru      FINE MTC + spatial conv front-end (as conformer_b)
        fine_gru_cwt  Morlet-scalogram 2D front-end (as conformer_b_cwt)
  - no positional embedding (the GRU is order-aware)
  - the 3/3 pool in front of the GRU is Option B's, so each window still gives
    13 / 25 / 50 / 66 tokens (40 / 75 / 150 / 200 samples at 50 Hz)
  - results go to GRU_FINE/results/<mode>/<model>/<PAIR_SLUG>/; the reference
    models are read from ../Conformer_decimated_CWT/results (compare_to_reference)
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

MODELS = ('fine_gru', 'fine_gru_cwt')
MODEL_LABELS = {
    'fine_gru': 'FINE + GRU',
    'fine_gru_cwt': 'FINE + GRU + CWT',
}
CWT_MODELS = ('fine_gru_cwt',)             # models that get the Morlet front-end

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # GRU_FINE/
# Local layout (Fine MI/FineMI_0.5_3hz). On TACC, submit_ls6.slurm passes
# --dataset-root $SCRATCH/finemi-dataset/FineMI_0.5_3hz instead.
DATASET_ROOT_DEFAULT = os.path.normpath(
    os.path.join(PROJECT_ROOT, '..', '..', 'FineMI_0.5_3hz'))
RESULTS_ROOT_DEFAULT = os.path.join(PROJECT_ROOT, 'results')

# Finished Conformer A / B / B + CWT sweep on the same data (scripts/compare_to_reference)
REFERENCE_ROOT_DEFAULT = os.path.normpath(
    os.path.join(PROJECT_ROOT, '..', 'Conformer_decimated_CWT', 'results'))
REFERENCE_MODELS = ('conformer_a', 'conformer_b', 'conformer_b_cwt')
REFERENCE_LABELS = {
    'conformer_a': 'Conformer A',
    'conformer_b': 'Conformer B',
    'conformer_b_cwt': 'Conformer B + CWT',
}


def _gru_defaults() -> dict:
    """Option B's front-end, fusion and head numbers; the transformer keys
    (n_layers, n_heads, d_ff, p_att, norm_first) are replaced by the GRU's."""
    return dict(
        mtc_kernels=(7, 15, 31), mtc_filters=32,
        spatial_filters=64, spatial_kernel=3,
        pre_rnn_pool_kernel=15 // DECIM, pre_rnn_pool_stride=15 // DECIM,   # 3 / 3
        gru_hidden=32, gru_layers=2, gru_bidirectional=True,   # 2 x 32 = 64 = d_model
        p_rnn=0.3,                    # between GRU layers and on the residual branch
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

    # FINE + GRU (MTC kernels 7/15/31 samples = 140/300/620 ms at 50 Hz)
    fine_gru: dict = field(default_factory=_gru_defaults)

    # FINE + GRU + CWT: same numbers; the MTC and spatial convs become 2D over (F, T)
    fine_gru_cwt: dict = field(default_factory=_gru_defaults)

    # Morlet front-end (fine_gru_cwt only), evaluated on the 50 Hz signal.
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
