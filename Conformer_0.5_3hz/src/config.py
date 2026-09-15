"""Experiment config for the 0.5-3 Hz FineMI comparison (FINE vs Conformer A vs B).

Same hyperparameters as ConformerEEG/src/config.py; differences:
  - three models ('fine' is the EMBC_deterministic-3.ipynb baseline)
  - data defaults to <Fine MI>/FineMI_0.5_3hz (files subject{N}_eeg_epochs_0.5_3hz_*.npz)
  - results go to Conformer_0.5_3hz/results/<mode>/<model>/<PAIR_SLUG>/
  - pairs are chosen explicitly (parse_pair / parse_pairs, "all" = 28), not by index range
"""
from __future__ import annotations   # `int | None` field annotations on Python < 3.10

from dataclasses import dataclass, field, asdict
from typing import Tuple, Literal
from copy import deepcopy
from itertools import combinations
import json, os

PAIRS = list(combinations(range(8), 2))   # 28 pairs, deterministic order
JOINT_NAMES = ('HOC', 'WFE', 'WAA', 'EPS', 'EFE', 'SPS', 'SAA', 'SFE')

MODELS = ('fine', 'conformer_a', 'conformer_b')
MODEL_LABELS = {
    'fine': 'FINE (base)',
    'conformer_a': 'Conformer A',
    'conformer_b': 'Conformer B',
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Conformer_0.5_3hz/
# Local layout (Fine MI/FineMI_0.5_3hz). On TACC, submit_ls6.slurm passes
# --dataset-root $SCRATCH/finemi-dataset/FineMI_0.5_3hz instead.
DATASET_ROOT_DEFAULT = os.path.normpath(
    os.path.join(PROJECT_ROOT, '..', '..', 'FineMI_0.5_3hz'))
RESULTS_ROOT_DEFAULT = os.path.join(PROJECT_ROOT, 'results')


@dataclass
class ExperimentConfig:
    # Identity
    model:              Literal['fine', 'conformer_a', 'conformer_b']

    # Data
    dataset_root:       str
    band_tag:           str  = '0.5_3hz'   # only subject{N}_eeg_epochs_<band_tag>_*.npz are loaded
    sampling_rate:      int  = 250
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

    # Option A hyperparameters
    conformer_a: dict = field(default_factory=lambda: dict(
        F1=40, temporal_kernel=25, pool_kernel=75, pool_stride=15,
        p_conv=0.5, n_layers=2, n_heads=10, d_ff=160,
        p_att=0.5, norm_first=True, proj_dim=32, p_head=0.3))

    # Option B hyperparameters
    conformer_b: dict = field(default_factory=lambda: dict(
        mtc_kernels=(7, 15, 31), mtc_filters=32,
        spatial_filters=64, spatial_kernel=3,
        pre_att_pool_kernel=15, pre_att_pool_stride=15,
        n_layers=2, n_heads=8, d_ff=256, p_att=0.3, norm_first=True,
        fusion_kernel=5, eca_reduction=4,
        embedding_dim=128, p_head=0.4))

    # Output: <results_root>/<PAIR_SLUG>/ (already namespaced by mode and model)
    results_root: str = ""

    # Execution mode (set by apply_mode)
    _max_subjects: int | None = None
    _mode: str = 'full'

    def to_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # write-then-rename: parallel SLURM workers may write the same config at once
        tmp_path = f"{path}.{os.getpid()}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp_path, path)


MODE_OVERRIDES = {
    'smoke': dict(
        max_subjects=1, n_folds=1, n_epochs=2,
        test_time_windows_ms=(800,)),
    'sanity': dict(
        max_subjects=3, n_folds=5, n_epochs=10,
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
