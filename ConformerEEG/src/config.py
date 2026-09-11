from __future__ import annotations   # `int | None` field annotations on Python < 3.10

from dataclasses import dataclass, field, asdict
from typing import Tuple, Literal
from copy import deepcopy
from itertools import combinations
import json, os

PAIRS = list(combinations(range(8), 2))   # 28 pairs, deterministic order


@dataclass
class ExperimentConfig:
    # Identity
    model:              Literal['fine', 'conformer_a', 'conformer_b']
    experiment_name:    str

    # Data
    dataset_root:       str
    sampling_rate:      int  = 250
    n_channels:         int  = 62
    test_time_windows_ms: Tuple[int, ...] = (800, 1500, 3000, 4000)

    # Class labels (index → abbreviation)
    joint_names: Tuple[str, ...] = (
        'HOC', 'WFE', 'WAA', 'EPS', 'EFE', 'SPS', 'SAA', 'SFE')
    n_pairs: int = 28   # C(8, 2)

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

    # Output
    results_root: str = ""

    # Execution mode (set by apply_mode, §11)
    _max_subjects: int | None = None
    _max_pairs_per_range: int | None = None
    _mode: str = 'full'

    def to_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


def _dataset_root(scratch: str) -> str:
    # FINEMI_DATASET_ROOT, if set, overrides the default location under $SCRATCH
    return os.environ.get('FINEMI_DATASET_ROOT') or f"{scratch}/finemi-dataset/FineMI"


def config_option_a(scratch: str, repo_root: str) -> ExperimentConfig:
    return ExperimentConfig(
        model='conformer_a',
        experiment_name='option_a',
        dataset_root=_dataset_root(scratch),
        results_root=f"{repo_root}/experiments/option_a/results")

def config_option_b(scratch: str, repo_root: str) -> ExperimentConfig:
    return ExperimentConfig(
        model='conformer_b',
        experiment_name='option_b',
        dataset_root=_dataset_root(scratch),
        results_root=f"{repo_root}/experiments/option_b/results")


CONFIG_FACTORIES = {
    'conformer_a': config_option_a,
    'conformer_b': config_option_b,
}


MODE_OVERRIDES = {
    'smoke': dict(
        max_subjects=1, n_folds=1, n_epochs=2,
        test_time_windows_ms=(800,), max_pairs_per_range=1),
    'sanity': dict(
        max_subjects=3, n_folds=5, n_epochs=10,
        test_time_windows_ms=(800, 1500, 3000, 4000),
        max_pairs_per_range=None),
    'full': dict(
        max_subjects=None, n_folds=5, n_epochs=50,
        test_time_windows_ms=(800, 1500, 3000, 4000),
        max_pairs_per_range=None),
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
    cfg._max_pairs_per_range = ov['max_pairs_per_range']
    cfg._mode = mode
    # Namespace the results dir by mode so nothing overwrites
    cfg.results_root = f"{cfg.results_root.rstrip('/')}/{mode}"
    return cfg


def make_config(model: str, mode: str, scratch: str, repo_root: str) -> ExperimentConfig:
    """Factory for `model` (§5) with `mode` overrides applied (§11)."""
    if model not in CONFIG_FACTORIES:
        raise ValueError(f"unknown model: {model}")
    return apply_mode(CONFIG_FACTORIES[model](scratch, repo_root), mode)


def write_config_if_missing(cfg: ExperimentConfig, repo_root: str) -> str:
    """Write <repo_root>/experiments/<experiment>/config_<mode>.json once (§10.1)."""
    path = os.path.join(repo_root, 'experiments', cfg.experiment_name, f'config_{cfg._mode}.json')
    if not os.path.exists(path):
        cfg.to_json(path)
    return path
