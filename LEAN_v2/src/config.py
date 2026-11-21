"""Experiment config for LEAN_v2: LEAN (ICASSP paper) rebuilt step by step with ideas from
EEGNet, EEG-TCNet, LMDA-Net and MBCL, on the fixation-cross-baseline, imagery-only
0.5-3 Hz FineMI data at 50 Hz (D = 5), the same data and training loop as LEAN, EEG
Conformer and EEGNet, so every result pairs with theirs subject by subject.

Every model here is a PRESET: an architecture dict (`arch`) plus a training dict
(`train`). One configurable network (src/models/lean_v2.py) covers every LEAN variant;
with LEAN_ARCH it rebuilds LEAN layer for layer (25,752 parameters, same construction
order, so the same initial weights for the same seed).

Preset groups (pass a group name or model names to --models / MODELS=):
  ladder     R0 ... R7, each rung = the previous rung + one change (cumulative)
  loo        leave-one-out: the final model (R7) minus one change at a time
  training   LEAN's architecture with only training changes
  baselines  EEGNet-8,2 at 50 Hz, with its original recipe and with R7's recipe
  all        everything above

Modes:
  smoke    2 subjects, 2 folds, 2 epochs, all 4 windows - proves the code runs
  screen   all subjects, 5 folds, the preset's epochs, 800 and 4000 ms only, 6 pairs
           (SCREEN_PAIRS: 2 distant, 2 adjacent, 2 same-joint) - cheap ladder screening
  full     all subjects, 5 folds, all 4 windows, all 28 pairs - paper numbers
"""
from __future__ import annotations   # `int | None` field annotations on Python < 3.10

from dataclasses import dataclass, field, asdict
from typing import Tuple
from copy import deepcopy
from itertools import combinations
import hashlib
import json
import os

# ---- Band / data ------------------------------------------------------------------
BAND_TAG = '0.5_3hz'          # only subject{N}_eeg_epochs_<BAND_TAG>_*.npz are loaded
BAND_LABEL = '0.5-3 Hz'
BAND_HZ = (0.5, 3.0)          # the preflight checks that most EEG power lies in here
DATASET_DIRNAME = 'fixationcrossbaseline_0.5_3hz_MIonly'
DECIM = 5                     # 250 Hz -> 50 Hz for every model in this folder

# Earlier full runs on the same data that summarize.py compares against, paired over
# subjects. Paths are relative to this folder; screen and full mode both read the
# references' FULL results (restricted to the screened pairs and windows). Smoke mode
# skips them.
REFERENCES = {
    'LEAN (paper run)': os.path.join('..', 'Fine Conformer Fixation Cross Fix', 'results',
                                     'full', 'conformer_b_slim'),
    'EEGNet 50 Hz (paper run)': os.path.join('..', 'EEGNet-experiments', 'results', 'full',
                                             'eegnet_50'),
}
# -----------------------------------------------------------------------------------

# Bump whenever a change alters the numbers a run produces. It is stamped into
# results.json together with the preset's hash; a results folder with another stamp
# is treated as not done and re-trained.
PIPELINE_VERSION = 1

ORIG_SAMPLING_RATE = 250
EXPECTED_EPOCH_SAMPLES = 1000    # 0 to 4 s after imagery onset at 250 Hz

PAIRS = list(combinations(range(8), 2))   # 28 pairs, deterministic order
JOINT_NAMES = ('HOC', 'WFE', 'WAA', 'EPS', 'EFE', 'SPS', 'SAA', 'SFE')

DEFAULT_PAIRS = ('WAA_SAA', 'HOC_WFE', 'EPS_SPS')          # smoke
# screen: 2 distant, 2 adjacent, 2 same-joint pairs
SCREEN_PAIRS = ('HOC_SPS', 'WAA_SFE', 'HOC_WFE', 'EPS_SAA', 'EPS_EFE', 'SPS_SAA')

# =====================================================================================
# Architectures
# =====================================================================================
# LEAN exactly as in the ICASSP paper / 'Fine Conformer Fixation Cross Fix'
# (conformer_b_slim). Keys that only matter for other settings are listed too, so
# every preset has the same keys.
LEAN_ARCH = dict(
    arch='lean_v2',
    # front end
    temporal='per_electrode',      # 'per_electrode' (LEAN) | 'shared' (EEGNet-style)
    mtc_kernels=(7, 15, 31),       # samples: 140 / 300 / 620 ms at 50 Hz (both fronts)
    mtc_filters=16,                # per_electrode: 1x1 mix to 16 per branch
    shared_filters=4,              # shared: temporal filters per branch (F1)
    depth_mult=2,                  # shared: spatial filters per temporal filter (D)
    depth_attention=False,         # LMDA depth attention (needs temporal='shared')
    da_kernel=7,
    # fusion + pooling into tokens
    fusion='lean',                 # 'lean': conv k3 + max pool 3 | 'separable': EEGNet block 2
    width=32,                      # channels from the fusion layer on (d_model)
    lean_fusion_kernel=3, lean_pool=3,
    sep_kernel=8, pool1=2, pool2=4,          # separable: avg pool 2, sep. conv, avg pool 4
    # temporal context
    context='transformer',         # 'transformer' (LEAN) | 'tcn' (EEG-TCNet)
    n_heads=4, d_ff=64, p_att=0.3,
    tcn_blocks=3, tcn_kernel=4, p_tcn=0.3,  # dilations 1, 2, 4 -> 43-token receptive field
    # feature refinement (LEAN Sec. 2.5): depthwise-separable conv k5 + channel gate
    refine_kernel=5, eca_reduction=4,
    # readout
    readout='gap_mlp',             # 'gap_mlp' (LEAN) | 'last_step' | 'flatten'
    embedding_dim=32, p_proj=0.3, classifier_hidden=32, p_head=0.4,
    # regularisation / activation
    act='relu',                    # conv-block activation: 'relu' | 'elu'
    max_norm=False,                # max-norm on the spatial filters and the linear readout
    max_norm_spatial=1.0, max_norm_linear=0.25,
    p_conv=0.0,                    # dropout after each pooling step (0 = no dropout layer)
)

# EEGNet-8,2 at 50 Hz, identical to EEGNet-experiments' eegnet_50.
EEGNET50_ARCH = dict(
    arch='eegnet', F1=8, D=2, F2=16, drop_prob=0.5, bn_momentum=0.1, bn_eps=1e-3,
    max_norm_depthwise=1.0, max_norm_linear=0.25,
    kern_length=25, pool1=2, sep_kern_length=8, pool2=4)

# =====================================================================================
# Training recipes
# =====================================================================================
# LEAN's recipe (the loop every earlier run used): 50 epochs, ReduceLROnPlateau on the
# validation loss, keep the checkpoint with the best validation accuracy.
LEAN_TRAIN = dict(
    n_epochs=50,
    schedule='best_val',      # 'best_val' (LEAN) | 'fixed_avg': cosine LR, average the
    avg_last=10,              #   weights of the last `avg_last` epochs, no selection
    supcon_weight=0.0,        # loss = CE + w * SupCon (MBCL / CLMT-Net); 0 = CE only
    supcon_tau=0.05, supcon_dim=16,
    n_crops=0,                # 0 = whole window; k = train on random crops of crop_frac
    crop_frac=0.9,            #   of the window, test on the mean logits of k crops
)


def _w(base: dict, **changes) -> dict:
    out = deepcopy(base)
    unknown = [k for k in changes if k not in out]
    if unknown:
        raise KeyError(f"unknown preset keys {unknown}")
    out.update(changes)
    return out


# ---- Ladder (cumulative) -----------------------------------------------------------
_A0 = LEAN_ARCH
_A1 = _w(_A0, fusion='separable')                                   # EEGNet block 2, avg pools
_A2 = _w(_A1, context='tcn', readout='last_step', width=16)        # EEG-TCNet context + readout
_A3 = _w(_A2, temporal='shared')                                    # shared temporal + depthwise spatial
_A4 = _w(_A3, depth_attention=True)                                 # LMDA depth attention
_A5 = _w(_A4, act='elu', max_norm=True, p_conv=0.25)               # EEGNet/EEG-TCNet regularisation
_T5 = _w(LEAN_TRAIN, schedule='fixed_avg')                          # no checkpoint selection
_T6 = _w(_T5, supcon_weight=0.5)                                    # + supervised contrastive
_T7 = _w(_T6, n_crops=3)                                            # + crops

PRESETS = {
    'lean_v2_r0': dict(label='R0 LEAN', arch=_A0, train=LEAN_TRAIN),
    'lean_v2_r1': dict(label='R1 +separable fusion, avg pools', arch=_A1, train=LEAN_TRAIN),
    'lean_v2_r2': dict(label='R2 +TCN, last-step readout', arch=_A2, train=LEAN_TRAIN),
    'lean_v2_r3': dict(label='R3 +shared temporal filters', arch=_A3, train=LEAN_TRAIN),
    'lean_v2_r4': dict(label='R4 +depth attention', arch=_A4, train=LEAN_TRAIN),
    'lean_v2_r5': dict(label='R5 +ELU/max-norm/dropout, fixed schedule', arch=_A5, train=_T5),
    'lean_v2_r6': dict(label='R6 +SupCon', arch=_A5, train=_T6),
    'lean_v2_r7': dict(label='R7 +crops (final)', arch=_A5, train=_T7),
}
LADDER = tuple(PRESETS)
FINAL = 'lean_v2_r7'

# ---- Leave-one-out from the final model --------------------------------------------
# (R7 minus crops is R6, so it is not repeated.) Removing the shared temporal filters
# also removes depth attention, which needs them.
LOO = {
    'lean_v2_loo_fusion': ('R7 - separable fusion', _w(_A5, fusion='lean'), _T7),
    'lean_v2_loo_tcn': ('R7 - TCN (transformer)', _w(_A5, context='transformer'), _T7),
    'lean_v2_loo_readout': ('R7 - last-step readout (GAP+MLP)', _w(_A5, readout='gap_mlp'), _T7),
    'lean_v2_loo_shared': ('R7 - shared temporal (and depth att.)',
                           _w(_A5, temporal='per_electrode', depth_attention=False), _T7),
    'lean_v2_loo_depthatt': ('R7 - depth attention', _w(_A5, depth_attention=False), _T7),
    'lean_v2_loo_reg': ('R7 - ELU/max-norm/dropout',
                        _w(_A5, act='relu', max_norm=False, p_conv=0.0), _T7),
    'lean_v2_loo_schedule': ('R7 - fixed schedule (best-val)', _A5, _w(_T7, schedule='best_val')),
    'lean_v2_loo_supcon': ('R7 - SupCon', _A5, _w(_T7, supcon_weight=0.0)),
}
for _name, (_label, _arch, _train) in LOO.items():
    PRESETS[_name] = dict(label=_label, arch=_arch, train=_train)

# ---- Training changes only, on LEAN's architecture ---------------------------------
TRAINING_ONLY = {
    'lean_schedule': ('LEAN + fixed schedule', _w(LEAN_TRAIN, schedule='fixed_avg')),
    'lean_supcon': ('LEAN + SupCon', _w(LEAN_TRAIN, supcon_weight=0.5)),
    'lean_recipe': ('LEAN + R7 recipe', _T7),
}
for _name, (_label, _train) in TRAINING_ONLY.items():
    PRESETS[_name] = dict(label=_label, arch=LEAN_ARCH, train=_train)

# ---- EEGNet baselines --------------------------------------------------------------
PRESETS['eegnet_50'] = dict(label='EEGNet 50 Hz', arch=EEGNET50_ARCH, train=LEAN_TRAIN)
PRESETS['eegnet_50_recipe'] = dict(label='EEGNet 50 Hz + R7 recipe', arch=EEGNET50_ARCH,
                                   train=_T7)

MODELS = tuple(PRESETS)
MODEL_LABELS = {m: p['label'] for m, p in PRESETS.items()}
MODEL_GROUPS = {
    'ladder': list(LADDER),
    'loo': list(LOO),
    'training': list(TRAINING_ONLY),
    'baselines': ['eegnet_50', 'eegnet_50_recipe'],
    'all': list(MODELS),
}


def expand_models(tokens) -> list[str]:
    """Model names and/or group names -> de-duplicated model names, in order."""
    out = []
    for t in tokens:
        for name in t.split():
            names = MODEL_GROUPS.get(name, [name])
            for n in names:
                if n not in PRESETS:
                    raise ValueError(f"unknown model or group {n!r} (groups: "
                                     f"{', '.join(MODEL_GROUPS)}; models: {', '.join(MODELS)})")
                if n not in out:
                    out.append(n)
    return out


def preset_hash(model: str) -> str:
    """Short hash of a preset's arch + train dicts, stamped into results.json so a
    changed preset is re-trained instead of silently reusing old results."""
    p = PRESETS[model]
    blob = json.dumps(dict(arch=p['arch'], train=p['train']), sort_keys=True, default=list)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def validate_preset(model: str):
    a, t = PRESETS[model]['arch'], PRESETS[model]['train']
    if a['arch'] == 'lean_v2':
        if a['depth_attention'] and a['temporal'] != 'shared':
            raise ValueError(f"{model}: depth attention needs temporal='shared'")
        if a['context'] == 'transformer' and a['width'] % a['n_heads']:
            raise ValueError(f"{model}: width {a['width']} not divisible by {a['n_heads']} heads")
        for k, allowed in (('temporal', ('per_electrode', 'shared')),
                           ('fusion', ('lean', 'separable')),
                           ('context', ('transformer', 'tcn')),
                           ('readout', ('gap_mlp', 'last_step', 'flatten')),
                           ('act', ('relu', 'elu'))):
            if a[k] not in allowed:
                raise ValueError(f"{model}: {k}={a[k]!r}, expected one of {allowed}")
    elif a['arch'] != 'eegnet':
        raise ValueError(f"{model}: unknown arch {a['arch']!r}")
    if t['schedule'] not in ('best_val', 'fixed_avg'):
        raise ValueError(f"{model}: unknown schedule {t['schedule']!r}")
    if t['n_crops'] and not 0 < t['crop_frac'] < 1:
        raise ValueError(f"{model}: crop_frac must be in (0, 1)")


for _m in PRESETS:
    validate_preset(_m)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT_DEFAULT = os.path.normpath(
    os.path.join(PROJECT_ROOT, '..', '..', DATASET_DIRNAME))
RESULTS_ROOT_DEFAULT = os.path.join(PROJECT_ROOT, 'results')


@dataclass
class ExperimentConfig:
    model:              str
    dataset_root:       str
    band_tag:           str  = BAND_TAG
    orig_sampling_rate: int  = ORIG_SAMPLING_RATE
    decim:              int  = DECIM
    sampling_rate:      int  = field(init=False)
    n_channels:         int  = 62
    test_time_windows_ms: Tuple[int, ...] = (800, 1500, 3000, 4000)
    joint_names: Tuple[str, ...] = JOINT_NAMES

    # CV (identical to LEAN / EEG Conformer / EEGNet)
    n_folds:           int   = 5
    val_size_of_train: float = 0.25

    # Optimiser and augmentation (identical to LEAN); epochs, schedule, SupCon and
    # crops come from the preset's train dict
    batch_size:   int   = 32
    n_epochs:     int   = 50
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    noise_level:  float = 0.15
    seed:         int   = 42

    arch:  dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    preset_hash: str = ''

    results_root: str = ""
    _max_subjects: int | None = None
    _mode: str = 'full'

    def __post_init__(self):
        if self.decim < 1 or self.orig_sampling_rate % self.decim != 0:
            raise ValueError(f"bad decim={self.decim} for {self.orig_sampling_rate} Hz")
        self.sampling_rate = self.orig_sampling_rate // self.decim

    def window_samples(self) -> list[int]:
        """Window lengths in samples at 50 Hz (40 / 75 / 150 / 200)."""
        return [int(t * self.sampling_rate / 1000) for t in self.test_time_windows_ms]

    def model_samples(self, t_samples: int) -> int:
        """Input length the network is built for: the window, or the crop length."""
        if self.train.get('n_crops'):
            return int(t_samples * self.train['crop_frac'])
        return int(t_samples)

    def max_model_samples(self) -> int:
        return max(self.model_samples(s) for s in self.window_samples())

    def to_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.{os.getpid()}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp_path, path)


# n_epochs None = the preset's own n_epochs
MODE_OVERRIDES = {
    'smoke': dict(max_subjects=2, n_folds=2, n_epochs=2,
                  test_time_windows_ms=(800, 1500, 3000, 4000), pairs=DEFAULT_PAIRS),
    'screen': dict(max_subjects=None, n_folds=5, n_epochs=None,
                   test_time_windows_ms=(800, 4000), pairs=SCREEN_PAIRS),
    'full': dict(max_subjects=None, n_folds=5, n_epochs=None,
                 test_time_windows_ms=(800, 1500, 3000, 4000), pairs=('all',)),
}


def mode_root(results_root: str, mode: str) -> str:
    return os.path.join(results_root, mode)


def make_config(model: str, mode: str,
                dataset_root: str = DATASET_ROOT_DEFAULT,
                results_root: str = RESULTS_ROOT_DEFAULT) -> ExperimentConfig:
    """Config for preset `model` in `mode`; results under <results_root>/<mode>/<model>."""
    if model not in PRESETS:
        raise ValueError(f"unknown model: {model} (expected one of {MODELS})")
    if mode not in MODE_OVERRIDES:
        raise ValueError(f"unknown mode: {mode}")
    p = PRESETS[model]
    ov = MODE_OVERRIDES[mode]
    cfg = ExperimentConfig(model=model, dataset_root=dataset_root,
                           arch=deepcopy(p['arch']), train=deepcopy(p['train']),
                           preset_hash=preset_hash(model))
    cfg.n_folds = ov['n_folds']
    cfg.n_epochs = ov['n_epochs'] if ov['n_epochs'] is not None else p['train']['n_epochs']
    cfg.test_time_windows_ms = tuple(ov['test_time_windows_ms'])
    cfg._max_subjects = ov['max_subjects']
    cfg._mode = mode
    cfg.results_root = os.path.join(mode_root(results_root, mode), model)
    return cfg


def pair_slug(class_a: int, class_b: int, joint_names=JOINT_NAMES) -> str:
    return f"{joint_names[class_a]}_{joint_names[class_b]}"


def parse_pair(token: str, joint_names=JOINT_NAMES) -> tuple[int, int]:
    """Parse one --pairs entry into (class_a, class_b) with class_a < class_b.

    Accepted forms (case-insensitive): WAA_SAA, WAA/SAA, WAA-SAA, WAA,SAA; class ids
    2_6; a pair index 0-27. Reversed order is normalized to PAIRS order.
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
                             f"(joints: {', '.join(joint_names)})")
    a, b = sorted(classes)
    if a == b:
        raise ValueError(f"pair {token!r} uses the same class twice")
    return a, b


def parse_pairs(tokens) -> list[tuple[int, int]]:
    """--pairs entries -> de-duplicated list in the order given; 'all' = all 28."""
    pairs = []
    for token in tokens:
        expanded = PAIRS if token.strip().lower() == 'all' else [parse_pair(token)]
        for pair in expanded:
            if pair not in pairs:
                pairs.append(pair)
    return pairs
