"""Data loading, binary filtering, time windowing, normalization, augmentation.

Ported from FineMI-enhancements/EMBC_deterministic-3.ipynb (via ConformerEEG/src/data.py).
Only change: load_pair() applies the binary class filter per subject and then
concatenates, instead of concatenating all 18 subjects first. Row order (subjects
ascending, trials in file order) and dtype are unchanged, so the resulting
arrays are identical to the notebook's.
"""
import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset


def get_subject_number(path):
    basename = os.path.basename(path)
    match = re.search(r"subject(\d+)", basename, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def discover_subject_files(dataset_root: str, band_tag: str) -> list[str]:
    """subject{N}_eeg_epochs_<band_tag>_*.npz under dataset_root, sorted by subject number."""
    pattern = os.path.join(dataset_root, f"subject*_eeg_epochs_{band_tag}_*.npz")
    files = sorted(glob.glob(pattern), key=get_subject_number)
    if not files:
        raise FileNotFoundError(
            f"No subject{{N}}_eeg_epochs_{band_tag}_*.npz files under {dataset_root!r} "
            f"(pass --dataset-root to point elsewhere)")
    sids = [get_subject_number(f) for f in files]
    dupes = sorted({s for s in sids if sids.count(s) > 1})
    if dupes:
        raise RuntimeError(f"multiple {band_tag} files for subjects {dupes} in {dataset_root!r}")
    return files


# (dataset_root, band_tag) -> [(sid, data, labels), ...] for every subject. The .npz
# files are compressed, so decompressing all 18 for every pair of a 28-pair sweep
# is slow; the raw arrays (~3.6 GB float64) are kept per process instead.
_RAW_CACHE = {}

# (dataset_root, band_tag, class_a, class_b) -> (X, y, subject_ids). Holds only the
# most recent pair, so running several models on one pair filters the data once.
# Callers must not modify the returned arrays in place.
_PAIR_CACHE = {}


def load_subjects(dataset_root: str, band_tag: str):
    """[(subject_id, data, labels), ...] for every subject file, sorted by subject number."""
    key = (os.path.abspath(dataset_root), band_tag)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]

    files = discover_subject_files(dataset_root, band_tag)
    print(f"\nFound {len(files)} subject .npz files ({band_tag}) in {dataset_root}", flush=True)
    subjects = []
    for path in files:
        with np.load(path, allow_pickle=True) as z:
            subjects.append((get_subject_number(path), z["data"], z["labels"]))

    _RAW_CACHE[key] = subjects
    return subjects


def load_pair(dataset_root: str, band_tag: str, class_a: int, class_b: int):
    """Binary-filtered (X, y, subject_ids) for class_a vs class_b across all subjects.

    Labels are remapped class_a -> 0, class_b -> 1 (as in the notebook).
    """
    key = (os.path.abspath(dataset_root), band_tag, int(class_a), int(class_b))
    if key in _PAIR_CACHE:
        return _PAIR_CACHE[key]
    _PAIR_CACHE.clear()

    X_parts, y_parts, sid_parts = [], [], []
    for sid, data, labels in load_subjects(dataset_root, band_tag):
        mask = (labels == class_a) | (labels == class_b)
        X_parts.append(data[mask])
        y_parts.append(labels[mask])
        sid_parts.append(np.full(int(mask.sum()), sid))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    subject_ids = np.concatenate(sid_parts, axis=0)
    y = np.where(y == class_b, 1, 0)

    _PAIR_CACHE[key] = (X, y, subject_ids)
    return X, y, subject_ids


def slice_time_windows(X, windows_samples: list[int]) -> dict[int, np.ndarray]:
    """First-`s`-samples crop of X for each window length `s`, keyed by `s`.

    Windows longer than the epoch are clipped to the epoch length, as in the
    notebook's TEST_TIME_WINDOWS_SAMPLES_FINAL.
    """
    full_len = X.shape[2]
    return {s: X[:, :, :min(s, full_len)] for s in windows_samples}


def z_score_normalize(X_train, X_val, X_test):
    """Per-channel z-score normalization using training data statistics only."""
    channel_means = np.mean(X_train, axis=(0, 2))
    channel_stds = np.std(X_train, axis=(0, 2))
    channel_stds = np.where(channel_stds == 0, 1.0, channel_stds)

    channel_means = channel_means[np.newaxis, :, np.newaxis]
    channel_stds = channel_stds[np.newaxis, :, np.newaxis]

    X_train_norm = (X_train - channel_means) / channel_stds
    X_val_norm = (X_val - channel_means) / channel_stds
    X_test_norm = (X_test - channel_means) / channel_stds

    return X_train_norm, X_val_norm, X_test_norm


def add_gaussian_noise_augmentation(X, noise_level=0.1, random_seed=None):
    """Add Gaussian noise to EEG data for augmentation.  DA = DR + NL*GN"""
    if random_seed is not None:
        np.random.seed(int(random_seed))            # [P5] int cast

    channel_stds = np.std(X, axis=(0, 2), keepdims=True)
    noise = np.random.normal(loc=0.0, scale=noise_level * channel_stds, size=X.shape)
    return X + noise


class EEGDataset(Dataset):
    """Dataset for EEG data."""

    def __init__(self, eeg_data, labels):
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'eeg': self.eeg_data[idx], 'label': self.labels[idx]}
