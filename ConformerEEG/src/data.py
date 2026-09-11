"""Data loading, binary filtering, time windowing, normalization, augmentation.

Ported from FineMI-enhancements/EMBC_deterministic-3.ipynb.
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


# dataset_root -> (X_full, y_full, subject_ids). Lets run_pairs_range reuse one
# load across pairs. Callers must not modify the returned arrays in place
# (filter_binary uses boolean indexing, which copies).
_LOAD_CACHE = {}


def load_all_subjects(dataset_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_full, y_full, subject_ids) for every subject found."""
    if dataset_root in _LOAD_CACHE:
        return _LOAD_CACHE[dataset_root]

    all_data, all_labels, subject_ids_list = [], [], []
    subject_folders = sorted(
        [f for f in glob.glob(os.path.join(dataset_root, "subject*")) if os.path.isdir(f)],
        key=get_subject_number)
    print(f"\nFound {len(subject_folders)} subject folders", flush=True)

    if not subject_folders:
        # Not in the notebook: layouts without subject* folders (only the .npz
        # files). Discover subjects from the .npz filenames instead, one entry per
        # subject number, same sort key.
        first_per_subject = {}
        for f in sorted(glob.glob(os.path.join(dataset_root, "subject*_eeg_epochs_*.npz"))):
            first_per_subject.setdefault(get_subject_number(f), f)
        subject_folders = sorted(first_per_subject.values(), key=get_subject_number)
        print(f"No subject folders; found {len(subject_folders)} subject .npz files",
              flush=True)

    for subject_path in subject_folders:
        sid = get_subject_number(subject_path)
        eeg_candidates = glob.glob(os.path.join(dataset_root, f"subject{sid}_eeg_epochs_*.npz"))
        if not eeg_candidates:
            print(f"  Skipping subject{sid}: EEG .npz not found", flush=True)
            continue
        z = np.load(eeg_candidates[0], allow_pickle=True)
        all_data.append(z["data"])
        all_labels.append(z["labels"])
        subject_ids_list.extend([sid] * len(z["labels"]))

    if not all_data:
        raise FileNotFoundError(
            f"No subject{{N}}_eeg_epochs_*.npz files found under {dataset_root!r} "
            f"(set FINEMI_DATASET_ROOT to override)")

    X_full = np.concatenate(all_data, axis=0)
    y_full = np.concatenate(all_labels, axis=0)
    subject_ids = np.array(subject_ids_list)

    _LOAD_CACHE[dataset_root] = (X_full, y_full, subject_ids)
    return X_full, y_full, subject_ids


def filter_binary(X, y, subject_ids, class_a, class_b):
    """Keep only class_a / class_b trials; remap labels class_a -> 0, class_b -> 1."""
    binary_mask = (y == class_a) | (y == class_b)
    X = X[binary_mask]
    y = y[binary_mask]
    subject_ids = subject_ids[binary_mask]
    y = np.where(y == class_b, 1, 0)
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
