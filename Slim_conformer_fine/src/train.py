"""train_pair() — within-subject training + testing loop for Slim Conformer B at 50 Hz.

Identical to Conformer_decimated_CWT/src/train.py minus the CWT front-end: same CV
splits, normalization, augmentation, seeds, epochs, checkpoint rule and result files,
so the slim model's results pair subject-by-subject with that project's Conformer A
and Conformer B.
"""
from src.determinism import seed_worker   # sets CUBLAS/PYTHONHASHSEED env before torch loads

import copy
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

from src.config import ExperimentConfig, PIPELINE_VERSION
from src.data import (load_pair, slice_time_windows,
                      z_score_normalize, add_gaussian_noise_augmentation, EEGDataset)
from src.models.conformer_b_slim import SlimConformerB
from src.utils import init_weights, is_pair_done, write_pair_results

warnings.filterwarnings('ignore')


def pair_names(cfg: ExperimentConfig, class_a: int, class_b: int):
    """(PAIR_NAME, PAIR_SLUG), e.g. ('WAA/SAA', 'WAA_SAA')."""
    pair_name = f"{cfg.joint_names[class_a]}/{cfg.joint_names[class_b]}"   # [P6]
    return pair_name, pair_name.replace("/", "_")


def build_model(cfg: ExperimentConfig, n_channels: int, n_timepoints: int, n_classes: int = 2):
    """Slim Conformer B with its hyperparameter dict plus the window/sampling info it
    needs to size the positional embedding."""
    if cfg.model != 'conformer_b_slim':
        raise ValueError(f"unknown model: {cfg.model}")
    model_cfg = dict(cfg.conformer_b_slim,
                     test_time_windows_ms=tuple(cfg.test_time_windows_ms),
                     sampling_rate=cfg.sampling_rate)
    return SlimConformerB(n_channels=n_channels, n_timepoints=n_timepoints,
                          n_classes=n_classes, embedding_dim=None, cfg=model_cfg)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(eeg)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)

            outputs = model(eeg)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(val_loader), 100.0 * correct / total, all_preds, all_labels


def make_loaders(cfg: ExperimentConfig, X_tr, y_tr, X_va, y_va, X_te, y_te, loader_seed: int):
    """Train (shuffled, seeded) / val / test loaders, as in the notebook."""
    train_generator = torch.Generator()
    train_generator.manual_seed(int(loader_seed))
    # [P4] worker_init_fn added to all three loaders
    train_loader = DataLoader(EEGDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, generator=train_generator,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(EEGDataset(X_va, y_va), batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, worker_init_fn=seed_worker)
    test_loader = DataLoader(EEGDataset(X_te, y_te), batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, worker_init_fn=seed_worker)
    return train_loader, val_loader, test_loader


def train_pair(class_a: int, class_b: int, cfg: ExperimentConfig, overwrite: bool = False) -> str:
    """Train + test cfg.model on one pair; returns the pair's results dir."""
    class_a, class_b = int(class_a), int(class_b)
    seed = int(cfg.seed)                                          # [P5]
    PAIR_NAME, PAIR_SLUG = pair_names(cfg, class_a, class_b)
    PAIR_DIR = os.path.join(cfg.results_root, PAIR_SLUG)

    if is_pair_done(PAIR_DIR) and not overwrite:
        print(f"SKIP {PAIR_NAME} [{cfg.model}]: results.json already exists in {PAIR_DIR} "
              f"(use --overwrite to re-run)", flush=True)
        return PAIR_DIR

    if cfg.n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 (StratifiedKFold), got {cfg.n_folds}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    print(f'\nUsing device: {device} | gpu={GPU} | torch={torch.__version__}', flush=True)
    print('NOTE: determinism holds per-GPU-model. Record this GPU name.', flush=True)
    if device.type == 'cpu':
        print('WARNING: CUDA not available - training on CPU will be slow.', flush=True)

    TEST_TIME_WINDOWS_MS = list(cfg.test_time_windows_ms)
    TEST_TIME_WINDOWS_SAMPLES = cfg.window_samples()

    print(f"\nPAIR: {PAIR_NAME}  (classes {class_a} vs {class_b})  "
          f"model={cfg.model}  mode={cfg._mode}  band={cfg.band_tag}  "
          f"decim={cfg.decim} ({cfg.orig_sampling_rate} -> {cfg.sampling_rate} Hz)", flush=True)
    print(f"Test time windows: {TEST_TIME_WINDOWS_MS} ms", flush=True)
    print(f"Test time windows (samples @ {cfg.sampling_rate} Hz): {TEST_TIME_WINDOWS_SAMPLES}",
          flush=True)

    # =======================================================================
    # LOAD DATA (DECIMATED) + BINARY FILTER + TIME WINDOWS
    # =======================================================================
    print(f"\nBINARY CLASSIFICATION: Class {class_a} vs Class {class_b}  ({PAIR_NAME})",
          flush=True)
    X_full, y_full, subject_ids = load_pair(cfg.dataset_root, cfg.band_tag, cfg.decim,
                                            class_a, class_b)
    print(f"  Class 0 (original {class_a}): {np.sum(y_full == 0)} trials", flush=True)
    print(f"  Class 1 (original {class_b}): {np.sum(y_full == 1)} trials", flush=True)

    full_len = X_full.shape[2]
    TEST_TIME_WINDOWS_SAMPLES_FINAL = [min(s, full_len) for s in TEST_TIME_WINDOWS_SAMPLES]
    TEST_TIME_WINDOWS_MS_FINAL = list(TEST_TIME_WINDOWS_MS)
    if TEST_TIME_WINDOWS_SAMPLES_FINAL != TEST_TIME_WINDOWS_SAMPLES:
        print(f"WARNING: epochs have only {full_len} samples; windows clipped to "
              f"{TEST_TIME_WINDOWS_SAMPLES_FINAL}", flush=True)

    windows_by_samples = slice_time_windows(X_full, TEST_TIME_WINDOWS_SAMPLES_FINAL)
    time_windows_data = {}
    print("\nExtracted time windows:", flush=True)
    for t_ms, t_samp in zip(TEST_TIME_WINDOWS_MS_FINAL, TEST_TIME_WINDOWS_SAMPLES_FINAL):
        time_windows_data[t_ms] = windows_by_samples[t_samp]
        print(f"  {t_ms}ms ({t_samp} samples): shape {time_windows_data[t_ms].shape}",
              flush=True)

    n_channels = X_full.shape[1]
    unique_subjects = [int(s) for s in np.unique(subject_ids)]     # [P5] python ints
    if cfg._max_subjects is not None:                              # mode hook
        unique_subjects = unique_subjects[:cfg._max_subjects]

    subject_data = {}
    for subj_id in unique_subjects:
        subj_mask = (subject_ids == subj_id)
        subject_data[subj_id] = {'y': y_full[subj_mask],
                                 'n_trials': int(np.sum(subj_mask))}
        for t_ms in TEST_TIME_WINDOWS_MS_FINAL:
            subject_data[subj_id][f'X_{t_ms}ms'] = time_windows_data[t_ms][subj_mask]

    n_epochs = cfg.n_epochs                                        # mode hook
    n_classes = 2
    log_epochs = cfg._mode != 'full'   # per-epoch curves in smoke; per-fold only in full

    n_params = count_params(build_model(cfg, n_channels, TEST_TIME_WINDOWS_SAMPLES_FINAL[0], n_classes))
    print(f"\nModel configuration:", flush=True)
    print(f"  Model: {cfg.model} ({n_params:,} parameters)", flush=True)
    print(f"  Channels: {n_channels}", flush=True)
    print(f"  Subjects: {unique_subjects}", flush=True)
    print(f"  Time windows: {TEST_TIME_WINDOWS_MS_FINAL} ms", flush=True)
    print(f"  Classes: {n_classes} (Binary: Class {class_a} vs Class {class_b})", flush=True)
    print(f"  Folds: {cfg.n_folds}", flush=True)
    print(f"  Batch size: {cfg.batch_size}", flush=True)
    print(f"  Epochs per subject per time window: {n_epochs}", flush=True)

    # =======================================================================
    # WITHIN-SUBJECT TRAINING + TESTING  (structure verbatim)
    # =======================================================================
    all_subject_results = []
    t_pair = time.time()

    for subj_idx, subj_id in enumerate(unique_subjects):
        subj_id = int(subj_id)                                    # [P5]
        t_subj = time.time()
        X_subj_time_windows = {t: subject_data[subj_id][f'X_{t}ms']
                               for t in TEST_TIME_WINDOWS_MS_FINAL}
        y_subj = subject_data[subj_id]['y']

        if len(np.unique(y_subj)) < 2 or len(y_subj) < 10:
            print(f"Skipping Subject {subj_id}: insufficient data", flush=True)
            continue

        n_folds = cfg.n_folds                                     # mode hook
        subj_indices = np.arange(len(y_subj))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=int(42 + subj_id))     # [P5]
        # same splits reused across every window -> windows are truly paired
        cv_splits = list(skf.split(subj_indices, y_subj))

        subject_time_window_results = {}

        for tw_idx, t_ms in enumerate(TEST_TIME_WINDOWS_MS_FINAL):
            t_samples = TEST_TIME_WINDOWS_SAMPLES_FINAL[tw_idx]
            X_subj_tw = X_subj_time_windows[t_ms]
            fold_results_tw = []

            for fold_idx, (train_val_indices, test_indices) in enumerate(cv_splits):
                train_indices, val_indices = train_test_split(
                    train_val_indices, test_size=cfg.val_size_of_train,
                    random_state=int(42 + subj_id),                # [P5]
                    stratify=y_subj[train_val_indices])

                X_train, y_train = X_subj_tw[train_indices], y_subj[train_indices]
                X_val, y_val = X_subj_tw[val_indices], y_subj[val_indices]
                X_test, y_test = X_subj_tw[test_indices], y_subj[test_indices]

                X_train_norm, X_val_norm, X_test_norm = z_score_normalize(X_train, X_val, X_test)

                aug_seed = int(seed + int(subj_id) + int(fold_idx) + int(tw_idx) * 100)
                X_train_augmented = add_gaussian_noise_augmentation(
                    X_train_norm, noise_level=cfg.noise_level, random_seed=aug_seed)
                X_train_final = np.concatenate([X_train_norm, X_train_augmented], axis=0)
                y_train_final = np.concatenate([y_train, y_train], axis=0)

                loader_seed = int(seed + int(subj_id) * 1000 + int(fold_idx) * 100 + int(tw_idx))
                train_loader, val_loader, test_loader = make_loaders(
                    cfg, X_train_final, y_train_final, X_val_norm, y_val, X_test_norm, y_test,
                    loader_seed)

                model_init_seed = loader_seed
                torch.manual_seed(model_init_seed)                 # [P5] int cast
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(model_init_seed)

                model = build_model(cfg, n_channels=n_channels, n_timepoints=t_samples,
                                    n_classes=n_classes).to(device)
                model.apply(init_weights)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5)

                best_val_acc = 0.0
                best_model_state = None
                best_epoch = 0

                for epoch in range(n_epochs):
                    train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                                        optimizer, device)
                    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
                    scheduler.step(val_loss)

                    # ORIGINAL selection rule. Deterministic once P1-P4 are in place.
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        # deepcopy: the notebook's state_dict().copy() was shallow
                        best_model_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch + 1

                    if log_epochs:
                        print(f"    S{subj_id} {t_ms}ms fold{fold_idx + 1} "
                              f"ep{epoch + 1:>3}/{n_epochs}: "
                              f"train_loss={train_loss:.4f} train_acc={train_acc:6.2f}% "
                              f"val_loss={val_loss:.4f} val_acc={val_acc:6.2f}% "
                              f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

                if best_model_state is not None:
                    model.load_state_dict(best_model_state)

                test_loss, test_acc, test_preds, test_labels = validate(
                    model, test_loader, criterion, device)
                test_preds = np.array(test_preds)
                test_labels = np.array(test_labels)

                print(f"  S{subj_id} {t_ms}ms fold{fold_idx + 1}/{len(cv_splits)}: "
                      f"best_val={best_val_acc:6.2f}% (ep{best_epoch}) "
                      f"test={test_acc:6.2f}%", flush=True)

                fold_results_tw.append({
                    'fold': fold_idx + 1,
                    'test_accuracy': test_acc,
                    'test_loss': test_loss,
                    'test_preds': test_preds,
                    'test_labels': test_labels,
                    'n_test_samples': len(test_labels),
                    'val_accuracy': best_val_acc,
                    'best_epoch': best_epoch,
                })

                del model, optimizer, scheduler, best_model_state
                del train_loader, val_loader, test_loader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_test_preds_tw = np.concatenate([r['test_preds'] for r in fold_results_tw])
            all_test_labels_tw = np.concatenate([r['test_labels'] for r in fold_results_tw])

            subject_time_window_results[t_ms] = {
                'test_accuracy': accuracy_score(all_test_labels_tw, all_test_preds_tw) * 100,
                'test_preds': all_test_preds_tw,
                'test_labels': all_test_labels_tw,
                'confusion_matrix': confusion_matrix(all_test_labels_tw, all_test_preds_tw),
                'n_test_samples': len(all_test_labels_tw),
                'fold_results': fold_results_tw,
            }

        all_subject_results.append({'subject_id': subj_id, 'n_folds': n_folds,
                                    'time_window_results': subject_time_window_results})

        print(f"S{subj_id:>2}: " + "  ".join(
            f"{t}ms={subject_time_window_results[t]['test_accuracy']:6.2f}%"
            for t in TEST_TIME_WINDOWS_MS_FINAL)
            + f"   ({time.time() - t_subj:.0f}s)", flush=True)

    print(f"\nALL SUBJECTS COMPLETED - {PAIR_NAME} [{cfg.model}]  "
          f"({time.time() - t_pair:.0f}s)", flush=True)

    if not all_subject_results:
        raise RuntimeError(f"{PAIR_NAME} [{cfg.model}]: every subject was skipped")

    # =======================================================================
    # [P7] SAVE PER-SUBJECT + PER-FOLD RESULTS
    # =======================================================================
    meta = dict(seed=seed, gpu=GPU, torch=torch.__version__,
                model=cfg.model, mode=cfg._mode, band=cfg.band_tag,
                dataset_root=cfg.dataset_root,
                pipeline_version=PIPELINE_VERSION,
                decim=cfg.decim, orig_sampling_rate=cfg.orig_sampling_rate,
                sampling_rate=cfg.sampling_rate,
                window_samples=list(TEST_TIME_WINDOWS_SAMPLES_FINAL),
                n_folds=cfg.n_folds, n_epochs=cfg.n_epochs,
                n_params=n_params,
                model_cfg=dict(cfg.conformer_b_slim),
                seconds=round(time.time() - t_pair, 1))
    wide = write_pair_results(PAIR_DIR, PAIR_NAME, class_a, class_b,
                              TEST_TIME_WINDOWS_MS_FINAL, all_subject_results, meta)

    print(f"\n{'=' * 68}", flush=True)
    print(f"SAVED {PAIR_NAME} [{cfg.model}] -> {PAIR_DIR}", flush=True)
    print(f"gpu={GPU} | torch={torch.__version__} | seed={seed} | "
          f"model={cfg.model} | mode={cfg._mode} | decim={cfg.decim}", flush=True)
    print(f"{'=' * 68}", flush=True)
    print(wide.round(2).to_string(), flush=True)
    print("\nmean:", wide.mean().round(2).to_dict(), flush=True)
    if len(wide) > 1:
        print("sd  :", wide.std(ddof=1).round(2).to_dict(), flush=True)
    if wide.isna().any().any():
        print("WARNING: missing subject/window cells - a subject was skipped.", flush=True)

    return PAIR_DIR
