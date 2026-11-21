"""train_pair() — within-subject training + testing loop for the LEAN_v2 presets.

EEGNet-experiments/src/train.py (= LEAN's loop) with three optional training changes,
switched by the preset's `train` dict (src/config.py):

  schedule='best_val'   LEAN: ReduceLROnPlateau on the validation loss, keep the
                        checkpoint with the best validation accuracy
  schedule='fixed_avg'  cosine LR over n_epochs, then load the AVERAGE of the weights
                        (and BN statistics) of the last `avg_last` epochs; the
                        validation split is still held out (so every preset trains on
                        exactly the same trials) but never used to pick a checkpoint
  supcon_weight > 0     loss = CE + w * supervised contrastive loss (Khosla et al.
                        2020; MBCL, CLMT-Net) on a training-only linear projection of
                        the features the classifier reads, L2-normalised, temperature
                        supcon_tau. The projection is not part of the model, so it is
                        not in the parameter count
  n_crops = k > 0       each training batch is a random crop of crop_frac of the
                        window per trial; validation/test use the mean logits of k
                        evenly spaced crops (start, middle, end)

With LEAN's recipe (best_val, no SupCon, no crops) every random number is drawn in the
same order as in LEAN's runs: same CV splits, augmentation, loader order, initial
weights (for LEAN's architecture) and dropout masks.
"""
from src.determinism import seed_worker   # sets CUBLAS/PYTHONHASHSEED env before torch loads

import copy
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

from src.config import ExperimentConfig, PIPELINE_VERSION, MODEL_LABELS
from src.data import (load_pair, slice_time_windows,
                      z_score_normalize, add_gaussian_noise_augmentation, EEGDataset)
from src.models.eegnet import EEGNet
from src.models.lean_v2 import LeanV2
from src.utils import init_weights, is_pair_done, write_pair_results

warnings.filterwarnings('ignore')


def pair_names(cfg: ExperimentConfig, class_a: int, class_b: int):
    """(PAIR_NAME, PAIR_SLUG), e.g. ('WAA/SAA', 'WAA_SAA')."""
    pair_name = f"{cfg.joint_names[class_a]}/{cfg.joint_names[class_b]}"
    return pair_name, pair_name.replace("/", "_")


def build_model(cfg: ExperimentConfig, n_channels: int, t_samples: int, n_classes: int = 2):
    """The preset's network for a window of t_samples (built for the crop length when
    the preset trains on crops)."""
    T = cfg.model_samples(t_samples)
    if cfg.arch['arch'] == 'eegnet':
        return EEGNet(n_channels=n_channels, n_timepoints=T, n_classes=n_classes,
                      cfg=dict(cfg.arch))
    if cfg.arch['arch'] == 'lean_v2':
        return LeanV2(n_channels=n_channels, n_timepoints=T, n_classes=n_classes,
                      cfg=dict(cfg.arch), max_timepoints=cfg.max_model_samples())
    raise ValueError(f"unknown arch: {cfg.arch['arch']}")


def build_projection(cfg: ExperimentConfig, model):
    """Training-only SupCon projection head, or None."""
    if not cfg.train['supcon_weight']:
        return None
    return nn.Linear(model.feature_dim, cfg.train['supcon_dim'])


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
def supcon_loss(z, y, tau):
    """Supervised contrastive loss (Khosla et al. 2020, L_out) on L2-normalised z.
    Every other trial of the same class in the batch (including a trial's noisy copy)
    is a positive. Written without boolean indexing, so it stays deterministic."""
    n = z.shape[0]
    sim = (z @ z.t()) / tau
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)                         # exclude self-similarity
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    n_pos = pos.sum(dim=1)
    has_pos = (n_pos > 0).to(z.dtype)
    per_row = -(log_prob * pos.to(z.dtype)).sum(dim=1) / n_pos.clamp(min=1).to(z.dtype)
    return (per_row * has_pos).sum() / has_pos.sum().clamp(min=1.0)


def crop_offsets(T: int, L: int, k: int) -> list[int]:
    """k evenly spaced crop starts from 0 to T - L (start, middle, end for k = 3)."""
    if k <= 1:
        return [(T - L) // 2]
    return [round(i * (T - L) / (k - 1)) for i in range(k)]


def random_crop(eeg, L: int, gen):
    """Per-trial random crop of length L from (B, C, T), drawn from `gen` (CPU)."""
    B, C, T = eeg.shape
    if L >= T:
        return eeg
    starts = torch.randint(0, T - L + 1, (B,), generator=gen)
    idx = starts[:, None] + torch.arange(L)[None, :]               # (B, L)
    return eeg.gather(2, idx[:, None, :].expand(B, C, L))


def predict_logits(model, eeg, cfg: ExperimentConfig):
    """Logits for a full-window batch; mean over k crops when the preset uses crops."""
    k = cfg.train['n_crops']
    if not k:
        return model(eeg)
    T = eeg.shape[-1]
    L = cfg.model_samples(T)
    outs = [model(eeg[..., o:o + L]) for o in crop_offsets(T, L, k)]
    return torch.stack(outs, dim=0).mean(dim=0)


def train_epoch(model, proj, train_loader, criterion, optimizer, device, cfg, crop_gen):
    """Train for one epoch."""
    model.train()
    if proj is not None:
        proj.train()
    w_sc, tau = cfg.train['supcon_weight'], cfg.train['supcon_tau']
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        eeg = batch['eeg']
        if cfg.train['n_crops']:
            eeg = random_crop(eeg, cfg.model_samples(eeg.shape[-1]), crop_gen)
        eeg = eeg.to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        if proj is None:
            outputs = model(eeg)
            loss = criterion(outputs, labels)
        else:
            outputs, feats = model(eeg, return_features=True)
            z = F.normalize(proj(feats), dim=1)
            loss = criterion(outputs, labels) + w_sc * supcon_loss(z, labels, tau)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device, cfg):
    """Evaluate (cross-entropy only)."""
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

            outputs = predict_logits(model, eeg, cfg)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(val_loader), 100.0 * correct / total, all_preds, all_labels


def make_loaders(cfg: ExperimentConfig, X_tr, y_tr, X_va, y_va, X_te, y_te, loader_seed: int):
    """Train (shuffled, seeded) / val / test loaders, as in LEAN's runs."""
    train_generator = torch.Generator()
    train_generator.manual_seed(int(loader_seed))
    train_loader = DataLoader(EEGDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, generator=train_generator,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(EEGDataset(X_va, y_va), batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, worker_init_fn=seed_worker)
    test_loader = DataLoader(EEGDataset(X_te, y_te), batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, worker_init_fn=seed_worker)
    return train_loader, val_loader, test_loader


class WeightAverager:
    """Running mean of a model's state_dict (float tensors in float64; integer buffers
    such as BN's num_batches_tracked keep their latest value)."""

    def __init__(self):
        self.sum, self.n = None, 0

    def add(self, model):
        sd = model.state_dict()
        if self.sum is None:
            self.sum = {k: (v.detach().double().clone() if v.is_floating_point()
                            else v.detach().clone()) for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    self.sum[k] += v.detach().double()
                else:
                    self.sum[k] = v.detach().clone()
        self.n += 1

    def load_into(self, model):
        sd = model.state_dict()
        model.load_state_dict({k: ((v / self.n).to(sd[k].dtype) if sd[k].is_floating_point()
                                   else v) for k, v in self.sum.items()})


def fit_fold(cfg, model, proj, train_loader, val_loader, device, log_prefix, log_epochs,
             crop_gen):
    """Train one fold with the preset's schedule; leaves the final weights in `model`.
    Returns (val_accuracy of the kept weights, best_epoch)."""
    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_epochs = cfg.n_epochs

    if cfg.train['schedule'] == 'best_val':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                         patience=5)
        best_val_acc, best_model_state, best_epoch = 0.0, None, 0
        for epoch in range(n_epochs):
            train_loss, train_acc = train_epoch(model, proj, train_loader, criterion,
                                                optimizer, device, cfg, crop_gen)
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device, cfg)
            scheduler.step(val_loss)
            # LEAN's selection rule
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            if log_epochs:
                print(f"    {log_prefix} ep{epoch + 1:>3}/{n_epochs}: "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:6.2f}% "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:6.2f}% "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return best_val_acc, best_epoch

    # fixed_avg
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    n_last = max(1, min(int(cfg.train['avg_last']), n_epochs))
    averager = WeightAverager()
    for epoch in range(n_epochs):
        lr_now = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_epoch(model, proj, train_loader, criterion,
                                            optimizer, device, cfg, crop_gen)
        scheduler.step()
        if epoch >= n_epochs - n_last:
            averager.add(model)
        if log_epochs:
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device, cfg)
            print(f"    {log_prefix} ep{epoch + 1:>3}/{n_epochs}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:6.2f}% "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:6.2f}% lr={lr_now:.2e}",
                  flush=True)
    averager.load_into(model)
    _, val_acc, _, _ = validate(model, val_loader, criterion, device, cfg)
    return val_acc, n_epochs


def train_pair(class_a: int, class_b: int, cfg: ExperimentConfig, overwrite: bool = False) -> str:
    """Train + test cfg.model on one pair; returns the pair's results dir."""
    class_a, class_b = int(class_a), int(class_b)
    seed = int(cfg.seed)
    PAIR_NAME, PAIR_SLUG = pair_names(cfg, class_a, class_b)
    PAIR_DIR = os.path.join(cfg.results_root, PAIR_SLUG)

    if is_pair_done(PAIR_DIR, cfg.model) and not overwrite:
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
          f"model={cfg.model} ({MODEL_LABELS[cfg.model]})  mode={cfg._mode}  "
          f"band={cfg.band_tag}  {cfg.sampling_rate} Hz", flush=True)
    print(f"  arch:  {cfg.arch}", flush=True)
    print(f"  train: {cfg.train} (epochs this mode: {cfg.n_epochs})", flush=True)
    print(f"Test time windows: {TEST_TIME_WINDOWS_MS} ms = {TEST_TIME_WINDOWS_SAMPLES} samples",
          flush=True)

    # =======================================================================
    # LOAD DATA + BINARY FILTER + TIME WINDOWS
    # =======================================================================
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
    time_windows_data = {t_ms: windows_by_samples[t_samp] for t_ms, t_samp
                         in zip(TEST_TIME_WINDOWS_MS_FINAL, TEST_TIME_WINDOWS_SAMPLES_FINAL)}

    n_channels = X_full.shape[1]
    unique_subjects = [int(s) for s in np.unique(subject_ids)]
    if cfg._max_subjects is not None:
        unique_subjects = unique_subjects[:cfg._max_subjects]

    subject_data = {}
    for subj_id in unique_subjects:
        subj_mask = (subject_ids == subj_id)
        subject_data[subj_id] = {'y': y_full[subj_mask]}
        for t_ms in TEST_TIME_WINDOWS_MS_FINAL:
            subject_data[subj_id][f'X_{t_ms}ms'] = time_windows_data[t_ms][subj_mask]

    n_classes = 2
    log_epochs = cfg._mode == 'smoke'

    n_params = {str(t_ms): count_params(build_model(cfg, n_channels, t_samp, n_classes))
                for t_ms, t_samp in zip(TEST_TIME_WINDOWS_MS_FINAL, TEST_TIME_WINDOWS_SAMPLES_FINAL)}
    print(f"\n  Parameters by window: "
          + ", ".join(f"{k} ms {v:,}" for k, v in n_params.items()), flush=True)
    print(f"  Subjects: {unique_subjects} | folds {cfg.n_folds} | epochs {cfg.n_epochs} | "
          f"batch {cfg.batch_size}", flush=True)

    # =======================================================================
    # WITHIN-SUBJECT TRAINING + TESTING  (structure and seeds as in LEAN's runs)
    # =======================================================================
    all_subject_results = []
    t_pair = time.time()

    for subj_id in unique_subjects:
        subj_id = int(subj_id)
        t_subj = time.time()
        y_subj = subject_data[subj_id]['y']

        if len(np.unique(y_subj)) < 2 or len(y_subj) < 10:
            print(f"Skipping Subject {subj_id}: insufficient data", flush=True)
            continue

        subj_indices = np.arange(len(y_subj))
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=int(42 + subj_id))
        cv_splits = list(skf.split(subj_indices, y_subj))   # same splits at every window

        subject_time_window_results = {}

        for tw_idx, t_ms in enumerate(TEST_TIME_WINDOWS_MS_FINAL):
            t_samples = TEST_TIME_WINDOWS_SAMPLES_FINAL[tw_idx]
            X_subj_tw = subject_data[subj_id][f'X_{t_ms}ms']
            fold_results_tw = []

            for fold_idx, (train_val_indices, test_indices) in enumerate(cv_splits):
                train_indices, val_indices = train_test_split(
                    train_val_indices, test_size=cfg.val_size_of_train,
                    random_state=int(42 + subj_id), stratify=y_subj[train_val_indices])

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
                crop_gen = torch.Generator()
                crop_gen.manual_seed(loader_seed + 7)

                model_init_seed = loader_seed
                torch.manual_seed(model_init_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(model_init_seed)

                model = build_model(cfg, n_channels, t_samples, n_classes).to(device)
                model.apply(init_weights)
                proj = build_projection(cfg, model)      # after the model: its init is unchanged
                if proj is not None:
                    proj = proj.to(device)
                    proj.apply(init_weights)

                val_acc, best_epoch = fit_fold(
                    cfg, model, proj, train_loader, val_loader, device,
                    log_prefix=f"S{subj_id} {t_ms}ms fold{fold_idx + 1}",
                    log_epochs=log_epochs, crop_gen=crop_gen)

                test_loss, test_acc, test_preds, test_labels = validate(
                    model, test_loader, nn.CrossEntropyLoss(), device, cfg)
                test_preds = np.array(test_preds)
                test_labels = np.array(test_labels)

                print(f"  S{subj_id} {t_ms}ms fold{fold_idx + 1}/{len(cv_splits)}: "
                      f"val={val_acc:6.2f}% (ep{best_epoch}) test={test_acc:6.2f}%", flush=True)

                fold_results_tw.append({
                    'fold': fold_idx + 1,
                    'test_accuracy': test_acc,
                    'test_loss': test_loss,
                    'test_preds': test_preds,
                    'test_labels': test_labels,
                    'n_test_samples': len(test_labels),
                    'val_accuracy': val_acc,
                    'best_epoch': best_epoch,
                })

                del model, proj, train_loader, val_loader, test_loader
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

        all_subject_results.append({'subject_id': subj_id, 'n_folds': cfg.n_folds,
                                    'time_window_results': subject_time_window_results})

        print(f"S{subj_id:>2}: " + "  ".join(
            f"{t}ms={subject_time_window_results[t]['test_accuracy']:6.2f}%"
            for t in TEST_TIME_WINDOWS_MS_FINAL)
            + f"   ({time.time() - t_subj:.0f}s)", flush=True)

    print(f"\nALL SUBJECTS COMPLETED - {PAIR_NAME} [{cfg.model}]  "
          f"({time.time() - t_pair:.0f}s)", flush=True)

    if not all_subject_results:
        raise RuntimeError(f"{PAIR_NAME} [{cfg.model}]: every subject was skipped")

    meta = dict(seed=seed, gpu=GPU, torch=torch.__version__,
                model=cfg.model, model_label=MODEL_LABELS[cfg.model],
                mode=cfg._mode, band=cfg.band_tag,
                dataset_root=cfg.dataset_root,
                pipeline_version=PIPELINE_VERSION, preset_hash=cfg.preset_hash,
                decim=cfg.decim, orig_sampling_rate=cfg.orig_sampling_rate,
                sampling_rate=cfg.sampling_rate,
                window_samples=list(TEST_TIME_WINDOWS_SAMPLES_FINAL),
                n_folds=cfg.n_folds, n_epochs=cfg.n_epochs,
                n_params=n_params,
                arch=dict(cfg.arch), train=dict(cfg.train),
                seconds=round(time.time() - t_pair, 1))
    wide = write_pair_results(PAIR_DIR, PAIR_NAME, class_a, class_b,
                              TEST_TIME_WINDOWS_MS_FINAL, all_subject_results, meta)

    print(f"\n{'=' * 68}", flush=True)
    print(f"SAVED {PAIR_NAME} [{cfg.model}] -> {PAIR_DIR}", flush=True)
    print(f"{'=' * 68}", flush=True)
    print(wide.round(2).to_string(), flush=True)
    print("\nmean:", wide.mean().round(2).to_dict(), flush=True)
    if wide.isna().any().any():
        print("WARNING: missing subject/window cells - a subject was skipped.", flush=True)

    return PAIR_DIR
