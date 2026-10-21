"""ShallowConvNet (Schirrmeister et al. 2017, "Deep learning with convolutional neural
networks for EEG decoding and visualization"), at 250 Hz.

Follows braindecode's ShallowFBCSPNet with its defaults: split first layer (temporal
conv with bias, then a bias-free full-scalp spatial conv), BatchNorm, square, mean
pool, safe log, dropout, and a conv classifier spanning the whole pooled length. The
classifier returns logits (no LogSoftmax), because training uses CrossEntropyLoss.

Shapes for a window of T samples (C = 62 channels):
  (B, C, T) -> (B, 1, C, T) -> temporal (B, 40, C, T-24) -> spatial (B, 40, 1, T-24)
  -> BN, square -> pool (B, 40, 1, n_pool_out) -> log, dropout -> classifier (B, 2)
  n_pool_out = (T - 24 - 75) // 15 + 1 = 7 / 19 / 44 / 61 at 200 / 375 / 750 / 1000.
"""
from __future__ import annotations   # `int | None` annotations on Python < 3.10

import torch
import torch.nn as nn


def _default_cfg() -> dict:
    # Single source of truth: ExperimentConfig defaults (src/config.py)
    from src.config import ExperimentConfig
    return dict(ExperimentConfig(model='shallowconvnet', dataset_root='').shallowconvnet)


def safe_log(x, eps: float = 1e-6):
    return torch.log(torch.clamp(x, min=eps))


class ShallowConvNet(nn.Module):
    """cfg: the `shallowconvnet` hyperparameter dict; missing keys fall back to the
    ExperimentConfig defaults."""

    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 cfg: dict | None = None):
        super().__init__()
        c = {**_default_cfg(), **(cfg or {})}
        F_t, K_t = c['n_filters_time'], c['filter_time_length']
        F_s = c['n_filters_spat']
        pool_k, pool_s = c['pool_time_length'], c['pool_time_stride']

        conv_len = n_timepoints - K_t + 1
        if conv_len < pool_k:
            raise ValueError(f"window of {n_timepoints} samples is too short for a "
                             f"{K_t}-sample conv followed by a {pool_k}-sample pool")
        self.n_pool_out = (conv_len - pool_k) // pool_s + 1

        self.conv_time = nn.Conv2d(1, F_t, kernel_size=(1, K_t), bias=True)
        self.conv_spat = nn.Conv2d(F_t, F_s, kernel_size=(n_channels, 1), bias=False)
        self.bnorm = nn.BatchNorm2d(F_s, momentum=c['bn_momentum'], affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_k), stride=(1, pool_s))
        self.drop = nn.Dropout(c['drop_prob'])
        self.classifier = nn.Conv2d(F_s, n_classes, kernel_size=(1, self.n_pool_out), bias=True)

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(1)                       # (B, 1, C, T)
        x = self.conv_time(x)                    # (B, F_t, C, T')
        x = self.conv_spat(x)                    # (B, F_s, 1, T')
        x = self.bnorm(x)
        x = x * x                                # square
        x = self.pool(x)                         # (B, F_s, 1, n_pool_out)
        x = safe_log(x)
        x = self.drop(x)
        x = self.classifier(x)                   # (B, n_classes, 1, 1)
        return x.flatten(1)
