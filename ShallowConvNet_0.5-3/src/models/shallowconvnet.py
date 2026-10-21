"""ShallowConvNet (Schirrmeister et al. 2017) as implemented by the FineMI dataset
authors (Yi et al. 2025, EEG_decoding.py in github.com/Bill-Chen-BJ/FJMI_EEG_fNIRS),
at 250 Hz.

Layer for layer the same as the authors' model: temporal conv (with bias), bias-free
full-scalp spatial conv, BatchNorm, square, mean pool, log(clip(x, 1e-6)), dropout,
linear classifier, softmax. As in their code:
  - max-norm constraints: before every forward pass each output filter's weights are
    renormalized to an L2 norm of at most 2.0 (temporal and spatial conv) or 0.5
    (classifier), via torch.renorm on weight.data.
  - the output is a SOFTMAX, and training feeds it to nn.CrossEntropyLoss (which applies
    log-softmax again). This double softmax is kept on purpose to match the authors.

Shapes for a window of T samples (C = 62 channels):
  (B, C, T) -> (B, 1, C, T) -> temporal (B, 40, C, T-24) -> spatial (B, 40, 1, T-24)
  -> BN, square -> pool (B, 40, 1, n_pool_out) -> log, dropout -> flatten -> linear (B, 2)
  n_pool_out = (T - 24 - 75) // 15 + 1 = 7 / 19 / 44 / 61 at 200 / 375 / 750 / 1000.
"""
from __future__ import annotations   # `int | None` annotations on Python < 3.10

import torch
import torch.nn as nn


def _default_cfg() -> dict:
    # Single source of truth: ExperimentConfig defaults (src/config.py)
    from src.config import ExperimentConfig
    return dict(ExperimentConfig(model='shallowconvnet', dataset_root='').shallowconvnet)


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d whose output filters are renormalized to L2 norm <= max_norm before use."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    """Linear whose output rows are renormalized to L2 norm <= max_norm before use."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


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

        self.conv_time = Conv2dWithConstraint(1, F_t, kernel_size=(1, K_t), bias=True,
                                              max_norm=c['max_norm_temporal_conv'])
        self.conv_spat = Conv2dWithConstraint(F_t, F_s, kernel_size=(n_channels, 1), bias=False,
                                              max_norm=c['max_norm_spatial_conv'])
        self.bnorm = nn.BatchNorm2d(F_s, momentum=c['bn_momentum'], affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_k), stride=(1, pool_s))
        self.drop = nn.Dropout(c['drop_prob'])
        self.classifier = LinearWithConstraint(F_s * self.n_pool_out, n_classes,
                                               max_norm=c['max_norm_linear'])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(1)                       # (B, 1, C, T)
        x = self.conv_time(x)                    # (B, F_t, C, T')
        x = self.conv_spat(x)                    # (B, F_s, 1, T')
        x = self.bnorm(x)
        x = x * x                                # square
        x = self.pool(x)                         # (B, F_s, 1, n_pool_out)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop(x)
        x = self.classifier(x.flatten(1))        # (B, n_classes)
        return self.softmax(x)                   # probabilities, as in the authors' code
