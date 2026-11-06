"""EEGNet-8,2 (Lawhern et al. 2018, J. Neural Eng. 15:056013), in PyTorch.

The FineMI authors' code has no EEGNet, so this follows the original paper and its
reference Keras code (arl-eegmodels EEGNet):

  block 1: temporal Conv2d(1, F1, (1, K1), 'same', no bias) -> BN
           -> depthwise Conv2d(F1, D*F1, (C, 1), groups=F1, no bias, max-norm 1.0)
           -> BN -> ELU -> AvgPool(1, P1) -> Dropout
  block 2: separable conv = depthwise Conv2d(D*F1, D*F1, (1, K2), groups=D*F1, 'same',
           no bias) + pointwise Conv2d(D*F1, F2, 1, no bias)
           -> BN -> ELU -> AvgPool(1, P2) -> Dropout
  classifier: flatten -> Linear(F2 * n_pool_out, n_classes, max-norm 0.25)

F1 = 8, D = 2, F2 = 16, dropout 0.5. The temporal kernel follows the paper's rule at
each sampling rate, K1 = half the sampling rate (0.5 s). Pools and the separable kernel
are the paper's 4 / 16 / 8 at 250 Hz and are halved (2 / 8 / 4) at 50 Hz (src/config.py).

The output is LOGITS. CrossEntropyLoss on logits is the same loss as Keras' softmax +
categorical cross-entropy, so (unlike the ShallowConvNet runs) there is no double softmax.

Max-norm is applied as in the ShallowConvNet folder: before every forward pass each
output filter's weights are renormalized to an L2 norm of at most max_norm.

Shapes for a window of T samples (C = 62 channels):
  (B, C, T) -> (B, 1, C, T) -> temporal (B, F1, C, T) -> depthwise (B, D*F1, 1, T)
  -> pool P1 (B, D*F1, 1, T//P1) -> separable (B, F2, 1, T//P1)
  -> pool P2 (B, F2, 1, n_pool_out) -> linear (B, n_classes)
  n_pool_out = (T // P1) // P2.
"""
from __future__ import annotations   # `dict | None` annotations on Python < 3.10

import torch
import torch.nn as nn


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


REQUIRED_KEYS = ('F1', 'D', 'F2', 'kern_length', 'pool1', 'sep_kern_length', 'pool2',
                 'drop_prob', 'bn_momentum', 'bn_eps', 'max_norm_depthwise', 'max_norm_linear')


class EEGNet(nn.Module):
    """cfg: one of the hyperparameter dicts in src/config.py (EEGNET_HPARAMS)."""

    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 cfg: dict | None = None):
        super().__init__()
        c = dict(cfg or {})
        missing = [k for k in REQUIRED_KEYS if k not in c]
        if missing:
            raise ValueError(f"EEGNet cfg is missing {missing}")
        F1, D, F2 = c['F1'], c['D'], c['F2']
        P1, P2 = c['pool1'], c['pool2']
        bn = dict(momentum=c['bn_momentum'], eps=c['bn_eps'])

        self.n_pool_out = (n_timepoints // P1) // P2
        if self.n_pool_out < 1:
            raise ValueError(f"window of {n_timepoints} samples is too short for pools "
                             f"{P1} and {P2}")

        # block 1
        self.conv_time = nn.Conv2d(1, F1, kernel_size=(1, c['kern_length']),
                                   padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1, **bn)
        self.conv_depth = Conv2dWithConstraint(F1, D * F1, kernel_size=(n_channels, 1),
                                               groups=F1, bias=False,
                                               max_norm=c['max_norm_depthwise'])
        self.bn2 = nn.BatchNorm2d(D * F1, **bn)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, P1))
        self.drop1 = nn.Dropout(c['drop_prob'])

        # block 2 (separable conv)
        self.conv_sep_depth = nn.Conv2d(D * F1, D * F1, kernel_size=(1, c['sep_kern_length']),
                                        groups=D * F1, padding='same', bias=False)
        self.conv_sep_point = nn.Conv2d(D * F1, F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2, **bn)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, P2))
        self.drop2 = nn.Dropout(c['drop_prob'])

        self.elu = nn.ELU()
        self.classifier = LinearWithConstraint(F2 * self.n_pool_out, n_classes,
                                               max_norm=c['max_norm_linear'])

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(1)                                   # (B, 1, C, T)
        x = self.bn1(self.conv_time(x))                      # (B, F1, C, T)
        x = self.elu(self.bn2(self.conv_depth(x)))           # (B, D*F1, 1, T)
        x = self.drop1(self.pool1(x))                        # (B, D*F1, 1, T//P1)
        x = self.conv_sep_point(self.conv_sep_depth(x))      # (B, F2, 1, T//P1)
        x = self.elu(self.bn3(x))
        x = self.drop2(self.pool2(x))                        # (B, F2, 1, n_pool_out)
        return self.classifier(x.flatten(1))                 # logits (B, n_classes)
