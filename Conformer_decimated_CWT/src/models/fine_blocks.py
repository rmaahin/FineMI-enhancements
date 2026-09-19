"""FINE building blocks used by Conformer B and Conformer B + CWT.

1D blocks: verbatim from Conformer_0.5_3hz/src/models/fine.py (itself a port of
EMBC_deterministic-3.ipynb), including [P3]: ECA uses x.amax(...) instead of the
adaptive max-pool module, whose CUDA backward is nondeterministic.

2D block: MultiscaleTemporalBlock2d from CWT-modified/fine_mi.py, for (C, F, T)
scalogram input. Its kernels are (1, k): purely temporal, frequency untouched.
"""
import torch
import torch.nn as nn

__all__ = [
    'EfficientChannelAttention',
    'DepthwiseSeparableConv1d',
    'MultiscaleTemporalBlock',
    'SpatialFeatureExtraction',
    'MultiscaleTemporalBlock2d',
]


class EfficientChannelAttention(nn.Module):
    """Efficient Channel Attention block."""

    def __init__(self, channels, reduction=4):
        super(EfficientChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # [P3] max-pool module removed: its CUDA backward is nondeterministic

        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, channels, time)
        avg_out = self.fc(self.avg_pool(x))
        # [P3] x.amax(dim=-1, keepdim=True) == global max-pool over time, deterministic
        max_out = self.fc(x.amax(dim=-1, keepdim=True))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise Separable Convolution block."""

    def __init__(self, in_channels, out_channels, kernel_size, groups, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False
        )
        self.bn1 = nn.BatchNorm1d(in_channels)

        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class MultiscaleTemporalBlock(nn.Module):
    """Multiscale temporal convolution with 3 parallel branches (kernels 7 / 15 / 31)."""

    def __init__(self, in_channels, out_channels_per_branch=32):
        super(MultiscaleTemporalBlock, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_channels_per_branch),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(out_channels_per_branch),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=31, padding=15, bias=False),
            nn.BatchNorm1d(out_channels_per_branch),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat([out1, out2, out3], dim=1)


class SpatialFeatureExtraction(nn.Module):
    """Spatial feature extraction with Conv and MaxPool (Conformer B uses only .conv)."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SpatialFeatureExtraction, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x


class MultiscaleTemporalBlock2d(nn.Module):
    """2D MTC over (F, T) scalograms: the EEG channels are the conv in-channels and
    each branch's kernel is (1, k), so it is purely temporal like the 1D block."""

    def __init__(self, in_channels, out_channels_per_branch=32, kernels=(7, 15, 31)):
        super().__init__()
        k1, k2, k3 = (int(k) for k in kernels)
        for k in (k1, k2, k3):
            if k % 2 != 1:
                raise ValueError(f"MTC kernels must be odd (padding k//2 keeps T), got {kernels}")

        def branch(k):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=(1, k),
                          padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(out_channels_per_branch),
                nn.ReLU())

        self.branch1, self.branch2, self.branch3 = branch(k1), branch(k2), branch(k3)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
