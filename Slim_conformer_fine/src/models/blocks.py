"""Building blocks for Slim Conformer B.

EfficientChannelAttention and DepthwiseSeparableConv1d are verbatim from
Conformer_decimated_CWT/src/models/fine_blocks.py (FINE, EMBC_deterministic-3.ipynb),
including [P3]: ECA uses x.amax(...) instead of the adaptive max-pool module, whose
CUDA backward is nondeterministic.

DepthwiseSeparableMTC is new: FINE's three-branch multiscale temporal block with each
full-scalp temporal conv split into a per-electrode (depthwise) temporal filter and a
1x1 conv that mixes the electrodes.
"""
import torch
import torch.nn as nn

__all__ = ['EfficientChannelAttention', 'DepthwiseSeparableConv1d', 'DepthwiseSeparableMTC']


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


class DepthwiseSeparableMTC(nn.Module):
    """Three parallel temporal branches, each: per-electrode temporal conv (kernel k,
    groups = electrodes) -> 1x1 conv to `out_channels_per_branch` -> BN -> ReLU.

    Parameters per branch: in*k + in*out + 2*out. The BN sits after the mixing conv
    only, matching where FINE's MultiscaleTemporalBlock puts it.
    """

    def __init__(self, in_channels, out_channels_per_branch=16, kernels=(7, 15, 31)):
        super().__init__()
        ks = tuple(int(k) for k in kernels)
        if len(ks) != 3 or any(k % 2 != 1 for k in ks):
            raise ValueError(f"need three odd kernels (padding k//2 keeps T), got {kernels}")

        def branch(k):
            return nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=k, padding=k // 2,
                          groups=in_channels, bias=False),
                nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels_per_branch),
                nn.ReLU())

        self.branch1, self.branch2, self.branch3 = (branch(k) for k in ks)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
