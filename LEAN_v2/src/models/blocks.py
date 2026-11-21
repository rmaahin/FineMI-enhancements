"""Building blocks for LEAN_v2.

EfficientChannelAttention is verbatim from 'Fine Conformer Fixation Cross Fix'
(LEAN's channel gate; [P3]: x.amax instead of the adaptive max-pool module, whose CUDA
backward is nondeterministic). The other LEAN blocks are the same layers with the
activation made selectable (ReLU for LEAN, ELU from R5) and an optional max-norm.
They are constructed in exactly LEAN's order, so with ReLU and no max-norm they draw
the same initial weights from the same seed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['make_act', 'Conv1dMaxNorm', 'Conv2dMaxNorm', 'LinearMaxNorm',
           'EfficientChannelAttention', 'DSConv1d', 'PerElectrodeMTC', 'DepthAttention',
           'TemporalBlock']


def make_act(name: str) -> nn.Module:
    return {'relu': nn.ReLU, 'elu': nn.ELU}[name]()


def _renorm_(weight, max_norm):
    if max_norm is not None:
        weight.data = torch.renorm(weight.data, p=2, dim=0, maxnorm=max_norm)


class Conv1dMaxNorm(nn.Conv1d):
    """Conv1d whose output filters are renormalized to L2 norm <= max_norm before use
    (max_norm=None: a plain Conv1d)."""

    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        _renorm_(self.weight, self.max_norm)
        return super().forward(x)


class Conv2dMaxNorm(nn.Conv2d):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        _renorm_(self.weight, self.max_norm)
        return super().forward(x)


class LinearMaxNorm(nn.Linear):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        _renorm_(self.weight, self.max_norm)
        return super().forward(x)


class EfficientChannelAttention(nn.Module):
    """Efficient Channel Attention block (LEAN's channel gate, paper Eq. 5)."""

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


class DSConv1d(nn.Module):
    """LEAN's DepthwiseSeparableConv1d (depthwise k, BN, act, pointwise, BN, act),
    'same' length, activation selectable."""

    def __init__(self, channels, kernel_size, act='relu'):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2,
                                   groups=channels, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = make_act(act)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        return self.act(self.bn2(self.pointwise(x)))


class PerElectrodeMTC(nn.Module):
    """LEAN's front end (paper Eq. 1): three branches, each a per-electrode temporal
    filter (kernel k, groups = electrodes) -> 1x1 conv mixing the electrodes into
    `filters` spatial filters (optionally max-norm) -> BN -> act. Output 3*filters."""

    def __init__(self, n_channels, filters=16, kernels=(7, 15, 31), act='relu',
                 max_norm=None):
        super().__init__()
        ks = tuple(int(k) for k in kernels)
        if len(ks) != 3 or any(k % 2 != 1 for k in ks):
            raise ValueError(f"need three odd kernels (padding k//2 keeps T), got {kernels}")

        def branch(k):
            return nn.Sequential(
                nn.Conv1d(n_channels, n_channels, kernel_size=k, padding=k // 2,
                          groups=n_channels, bias=False),
                Conv1dMaxNorm(n_channels, filters, kernel_size=1, bias=False,
                              max_norm=max_norm),
                nn.BatchNorm1d(filters),
                make_act(act))

        self.branch1, self.branch2, self.branch3 = (branch(k) for k in ks)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class DepthAttention(nn.Module):
    """LMDA-Net's depth attention (Miao et al. 2023) on (B, D, C, T) feature maps:
    average over the electrodes, a k x 1 conv ACROSS the D filter maps, softmax over D,
    then rescale every map. LMDA multiplies by the electrode count; here the weights are
    multiplied by D so that uniform attention leaves the input unchanged. Parameters:
    k + 1."""

    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)

    def forward(self, x):
        d = x.shape[1]
        s = x.mean(dim=2, keepdim=True)            # (B, D, 1, T)
        s = self.conv(s.transpose(1, 2))           # (B, 1, D, T)
        w = torch.softmax(s, dim=2).transpose(1, 2)  # (B, D, 1, T)
        return x * (w * d)


class TemporalBlock(nn.Module):
    """EEG-TCNet residual block (Ingolfsson et al. 2020): two causal dilated convs, each
    followed by BN, activation and dropout, plus the identity skip, then activation.
    Same channel count in and out, so no 1x1 skip conv. Parameters: 2*ch*ch*k + 4*ch."""

    def __init__(self, channels, kernel_size, dilation, p_drop, act='elu'):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = make_act(act)
        self.drop1 = nn.Dropout(p_drop)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = make_act(act)
        self.drop2 = nn.Dropout(p_drop)
        self.act_out = make_act(act)

    def forward(self, x):
        y = self.drop1(self.act1(self.bn1(self.conv1(F.pad(x, (self.pad, 0))))))
        y = self.drop2(self.act2(self.bn2(self.conv2(F.pad(y, (self.pad, 0))))))
        return self.act_out(x + y)
