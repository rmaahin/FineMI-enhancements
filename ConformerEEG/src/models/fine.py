"""FINE model — verbatim port from FineMI-enhancements/EMBC_deterministic-3.ipynb.

Only changes vs. the notebook:
  - [P3] EfficientChannelAttention uses x.amax(...) instead of the adaptive
    max-pool module (nondeterministic CUDA backward), as in the notebook.
  - CNNEarlyClassificationModel takes embedding_dim=None (-> 128, the
    notebook's value) and an ignored cfg=None, so all three models share one
    constructor signature.
"""
import torch
import torch.nn as nn

__all__ = [
    'EfficientChannelAttention',
    'DepthwiseSeparableConv1d',
    'FeatureFusionModule',
    'MultiscaleTemporalBlock',
    'SpatialFeatureExtraction',
    'CNNEarlyClassificationModel',
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


class FeatureFusionModule(nn.Module):
    """Feature Fusion Module with Depthwise Separable Conv and ECA."""

    def __init__(self, channels, kernel_size=(1, 5), groups=None):
        super(FeatureFusionModule, self).__init__()
        if groups is None:
            groups = channels

        k_size = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size
        self.ds_conv = DepthwiseSeparableConv1d(
            channels, channels, k_size, groups=groups, padding=k_size // 2
        )
        self.eca = EfficientChannelAttention(channels)

    def forward(self, x):
        x = self.ds_conv(x)
        x = self.eca(x)
        return x


class MultiscaleTemporalBlock(nn.Module):
    """Multiscale temporal convolution with 3 parallel branches."""

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
    """Spatial feature extraction with Conv and MaxPool."""

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


class CNNEarlyClassificationModel(nn.Module):
    """CNN model for early EEG classification."""

    def __init__(self, n_channels, n_timepoints, n_classes=2, embedding_dim=None, cfg=None):
        super(CNNEarlyClassificationModel, self).__init__()
        if embedding_dim is None:
            embedding_dim = 128   # notebook value

        self.multiscale_temporal = MultiscaleTemporalBlock(
            in_channels=n_channels, out_channels_per_branch=32)
        temporal_out_channels = 32 * 3

        self.spatial_extraction = SpatialFeatureExtraction(
            in_channels=temporal_out_channels, out_channels=64)

        self.feature_fusion = FeatureFusionModule(
            channels=64, kernel_size=(1, 5), groups=64)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.cnn_projection = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # NOTE: two-layer head — this is your original. Do not simplify.
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

    def forward(self, x_eeg):
        x = self.multiscale_temporal(x_eeg)
        x = self.spatial_extraction(x)
        x = self.feature_fusion(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        cnn_embedding = self.cnn_projection(x)
        return self.classifier(cnn_embedding)
