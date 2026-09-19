"""Option B — FINE-Conformer hybrid, on 50 Hz input.

FINE MTC + spatial conv -> pool -> Transformer encoder -> FINE depthwise-separable
conv + ECA -> FINE-style head. Same module as Conformer_0.5_3hz/src/models/conformer_b.py.
At 50 Hz the config keeps the 7/15/31-sample MTC kernels (now 140/300/620 ms) and
divides the pre-transformer pool by D = 5 (15/15 -> 3/3), so each window still
gives 13 / 25 / 50 / 66 tokens.
"""
from __future__ import annotations   # `int | None` annotations on Python < 3.10

import torch
import torch.nn as nn

from src.models.fine_blocks import (
    MultiscaleTemporalBlock,
    SpatialFeatureExtraction,
    DepthwiseSeparableConv1d,
    EfficientChannelAttention,
)


def _default_cfg() -> dict:
    # Single source of truth: ExperimentConfig defaults (src/config.py)
    from src.config import ExperimentConfig
    d = ExperimentConfig(model='conformer_b', dataset_root='')
    return dict(d.conformer_b,
                test_time_windows_ms=d.test_time_windows_ms,
                sampling_rate=d.sampling_rate)


class TransformerTail(nn.Module):
    """Everything in Option B after the (B, d_model, T) feature map: pre-transformer
    pool -> positional embedding -> encoder -> DS-conv + ECA -> global pool -> head.

    Shared by Conformer B and Conformer B + CWT, so the two differ only in the
    front-end that produces the (B, d_model, T) map.
    """

    def __init__(self, d_model: int, n_timepoints: int, n_classes: int,
                 embedding_dim: int, c: dict):
        super().__init__()
        pool_kernel, pool_stride = c['pre_att_pool_kernel'], c['pre_att_pool_stride']

        def n_tokens(t):
            return (t - pool_kernel) // pool_stride + 1

        if n_timepoints < pool_kernel:
            raise ValueError(f"window of {n_timepoints} samples is shorter than the "
                             f"{pool_kernel}-sample pool (no transformer tokens)")
        self.n_tokens = n_tokens(n_timepoints)

        # ─── Pre-transformer pool ──────────────────────────────────────────
        self.pre_att_pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)

        # ─── Transformer encoder ───────────────────────────────────────────
        windows_samples = [int(t * c['sampling_rate'] / 1000) for t in c['test_time_windows_ms']]
        # max(..., self.n_tokens) only matters if n_timepoints exceeds every configured window
        max_tokens = max([n_tokens(s) for s in windows_samples] + [self.n_tokens])
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_tokens, d_model))

        # Separately constructed layers (not nn.TransformerEncoder, which deep-copies
        # one layer and so starts every layer with identical attention in_proj weights)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=c['n_heads'], dim_feedforward=c['d_ff'],
                dropout=c['p_att'], activation='gelu',
                batch_first=True, norm_first=c['norm_first'])
            for _ in range(c['n_layers'])
        ])
        self.norm = nn.LayerNorm(d_model)

        # ─── FINE feature fusion + ECA ─────────────────────────────────────
        k = c['fusion_kernel']
        self.feature_fusion = nn.Sequential(
            DepthwiseSeparableConv1d(d_model, d_model, k, groups=d_model, padding=k // 2),
            EfficientChannelAttention(d_model, reduction=c['eca_reduction']),
        )

        # ─── Head (mirrors FINE) ───────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.cnn_projection = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(c['p_head']),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: (B, d_model, T)
        x = self.pre_att_pool(x)                 # (B, d_model, T'')
        x = x.transpose(1, 2)                    # (B, T'', d_model)

        x = x + self.pos_embedding[:, :x.size(1)]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)                         # (B, T'', d_model)
        x = x.transpose(1, 2)                    # (B, d_model, T'')

        x = self.feature_fusion(x)               # (B, d_model, T'')
        x = self.global_pool(x)
        x = x.squeeze(-1)                        # (B, d_model)
        cnn_embedding = self.cnn_projection(x)
        return self.classifier(cnn_embedding)


class FineConformerHybrid(nn.Module):
    """cfg: the `conformer_b` hyperparameter dict, plus `test_time_windows_ms` and
    `sampling_rate` (used to size the positional embedding). Missing keys fall
    back to the ExperimentConfig defaults.
    """

    def __init__(self, n_channels: int, n_timepoints: int,
                 n_classes: int = 2, embedding_dim: int | None = None,
                 cfg: dict | None = None):
        super().__init__()
        c = {**_default_cfg(), **(cfg or {})}
        if embedding_dim is None:
            embedding_dim = c['embedding_dim']

        # MultiscaleTemporalBlock hardcodes its kernels; refuse a cfg it can't honor
        if tuple(c['mtc_kernels']) != (7, 15, 31):
            raise ValueError(f"mtc_kernels must be (7, 15, 31) (fixed in FINE's "
                             f"MultiscaleTemporalBlock), got {c['mtc_kernels']}")

        d_model = c['spatial_filters']

        # ─── FINE MTC block ────────────────────────────────────────────────
        self.multiscale_temporal = MultiscaleTemporalBlock(
            in_channels=n_channels, out_channels_per_branch=c['mtc_filters'])

        # ─── FINE spatial extraction ───────────────────────────────────────
        # Only its .conv (Conv1d + BN + ReLU) is used in forward; FINE's 2x maxpool
        # is skipped, as in Conformer_0.5_3hz.
        self.spatial_extraction = SpatialFeatureExtraction(
            in_channels=3 * c['mtc_filters'], out_channels=d_model,
            kernel_size=c['spatial_kernel'])

        self.tail = TransformerTail(d_model, n_timepoints, n_classes, embedding_dim, c)
        self.n_tokens = self.tail.n_tokens

    def forward(self, x_eeg):
        # x_eeg: (B, C, T)
        x = self.multiscale_temporal(x_eeg)      # (B, 96, T)
        x = self.spatial_extraction.conv(x)      # (B, 64, T)
        return self.tail(x)
