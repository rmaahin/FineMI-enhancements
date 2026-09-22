"""Slim Conformer B — Conformer B (FINE-Conformer hybrid) cut to 25,752 parameters.

Same pipeline as Conformer B at 50 Hz, with four changes:
  1. Temporal branches are depthwise-separable: each electrode gets its own 7 / 15 /
     31-sample filter, then a 1x1 conv mixes the 62 electrodes into 16 filters per
     branch (Conformer B: 32 full-scalp filters per branch).
  2. Width 32 instead of 64 from the spatial conv onward.
  3. One transformer layer instead of two, feed-forward 64 instead of 256.
  4. Head 32 -> 32 -> 32 -> 2 instead of 64 -> 128 -> 128 -> 2.
The pooling, token counts, fusion conv + ECA, and every activation and dropout
position are unchanged.

Parameter budget (checked against EXPECTED_PARAMS by scripts/preflight.py):
  temporal branches   62*(7+15+31) + 3*(62*16) + 3*32      =  6,358
  spatial conv        48*32*3 + 64                          =  4,672
  positional emb      66*32                                 =  2,112
  transformer layer   (3*32*32+96) + (32*32+32)
                      + (32*64+64) + (64*32+32) + 2*64      =  8,544
  final LayerNorm     2*32                                  =     64
  fusion + ECA        32*5 + 64 + 32*32 + 64 + 2*(32*8)     =  1,824
  head                (32*32+32) + (32*32+32) + (32*2+2)    =  2,178
                                                   total    = 25,752
"""
from __future__ import annotations   # `int | None` annotations on Python < 3.10

import torch
import torch.nn as nn

from src.models.blocks import (DepthwiseSeparableMTC, DepthwiseSeparableConv1d,
                               EfficientChannelAttention)


def _default_cfg() -> dict:
    # Single source of truth: ExperimentConfig defaults (src/config.py)
    from src.config import ExperimentConfig
    d = ExperimentConfig(model='conformer_b_slim', dataset_root='')
    return dict(d.conformer_b_slim,
                test_time_windows_ms=d.test_time_windows_ms,
                sampling_rate=d.sampling_rate)


class SlimConformerB(nn.Module):
    """cfg: the `conformer_b_slim` hyperparameter dict, plus `test_time_windows_ms` and
    `sampling_rate` (used to size the positional embedding). Missing keys fall back to
    the ExperimentConfig defaults.
    """

    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 embedding_dim: int | None = None, cfg: dict | None = None):
        super().__init__()
        c = {**_default_cfg(), **(cfg or {})}
        if embedding_dim is None:
            embedding_dim = c['embedding_dim']

        d_model = c['spatial_filters']
        pool_kernel, pool_stride = c['pre_att_pool_kernel'], c['pre_att_pool_stride']

        def n_tokens(t):
            return (t - pool_kernel) // pool_stride + 1

        if n_timepoints < pool_kernel:
            raise ValueError(f"window of {n_timepoints} samples is shorter than the "
                             f"{pool_kernel}-sample pool (no transformer tokens)")
        if d_model % c['n_heads'] != 0:
            raise ValueError(f"width {d_model} is not divisible by {c['n_heads']} heads")
        self.n_tokens = n_tokens(n_timepoints)

        # ─── Temporal branches (per-electrode filters + 1x1 mixing) ───────
        self.multiscale_temporal = DepthwiseSeparableMTC(
            in_channels=n_channels, out_channels_per_branch=c['mtc_filters'],
            kernels=tuple(c['mtc_kernels']))

        # ─── Spatial conv (Conv1d + BN + ReLU; no maxpool, as in Conformer B) ─
        k = c['spatial_kernel']
        self.spatial = nn.Sequential(
            nn.Conv1d(3 * c['mtc_filters'], d_model, kernel_size=k, padding=k // 2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU())

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
        fk = c['fusion_kernel']
        self.feature_fusion = nn.Sequential(
            DepthwiseSeparableConv1d(d_model, d_model, fk, groups=d_model, padding=fk // 2),
            EfficientChannelAttention(d_model, reduction=c['eca_reduction']),
        )

        # ─── Head (FINE layout, narrower) ──────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.cnn_projection = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.ReLU(),
            nn.Dropout(c['p_proj']))
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, c['classifier_hidden']),
            nn.ReLU(),
            nn.Dropout(c['p_head']),
            nn.Linear(c['classifier_hidden'], n_classes))

    def forward(self, x_eeg):
        # x_eeg: (B, C, T)
        x = self.multiscale_temporal(x_eeg)      # (B, 48, T)
        x = self.spatial(x)                      # (B, 32, T)
        x = self.pre_att_pool(x)                 # (B, 32, T'')
        x = x.transpose(1, 2)                    # (B, T'', 32)

        x = x + self.pos_embedding[:, :x.size(1)]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)                         # (B, T'', 32)
        x = x.transpose(1, 2)                    # (B, 32, T'')

        x = self.feature_fusion(x)               # (B, 32, T'')
        x = self.global_pool(x).squeeze(-1)      # (B, 32)
        return self.classifier(self.cnn_projection(x))
