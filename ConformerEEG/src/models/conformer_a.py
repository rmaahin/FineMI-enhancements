"""Option A — pure EEG Conformer (Song et al. 2022), adapted to FineMI (SPEC §6.1)."""
from __future__ import annotations   # `int | None` annotations on Python < 3.10

import torch
import torch.nn as nn


def _default_cfg() -> dict:
    # Single source of truth: ExperimentConfig defaults (src/config.py)
    from src.config import ExperimentConfig
    d = ExperimentConfig(model='conformer_a', experiment_name='', dataset_root='')
    return dict(d.conformer_a,
                test_time_windows_ms=d.test_time_windows_ms,
                sampling_rate=d.sampling_rate)


class EEGConformer(nn.Module):
    """Conv module -> Transformer encoder -> flatten + MLP head.

    cfg: the `conformer_a` hyperparameter dict, plus `test_time_windows_ms` and
    `sampling_rate` (used to size the positional embedding). Missing keys fall
    back to the ExperimentConfig defaults.
    """

    def __init__(self, n_channels: int, n_timepoints: int,
                 n_classes: int = 2, embedding_dim: int | None = None,
                 cfg: dict | None = None):
        super().__init__()
        c = {**_default_cfg(), **(cfg or {})}
        # embedding_dim is accepted and ignored (signature compatibility)

        F1 = c['F1']
        pool_kernel, pool_stride = c['pool_kernel'], c['pool_stride']

        def n_tokens(t):
            return (t - pool_kernel) // pool_stride + 1

        # ─── Convolutional module ──────────────────────────────────────────
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, c['temporal_kernel']),
                      padding=(0, c['temporal_kernel'] // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1, kernel_size=(n_channels, 1), bias=False),   # full scalp
            nn.BatchNorm2d(F1),
            nn.ELU(),
        )
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride))
        self.conv_dropout = nn.Dropout(c['p_conv'])

        # ─── Transformer encoder ───────────────────────────────────────────
        self.n_tokens = n_tokens(n_timepoints)
        windows_samples = [int(t * c['sampling_rate'] / 1000) for t in c['test_time_windows_ms']]
        # max(..., self.n_tokens) only matters if n_timepoints exceeds every configured window
        max_tokens = max([n_tokens(s) for s in windows_samples] + [self.n_tokens])
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_tokens, F1))

        # Separately constructed layers (not nn.TransformerEncoder, which deep-copies
        # one layer and so starts every layer with identical attention in_proj weights)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=F1, nhead=c['n_heads'], dim_feedforward=c['d_ff'],
                dropout=c['p_att'], activation='gelu',
                batch_first=True, norm_first=c['norm_first'])
            for _ in range(c['n_layers'])
        ])
        self.norm = nn.LayerNorm(F1)

        # ─── Classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(self.n_tokens * F1, c['proj_dim']),
            nn.GELU(),
            nn.Dropout(c['p_head']),
            nn.Linear(c['proj_dim'], n_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(1)                       # (B, 1, C, T)
        x = self.temporal_conv(x)                # (B, F1, C, T)
        x = self.spatial_conv(x)                 # (B, F1, 1, T)
        x = self.pool(x)                         # (B, F1, 1, T')
        x = self.conv_dropout(x)
        x = x.squeeze(2).transpose(1, 2)         # (B, T', F1)

        x = x + self.pos_embedding[:, :x.size(1)]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)                         # (B, T', F1)

        x = x.flatten(1)                         # (B, T' * F1)
        return self.head(x)
