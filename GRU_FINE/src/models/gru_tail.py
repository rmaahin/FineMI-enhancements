"""GRUTail — Option B's TransformerTail with the transformer encoder replaced by a
2-layer bidirectional GRU.

Mirrors Conformer_decimated_CWT/src/models/conformer_b.py:TransformerTail step for
step; only the sequence mixer differs:

  pool 3/3 -> [GRU mixer] -> DS-conv + ECA -> global pool -> FINE-style head

  transformer (Option B)                 GRU (here)
  x + pos_embedding                      no positional embedding (GRU is order-aware)
  2 x pre-norm encoder layer (residual)  x = LayerNorm(x + Dropout(BiGRU(x)))
  final LayerNorm

The BiGRU has hidden 32 per direction, so its output is 2 x 32 = 64 = d_model and
the residual, fusion conv and head are unchanged. It keeps PyTorch's default GRU
init: init_weights (src/utils.py) only touches Conv / Linear / BN / LayerNorm.
"""
from __future__ import annotations   # `int | None` annotations on Python < 3.10

import torch.nn as nn

from src.models.fine_blocks import DepthwiseSeparableConv1d, EfficientChannelAttention


class GRUTail(nn.Module):
    """Everything after the (B, d_model, T) feature map: pre-GRU pool -> BiGRU ->
    DS-conv + ECA -> global pool -> head. Shared by FINE + GRU and FINE + GRU + CWT,
    so the two differ only in the front-end that produces the (B, d_model, T) map.
    """

    def __init__(self, d_model: int, n_timepoints: int, n_classes: int,
                 embedding_dim: int, c: dict):
        super().__init__()
        pool_kernel, pool_stride = c['pre_rnn_pool_kernel'], c['pre_rnn_pool_stride']

        if n_timepoints < pool_kernel:
            raise ValueError(f"window of {n_timepoints} samples is shorter than the "
                             f"{pool_kernel}-sample pool (no GRU steps)")
        self.n_tokens = (n_timepoints - pool_kernel) // pool_stride + 1

        n_dirs = 2 if c['gru_bidirectional'] else 1
        if c['gru_hidden'] * n_dirs != d_model:
            raise ValueError(f"gru_hidden x directions = {c['gru_hidden']} x {n_dirs} must "
                             f"equal d_model = {d_model} (residual + unchanged fusion/head)")

        # ─── Pre-GRU pool ──────────────────────────────────────────────────
        self.pre_rnn_pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)

        # ─── Bidirectional GRU (replaces the transformer encoder) ──────────
        # One single-layer nn.GRU per layer with nn.Dropout in between, NOT
        # nn.GRU(num_layers=2, dropout=p): on CUDA that inter-layer dropout runs in
        # cuDNN's dropout state, which ATen caches per device and seeds once per
        # process, so masks would depend on what trained earlier in the same worker.
        # Same parameters and the same computation otherwise.
        self.gru_layers = nn.ModuleList([
            nn.GRU(input_size=d_model, hidden_size=c['gru_hidden'], num_layers=1,
                   batch_first=True, bidirectional=c['gru_bidirectional'])
            for _ in range(c['gru_layers'])      # every layer's input is d_model wide
        ])
        self.rnn_dropout = nn.Dropout(c['p_rnn'])   # between layers and on the residual branch
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
        x = self.pre_rnn_pool(x)                 # (B, d_model, T'')
        x = x.transpose(1, 2)                    # (B, T'', d_model)

        h = x
        for i, gru in enumerate(self.gru_layers):
            if i > 0:
                h = self.rnn_dropout(h)
            h, _ = gru(h)                        # (B, T'', 2 x hidden) = (B, T'', d_model)
        x = self.norm(x + self.rnn_dropout(h))   # (B, T'', d_model)
        x = x.transpose(1, 2)                    # (B, d_model, T'')

        x = self.feature_fusion(x)               # (B, d_model, T'')
        x = self.global_pool(x)
        x = x.squeeze(-1)                        # (B, d_model)
        cnn_embedding = self.cnn_projection(x)
        return self.classifier(cnn_embedding)
