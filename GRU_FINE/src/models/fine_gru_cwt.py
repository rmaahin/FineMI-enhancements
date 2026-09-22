"""FINE + GRU + CWT — FINE + GRU with a Morlet-scalogram front-end, on 50 Hz input.

Input is the (C, F, T) = (62, 24, T) log-magnitude scalogram from src/cwt.py. The
front-end is Conformer_decimated_CWT's FineConformerHybridCWT (conformer_b_cwt)
unchanged:

  MTC 2D        Conv2d (1, k), k = 7 / 15 / 31: temporal only, EEG channels are the
                in-channels, 3 x 32 filters                    -> (B, 96, F, T)
  spatial 2D    Conv2d 3x3 + BN + ReLU (mixes frequency and time) -> (B, 64, F, T)
  freq collapse Conv2d (F, 1) + BN + ReLU over all 24 rows         -> (B, 64, T)

Everything after that is GRUTail (src/models/gru_tail.py): pool 3/3 -> 2-layer BiGRU
-> DS-conv + ECA -> head. So FINE + GRU + CWT and Conformer B + CWT differ only in
the mixer, and FINE + GRU + CWT and FINE + GRU only in the front-end.
"""
from __future__ import annotations   # `int | None` annotations on Python < 3.10

import torch.nn as nn

from src.models.fine_blocks import MultiscaleTemporalBlock2d
from src.models.gru_tail import GRUTail


def _default_cfg() -> dict:
    # Single source of truth: ExperimentConfig defaults (src/config.py)
    from src.config import ExperimentConfig
    d = ExperimentConfig(model='fine_gru_cwt', dataset_root='')
    return dict(d.fine_gru_cwt)


class FineGRUCWT(nn.Module):
    """cfg: the `fine_gru_cwt` hyperparameter dict. Missing keys fall back to the
    ExperimentConfig defaults.
    """

    def __init__(self, n_channels: int, n_timepoints: int, n_freqs: int,
                 n_classes: int = 2, embedding_dim: int | None = None,
                 cfg: dict | None = None):
        super().__init__()
        c = {**_default_cfg(), **(cfg or {})}
        if embedding_dim is None:
            embedding_dim = c['embedding_dim']

        d_model = c['spatial_filters']
        k = c['spatial_kernel']
        self.n_freqs = int(n_freqs)

        # ─── 2D MTC block ──────────────────────────────────────────────────
        self.multiscale_temporal = MultiscaleTemporalBlock2d(
            in_channels=n_channels, out_channels_per_branch=c['mtc_filters'],
            kernels=tuple(c['mtc_kernels']))

        # ─── 2D spatial extraction (k x k over frequency x time) ───────────
        self.spatial_extraction = nn.Sequential(
            nn.Conv2d(3 * c['mtc_filters'], d_model, kernel_size=k, padding=k // 2, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU())

        # ─── Collapse the frequency axis ───────────────────────────────────
        self.freq_collapse = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(self.n_freqs, 1), bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU())

        self.tail = GRUTail(d_model, n_timepoints, n_classes, embedding_dim, c)
        self.n_tokens = self.tail.n_tokens

    def forward(self, x):
        # x: (B, C, F, T)
        if x.dim() != 4 or x.size(2) != self.n_freqs:
            raise ValueError(f"expected (B, C, {self.n_freqs}, T) scalograms, "
                             f"got {tuple(x.shape)}")
        x = self.multiscale_temporal(x)          # (B, 96, F, T)
        x = self.spatial_extraction(x)           # (B, 64, F, T)
        x = self.freq_collapse(x)                # (B, 64, 1, T)
        x = x.squeeze(2)                         # (B, 64, T)
        return self.tail(x)
