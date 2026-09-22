"""FINE + GRU — Option B with its transformer replaced by a BiGRU, on 50 Hz input.

FINE MTC + spatial conv -> pool -> 2-layer BiGRU -> FINE depthwise-separable conv +
ECA -> FINE-style head. The front-end is Conformer_decimated_CWT's
FineConformerHybrid (conformer_b) unchanged; the tail is GRUTail
(src/models/gru_tail.py). So FINE + GRU and Conformer B differ only in the mixer.
"""
from __future__ import annotations   # `int | None` annotations on Python < 3.10

import torch.nn as nn

from src.models.fine_blocks import MultiscaleTemporalBlock, SpatialFeatureExtraction
from src.models.gru_tail import GRUTail


def _default_cfg() -> dict:
    # Single source of truth: ExperimentConfig defaults (src/config.py)
    from src.config import ExperimentConfig
    d = ExperimentConfig(model='fine_gru', dataset_root='')
    return dict(d.fine_gru)


class FineGRU(nn.Module):
    """cfg: the `fine_gru` hyperparameter dict. Missing keys fall back to the
    ExperimentConfig defaults.
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
        # is skipped, as in Conformer B.
        self.spatial_extraction = SpatialFeatureExtraction(
            in_channels=3 * c['mtc_filters'], out_channels=d_model,
            kernel_size=c['spatial_kernel'])

        self.tail = GRUTail(d_model, n_timepoints, n_classes, embedding_dim, c)
        self.n_tokens = self.tail.n_tokens

    def forward(self, x_eeg):
        # x_eeg: (B, C, T)
        x = self.multiscale_temporal(x_eeg)      # (B, 96, T)
        x = self.spatial_extraction.conv(x)      # (B, 64, T)
        return self.tail(x)
