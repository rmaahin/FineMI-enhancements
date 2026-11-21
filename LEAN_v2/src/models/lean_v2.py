"""LeanV2: one configurable network covering LEAN and every LEAN_v2 variant.

The `arch` dict (src/config.py) picks each stage:

  front     'per_electrode'  LEAN Eq. (1): per-electrode temporal filters 7/15/31 ->
                             1x1 electrode mix to 16 per branch -> BN -> act  (48 maps)
            'shared'         EEGNet-style: 4 temporal filters per branch shared by all
                             electrodes (Conv2d (1,k)) -> BN -> [depth attention] ->
                             depthwise spatial conv (62,1), D = 2 -> BN -> act (24 maps)
  fusion    'lean'           LEAN Eq. (2): conv k3 to `width` -> BN -> act -> max pool 3
            'separable'      EEGNet block 2: avg pool 2 -> depthwise conv k8 -> 1x1 to
                             `width` -> BN -> act -> avg pool 4  (160 ms tokens)
  context   'transformer'    LEAN Eqs. (3)-(4): learnable positional embedding, one
                             pre-norm layer (4 heads, FFN 64, GELU), final LayerNorm
            'tcn'            EEG-TCNet: 3 causal residual blocks, kernel 4, dilations
                             1/2/4 (receptive field 43 tokens)
  refine    (always)         LEAN Sec. 2.5: depthwise-separable conv k5 + channel gate
  readout   'gap_mlp'        LEAN: global average pool -> 32 -> 32 -> 32 -> 2 MLP
            'last_step'      last time step -> one linear layer (EEG-TCNet)
            'flatten'        all time steps -> one linear layer (EEGNet)

`act` (relu | elu), `max_norm` (spatial filters 1.0, linear readout 0.25) and `p_conv`
(dropout after each pooling step) are the R5 regularisation switches.

With LEAN_ARCH the layers are created in exactly the order of
'Fine Conformer Fixation Cross Fix'/src/models/conformer_b_slim.py, so the same seed
gives the same initial weights; the preflight checks the parameter layout against
LEAN's.

forward(x, return_features=True) also returns the vector the final linear layer reads
(the SupCon embedding): the pooled features (gap_mlp), the last step, or the flattened
map. Output: logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import (make_act, Conv2dMaxNorm, LinearMaxNorm, DSConv1d,
                               EfficientChannelAttention, PerElectrodeMTC, DepthAttention,
                               TemporalBlock)


def n_tokens(arch: dict, n_timepoints: int) -> int:
    """Token count after the fusion layer's pooling."""
    if arch['fusion'] == 'lean':
        p = arch['lean_pool']
        return (n_timepoints - p) // p + 1 if n_timepoints >= p else 0
    return (n_timepoints // arch['pool1']) // arch['pool2']


def _drop(p):
    # nn.Identity rather than Dropout(0), so p = 0 consumes no random numbers
    return nn.Dropout(p) if p > 0 else nn.Identity()


class LeanV2(nn.Module):

    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 cfg: dict | None = None, max_timepoints: int | None = None):
        super().__init__()
        a = dict(cfg)
        self.a = a
        act, w = a['act'], a['width']
        mn_spatial = a['max_norm_spatial'] if a['max_norm'] else None
        mn_linear = a['max_norm_linear'] if a['max_norm'] else None
        max_timepoints = max(max_timepoints or n_timepoints, n_timepoints)

        self.n_tokens = n_tokens(a, n_timepoints)
        if self.n_tokens < 1:
            raise ValueError(f"window of {n_timepoints} samples is too short for the "
                             f"{a['fusion']} fusion pooling")

        # ─── Front end ─────────────────────────────────────────────────────
        if a['temporal'] == 'per_electrode':
            self.front = PerElectrodeMTC(n_channels, a['mtc_filters'], a['mtc_kernels'],
                                         act=act, max_norm=mn_spatial)
            front_out = 3 * a['mtc_filters']
        else:
            f1, d = a['shared_filters'], a['depth_mult']
            self.temporal = nn.ModuleList([
                nn.Conv2d(1, f1, kernel_size=(1, int(k)), padding='same', bias=False)
                for k in a['mtc_kernels']])
            n_maps = f1 * len(a['mtc_kernels'])
            self.temporal_bn = nn.BatchNorm2d(n_maps)
            self.depth_att = DepthAttention(a['da_kernel']) if a['depth_attention'] else None
            self.spatial = Conv2dMaxNorm(n_maps, n_maps * d, kernel_size=(n_channels, 1),
                                         groups=n_maps, bias=False, max_norm=mn_spatial)
            self.spatial_bn = nn.BatchNorm2d(n_maps * d)
            self.spatial_act = make_act(act)
            front_out = n_maps * d

        # ─── Fusion + pooling into tokens ──────────────────────────────────
        if a['fusion'] == 'lean':
            k = a['lean_fusion_kernel']
            self.fusion = nn.Sequential(
                nn.Conv1d(front_out, w, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(w),
                make_act(act))
            self.fusion_pool = nn.MaxPool1d(kernel_size=a['lean_pool'], stride=a['lean_pool'])
        else:
            self.pool1 = nn.AvgPool1d(a['pool1'])
            self.drop1 = _drop(a['p_conv'])
            self.fusion = nn.Sequential(
                nn.Conv1d(front_out, front_out, kernel_size=a['sep_kernel'], padding='same',
                          groups=front_out, bias=False),
                nn.Conv1d(front_out, w, kernel_size=1, bias=False),
                nn.BatchNorm1d(w),
                make_act(act))
            self.fusion_pool = nn.AvgPool1d(a['pool2'])
        self.drop2 = _drop(a['p_conv'])

        # ─── Temporal context ──────────────────────────────────────────────
        if a['context'] == 'transformer':
            max_tokens = max(n_tokens(a, max_timepoints), self.n_tokens)
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_tokens, w))
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=w, nhead=a['n_heads'], dim_feedforward=a['d_ff'],
                    dropout=a['p_att'], activation='gelu', batch_first=True,
                    norm_first=True)])
            self.norm = nn.LayerNorm(w)
        else:
            self.tcn = nn.ModuleList([
                TemporalBlock(w, a['tcn_kernel'], 2 ** i, a['p_tcn'], act=act)
                for i in range(a['tcn_blocks'])])

        # ─── Feature refinement (DS conv + channel gate) ───────────────────
        self.refine = nn.Sequential(
            DSConv1d(w, a['refine_kernel'], act=act),
            EfficientChannelAttention(w, reduction=a['eca_reduction']))

        # ─── Readout ───────────────────────────────────────────────────────
        if a['readout'] == 'gap_mlp':
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.cnn_projection = nn.Sequential(
                nn.Linear(w, a['embedding_dim']), nn.ReLU(), nn.Dropout(a['p_proj']))
            self.classifier = nn.Sequential(
                nn.Linear(a['embedding_dim'], a['classifier_hidden']), nn.ReLU(),
                nn.Dropout(a['p_head']), nn.Linear(a['classifier_hidden'], n_classes))
            self.feature_dim = w
        else:
            in_dim = w if a['readout'] == 'last_step' else w * self.n_tokens
            self.classifier = LinearMaxNorm(in_dim, n_classes, max_norm=mn_linear)
            self.feature_dim = in_dim

    def forward(self, x, return_features: bool = False):
        a = self.a
        # x: (B, C, T)
        if a['temporal'] == 'per_electrode':
            h = self.front(x)                                     # (B, 48, T)
        else:
            x4 = x.unsqueeze(1)                                   # (B, 1, C, T)
            h = torch.cat([conv(x4) for conv in self.temporal], dim=1)   # (B, 12, C, T)
            h = self.temporal_bn(h)
            if self.depth_att is not None:
                h = self.depth_att(h)
            h = self.spatial_act(self.spatial_bn(self.spatial(h)))       # (B, 24, 1, T)
            h = h.squeeze(2)                                      # (B, 24, T)

        if a['fusion'] == 'lean':
            h = self.fusion_pool(self.fusion(h))                  # (B, w, N)
        else:
            h = self.fusion(self.drop1(self.pool1(h)))
            h = self.fusion_pool(h)                               # (B, w, N)
        h = self.drop2(h)

        if a['context'] == 'transformer':
            z = h.transpose(1, 2)                                 # (B, N, w)
            z = z + self.pos_embedding[:, :z.size(1)]
            for layer in self.layers:
                z = layer(z)
            h = self.norm(z).transpose(1, 2)                      # (B, w, N)
        else:
            for block in self.tcn:
                h = block(h)

        h = self.refine(h)                                        # (B, w, N)

        if a['readout'] == 'gap_mlp':
            g = self.global_pool(h).squeeze(-1)                   # (B, w)
            logits = self.classifier(self.cnn_projection(g))
        else:
            g = h[:, :, -1] if a['readout'] == 'last_step' else h.flatten(1)
            logits = self.classifier(g)
        return (logits, g) if return_features else logits
