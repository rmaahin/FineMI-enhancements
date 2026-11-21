"""Expected parameter counts, worked out by hand from the arch dict, independently of
the model code. preflight.py and check_results.py assert that every built model has
exactly this many parameters, so a wiring mistake in src/models shows up before any
training.

Checks against earlier projects:
  LEAN_ARCH, T = 200  -> 25,752 (LEAN in the ICASSP paper)
  EEGNET50_ARCH       -> 1,656 + 16 * n_pool_out * 2 + 2 (EEGNet-experiments)
"""

C = 62        # electrodes
N_CLASSES = 2


def tokens(a: dict, T: int) -> int:
    if a['fusion'] == 'lean':
        p = a['lean_pool']
        return (T - p) // p + 1
    return (T // a['pool1']) // a['pool2']


def lean_v2_params(a: dict, T: int, T_max: int) -> int:
    """T: input length the model is built for; T_max: longest input it must accept
    (sizes the transformer's positional embedding)."""
    w = a['width']
    K = [int(k) for k in a['mtc_kernels']]

    # front end
    if a['temporal'] == 'per_electrode':
        F = a['mtc_filters']
        # per branch: per-electrode filter C*k, 1x1 mix C*F, BN 2F
        front = sum(C * k + C * F + 2 * F for k in K)
        front_out = 3 * F
    else:
        f1, d = a['shared_filters'], a['depth_mult']
        n_maps = f1 * len(K)
        front = (f1 * sum(K) + 2 * n_maps          # temporal convs + BN
                 + n_maps * d * C + 2 * n_maps * d)  # depthwise spatial + BN
        if a['depth_attention']:
            front += a['da_kernel'] + 1
        front_out = n_maps * d

    # fusion
    if a['fusion'] == 'lean':
        fusion = front_out * w * a['lean_fusion_kernel'] + 2 * w
    else:
        fusion = front_out * a['sep_kernel'] + front_out * w + 2 * w

    # context
    if a['context'] == 'transformer':
        dff = a['d_ff']
        n_max = max(tokens(a, T_max), tokens(a, T))
        context = (n_max * w                                   # positional embedding
                   + (3 * w * w + 3 * w) + (w * w + w)         # attention in/out proj
                   + (w * dff + dff) + (dff * w + w)           # feed-forward
                   + 4 * w                                     # two LayerNorms
                   + 2 * w)                                    # final LayerNorm
    else:
        k = a['tcn_kernel']
        context = a['tcn_blocks'] * (2 * w * w * k + 4 * w)

    # refinement: DS conv (w*k + BN + w*w + BN) + channel gate (two 1x1 convs, no bias)
    r = w // a['eca_reduction']
    refine = w * a['refine_kernel'] + 2 * w + w * w + 2 * w + 2 * w * r

    # readout
    if a['readout'] == 'gap_mlp':
        e, h = a['embedding_dim'], a['classifier_hidden']
        readout = (w * e + e) + (e * h + h) + (h * N_CLASSES + N_CLASSES)
    elif a['readout'] == 'last_step':
        readout = w * N_CLASSES + N_CLASSES
    else:
        readout = w * tokens(a, T) * N_CLASSES + N_CLASSES

    return front + fusion + context + refine + readout


def eegnet_params(a: dict, T: int) -> int:
    F1, D, F2 = a['F1'], a['D'], a['F2']
    n_pool_out = (T // a['pool1']) // a['pool2']
    return (F1 * a['kern_length'] + 2 * F1               # temporal conv + BN
            + D * F1 * C + 2 * D * F1                    # depthwise spatial + BN
            + D * F1 * a['sep_kern_length'] + D * F1 * F2 + 2 * F2   # separable + BN
            + F2 * n_pool_out * N_CLASSES + N_CLASSES)   # classifier


def expected_params(cfg, t_samples: int) -> int:
    """Parameters of cfg's model for a window of t_samples (crop-adjusted)."""
    T = cfg.model_samples(t_samples)
    if cfg.arch['arch'] == 'eegnet':
        return eegnet_params(cfg.arch, T)
    return lean_v2_params(cfg.arch, T, cfg.max_model_samples())


# LEAN's parameter layout, in model.parameters() order (the root module's own
# parameter, the positional embedding, comes first; then the submodules in creation
# order), from 'Fine Conformer Fixation Cross Fix'/src/models/conformer_b_slim.py.
# lean_v2_r0 must match it exactly, which together with the identical creation order
# means the same initial weights for the same seed.
LEAN_LAYOUT = (
    [(1, 66, 32),                                       # positional embedding
     (62, 1, 7), (16, 62, 1), (16,), (16,),
     (62, 1, 15), (16, 62, 1), (16,), (16,),
     (62, 1, 31), (16, 62, 1), (16,), (16,),
     (32, 48, 3), (32,), (32,),                         # spatial conv + BN
     (96, 32), (96,), (32, 32), (32,),                  # self-attention
     (64, 32), (64,), (32, 64), (32,),                  # feed-forward
     (32,), (32,), (32,), (32,),                        # norm1, norm2
     (32,), (32,),                                      # final LayerNorm
     (32, 1, 5), (32,), (32,), (32, 32, 1), (32,), (32,),   # DS conv
     (8, 32, 1), (32, 8, 1),                            # channel gate
     (32, 32), (32,),                                   # projection
     (32, 32), (32,), (2, 32), (2,)])                   # classifier
