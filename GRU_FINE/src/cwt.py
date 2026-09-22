"""Morlet-scalogram front-end for fine_gru_cwt (fixed, zero trainable parameters).

Ported from CWT-modified/fine_mi.py (kernels, iterated reflect padding, FFT
convolution, log-magnitude, per-(channel, frequency) z-score). Same 0.5-3 Hz bank,
24 log-spaced rows, n_cycles 3 -> 7, evaluated on the DECIMATED 50 Hz signal.

One generalization: CWT-modified only used the 4000 ms window (200 samples). Here
every window length is transformed on its own crop, (N, C, L) -> (N, C, F, L), with
both edges reflect-padded. The scalogram of a window therefore never sees samples
after the window ends, so the 800 ms result is a true 800 ms decision.

Caveat worth knowing when reading results: the 0.5 Hz wavelet is 479 samples long
(9.6 s at 50 Hz), longer than every window, so the low rows are dominated by the
reflected padding. That is inherent to a 0.5 Hz wavelet on a <= 4 s window.
"""
import numpy as np
import torch


def make_morlet_kernels(freqs, n_cycles, fs, trunc):
    """Complex Morlet kernels, zero-mean and L1-normalised.

    L1 (not L2) normalisation gives a flat amplitude response across frequency.
    Returns (psi_real, psi_imag), each (F, K) float64 with K odd.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    n_cycles = np.broadcast_to(np.asarray(n_cycles, dtype=np.float64), freqs.shape)
    sigma_t = n_cycles / (2.0 * np.pi * freqs)
    half = int(np.ceil(trunc * sigma_t.max() * fs))
    t = np.arange(-half, half + 1) / fs
    psi = np.zeros((freqs.size, t.size), dtype=np.complex128)
    for i, (f, s) in enumerate(zip(freqs, sigma_t)):
        w = np.exp(-(t ** 2) / (2.0 * s ** 2)) * np.exp(2j * np.pi * f * t)
        w = w - w.mean()                    # admissibility: zero mean
        w = w / np.sum(np.abs(w))           # L1 normalise
        psi[i] = w
    return psi.real.copy(), psi.imag.copy()


def reflect_pad(x, left, right):
    """Reflect-pad the last axis by (left, right), iterating when a pad exceeds the
    signal length (torch's 'reflect' requires pad < length). The 0.5 Hz wavelet
    needs 239 samples per side, more than any window here."""
    while left > 0 or right > 0:
        n = x.shape[-1] - 1
        l, r = min(left, n), min(right, n)
        x = torch.nn.functional.pad(x, (l, r), mode='reflect')
        left, right = left - l, right - r
    return x


class MorletCWT:
    """Callable: (N, C, L) z-scored numpy array -> (N, C, F, L) float32 log-magnitude."""

    def __init__(self, cwt_cfg: dict, fs: int):
        self.cfg = dict(cwt_cfg)
        self.fs = int(fs)
        self.n_freqs = int(cwt_cfg['n_freqs'])
        self.freqs = np.logspace(np.log10(cwt_cfg['fmin']), np.log10(cwt_cfg['fmax']),
                                 self.n_freqs)
        self.n_cycles = np.linspace(cwt_cfg['n_cycles_min'], cwt_cfg['n_cycles_max'],
                                    self.n_freqs)
        self.sigma_t = self.n_cycles / (2 * np.pi * self.freqs)
        self.psi_r, self.psi_i = make_morlet_kernels(self.freqs, self.n_cycles, self.fs,
                                                     cwt_cfg['trunc'])
        self.K = self.psi_r.shape[1]        # 479 samples at 50 Hz
        self.pad = self.K // 2              # 239
        self.log_eps = float(cwt_cfg['log_eps'])
        self.chunk = int(cwt_cfg['chunk'])
        self._kernels = {}

    def _kernels_on(self, device):
        key = str(device)
        if key not in self._kernels:
            self._kernels[key] = (torch.from_numpy(self.psi_r).float().to(device),
                                  torch.from_numpy(self.psi_i).float().to(device))
        return self._kernels[key]

    def __call__(self, x_np, device, chunk=None):
        """CONTRACT: input must already be per-channel z-scored. log_eps is an ABSOLUTE
        floor, so on volt-scale data it would dominate the log."""
        if x_np.ndim != 3:
            raise ValueError(f"expected (N, C, L), got shape {x_np.shape}")
        chunk = self.chunk if chunk is None else int(chunk)
        n, c, L = x_np.shape
        psi_r, psi_i = self._kernels_on(device)

        padded_len = L + 2 * self.pad
        n_fft = 1
        while n_fft < padded_len + self.K - 1:     # full linear convolution, no wrap
            n_fft *= 2
        # Output sample j of the window sits at index j + 2*pad of the full convolution
        # (pad from the left reflection + pad from the kernel centre).
        lo, hi = 2 * self.pad, 2 * self.pad + L

        out = np.empty((n, c, self.n_freqs, L), dtype=np.float32)
        with torch.no_grad():
            kr = torch.fft.rfft(psi_r, n=n_fft)                     # (F, n/2+1)
            ki = torch.fft.rfft(psi_i, n=n_fft)
            for s in range(0, n, chunk):
                xb = torch.from_numpy(np.ascontiguousarray(x_np[s:s + chunk])).float().to(device)
                xb = reflect_pad(xb, self.pad, self.pad)
                xf = torch.fft.rfft(xb, n=n_fft).unsqueeze(-2)     # (b, C, 1, n/2+1)
                re = torch.fft.irfft(xf * kr, n=n_fft)[..., lo:hi]
                im = torch.fft.irfft(xf * ki, n=n_fft)[..., lo:hi]
                mag = torch.log(torch.sqrt(re * re + im * im) + self.log_eps)
                out[s:s + chunk] = mag.cpu().numpy()
                del xb, xf, re, im, mag
        return out


_CWT_CACHE = {}


def get_cwt(cfg) -> MorletCWT:
    """One MorletCWT per (bank settings, sampling rate), shared across folds."""
    key = (tuple(sorted(cfg.cwt.items())), int(cfg.sampling_rate))
    if key not in _CWT_CACHE:
        _CWT_CACHE[key] = MorletCWT(cfg.cwt, cfg.sampling_rate)
    return _CWT_CACHE[key]


def scalogram_normalize(s_train, s_val, s_test):
    """Per-(channel, frequency) z-score, statistics from the TRAINING fold only."""
    mu = s_train.mean(axis=(0, 3), keepdims=True)
    sd = s_train.std(axis=(0, 3), keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (s_train - mu) / sd, (s_val - mu) / sd, (s_test - mu) / sd
