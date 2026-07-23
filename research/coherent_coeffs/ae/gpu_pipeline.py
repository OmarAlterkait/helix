"""GPU (torch) noise -> smart-removal -> sparsify pipeline for precomputing AE coefficients.

Fully on GPU and in COEFFICIENT SPACE (no IDWT): one forward DWT, the smart common-mode gate,
then the production universal threshold applied to the gated bands. This equals the validated
"smart_removal then sparsify" pipeline because the DWT is orthonormal (thresholding the gated
bands == sparsifying the reconstructed cleaned image), so no inverse transform is needed.

  - smart gate     : same algorithm as numpy smart_removal (per-band, per-(block,pos) k-sigma
                     masked-mean common-mode -> magnitude gate), in torch.
  - threshold      : production TPC spec replicated to match helix jax _universal_core
                     (per_band_sigma + threshold_approx): per band, sigma = median(|band|)/.6745,
                     t = scale * sigma * sqrt(2 ln L_band), hard.

The torch DWT is FFT-periodic (machine-precision PR) but a different phase than pywt, so coeffs
are not bit-identical to the numpy path; validate() checks F0 and kept-count agree with numpy.
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..'))
import cc_common as cc  # noqa: E402
import common as wd  # noqa: E402
from helix.core import backend as _be  # noqa: E402
_be.set_backend('torch')
from helix.core.wavelet_ops_torch import _wavedec, _waverec  # noqa: E402
from helix.tpc.config import DetectorConfig  # noqa: E402

GS = cc.GROUP_SIZE
CFG = DetectorConfig(group_size=64)
SCALE = float(CFG.threshold_spec().scale)          # production kappa (1.0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def smart_gate_bands(noisy, wavelet=cc.WAVELET, level=cc.LEVEL, kgate=4.0, ksig=3.0):
    """torch smart common-mode gate -> list of GATED (cleaned) bands [cA, cD_L, ..., cD_1]."""
    nw = noisy.shape[0]
    bands = _wavedec(noisy, wavelet, level)
    nblk = (nw + GS - 1) // GS
    blk_idx = torch.clamp(torch.arange(nw, device=noisy.device) // GS, max=nblk - 1)
    out = []
    for b in bands:
        Lb = b.shape[-1]
        M = torch.empty((nblk, Lb), device=b.device, dtype=b.dtype)
        for g in range(nblk):
            lo, hi = g * GS, min((g + 1) * GS, nw)
            blk = b[lo:hi]
            med = blk.median(0).values
            resid = blk - med
            sg = (resid.abs().median() / 0.6745).clamp_min(1e-6)
            uf = (resid.abs() <= ksig * sg)
            nuf = uf.sum(0)
            mean = (blk * uf).sum(0) / nuf.clamp_min(1)
            M[g] = torch.where(nuf > 0, mean, med)
        sigc = (M.abs().median() / 0.6745).clamp_min(1e-6)
        Mc = torch.where(M.abs() < kgate * sigc, M, torch.zeros_like(M))
        out.append(b - Mc[blk_idx])
    return out


def threshold_bands(bands, scale=SCALE):
    """Production universal hard threshold per band (per_band_sigma + threshold_approx)."""
    out = []
    for b in bands:
        sigma = (b.abs().median() / 0.6745).clamp_min(1e-12)
        t = scale * sigma * float(np.sqrt(2.0 * np.log(max(b.shape[-1], 2))))
        out.append(torch.where(b.abs() >= t, b, torch.zeros_like(b)))
    return out


def process(noisy_t, kgate=4.0):
    """Full GPU coeff-space pipeline. noisy_t: (nw,nt) torch tensor. Returns thresholded bands."""
    return threshold_bands(smart_gate_bands(noisy_t, kgate=kgate))


def to_sparse(bands):
    """Per band -> (coords (nnz,2) int32 [wire,pos], vals (nnz,) float32) of survivors."""
    sp = []
    for b in bands:
        nz = torch.nonzero(b, as_tuple=False)            # (nnz, 2)
        sp.append((nz.to(torch.int32).cpu().numpy(), b[nz[:, 0], nz[:, 1]].cpu().numpy()))
    return sp


def _f0(recon, clean):
    sig = np.abs(clean) > 0
    return 1.0 - float(np.abs(recon - clean)[sig].sum()) / max(float(np.abs(clean)[sig].sum()), 1e-9)


def validate(events=(0, 5, 10), planes=('U', 'V', 'Y')):
    """GPU torch (fused, coeff-space) vs validated numpy smart_removal+sparsify (F0, n_kept)."""
    import smart as sm
    from helix.core import backend as be2
    from helix.core import wavelet as cw
    print(f"device={DEV}  scale(kappa)={SCALE}")
    print(f"  {'pl/ev':>7} | {'F0_np':>7} {'F0_gpu':>7} | {'kept_np':>8} {'kept_gpu':>8}  {'ratio':>5}")
    for p in planes:
        for e in events:
            s, c, i = cc.components(p, e)
            noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            be2.set_backend('numpy')
            cl_np, _ = sm.smart_removal(noisy, kgate=4.0)
            r_np = cw.sparsify(cl_np, wavelet=CFG.wavelet, level=CFG.dwt_level, mode=CFG.dwt_mode,
                               threshold=CFG.threshold_spec())
            rec_np = np.asarray(cw.reconstruct(r_np, noisy.shape[-1]))
            be2.set_backend('torch')
            nt_ = torch.as_tensor(noisy, dtype=torch.float32, device=DEV)
            tb = process(nt_)
            kept_gpu = int(sum(int(torch.count_nonzero(b)) for b in tb))
            rec_t = np.asarray(_waverec(tb, cc.WAVELET)[..., :noisy.shape[-1]].cpu())
            print(f"  {p}/{e:>4} | {_f0(rec_np, s):>7.4f} {_f0(rec_t, s):>7.4f} | "
                  f"{r_np.n_kept:>8} {kept_gpu:>8}  {kept_gpu/max(r_np.n_kept,1):>5.2f}")


if __name__ == '__main__':
    validate()
