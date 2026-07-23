"""Band-split hybrid: low-freq coherent from SAMPLE space, detail bands from COEFFICIENTS.

smart's only weakness is the approx band A4 in signal blocks: a coarse A4 position bins
~16 ticks, during which a track crosses several wires -> high per-position occupancy ->
median fooled -> gate leaves coherent. But per TICK the occupancy is LOW (a track hits
few wires at any one tick). So estimate the LOW-FREQUENCY coherent (A4, +maybe D4) from a
sample-space per-tick masked-mean (helix's strength), and the DETAIL bands from the
coefficient masked-mean + gate (smart's strength, near-floor). split_band = #low-freq
bands taken from the sample estimate.
"""
import os
import sys
import numpy as np
import pywt
import cc_common as cc
import smart as sm
from smarter import sample_signal_mask

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import common as wd  # noqa: E402
from helix.core import backend as _be  # noqa: E402
_be.set_backend('numpy')
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402
GS = cc.GROUP_SIZE


def sample_coherent_est(noisy, smask):
    """Per-tick block masked-mean over non-signal wires (helix-style coherent waveform)."""
    nw, nt = noisy.shape
    nblk = cc.n_groups(nw)
    est = np.empty_like(noisy)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        uf = ~smask[lo:hi]
        nuf = uf.sum(0)
        m = (noisy[lo:hi] * uf).sum(0) / np.maximum(nuf, 1)
        med = np.median(noisy[lo:hi], axis=0)
        est[lo:hi] = np.where(nuf >= 6, m, med)[None, :]
    return est


def hybrid_removal(noisy, split_band=1, kgate=4.0, ksig=3.0):
    nw, nt = noisy.shape
    smask = sample_signal_mask(noisy, ksig, dilate=11)
    se = sample_coherent_est(noisy, smask)
    se_bands = pywt.wavedec(se.astype(np.float32), cc.WAVELET, level=cc.LEVEL, mode=cc.MODE, axis=-1)
    bands = pywt.wavedec(noisy.astype(np.float32), cc.WAVELET, level=cc.LEVEL, mode=cc.MODE, axis=-1)
    est = []
    for j, b in enumerate(bands):
        if j < split_band:                                   # low-freq from sample est
            est.append(se_bands[j])
        else:                                                # detail from coeff gate
            M = sm.block_common_mode(b, nw, ksig)
            sigc = max(float(np.median(np.abs(M)) / 0.6745), 1e-6)
            Mc = np.where(np.abs(M) < kgate * sigc, M, 0.0)
            est.append(sm.broadcast_blocks(Mc, nw))
    cleaned = pywt.waverec([b - e for b, e in zip(bands, est)], cc.WAVELET, mode=cc.MODE, axis=-1)[..., :nt]
    coh_hat = pywt.waverec(est, cc.WAVELET, mode=cc.MODE, axis=-1)[..., :nt]
    return cleaned.astype(np.float32), coh_hat.astype(np.float32)


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    events = list(range(0, n_ev * 13, 13))
    cfg = DetectorConfig(group_size=64)
    for p in ['Y', 'U', 'V']:
        agg = {'helix': [], 'smart': [], 'hybrid_s1': [], 'hybrid_s2': []}
        for e in events:
            s, c, i = cc.components(p, e)
            noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            agg['helix'].append(sm.metrics(np.asarray(remove_coherent(noisy, cfg)), s, c,
                                           noisy - np.asarray(remove_coherent(noisy, cfg))))
            cl, hc = sm.smart_removal(noisy, kgate=4.0); agg['smart'].append(sm.metrics(cl, s, c, hc))
            cl, hc = hybrid_removal(noisy, 1); agg['hybrid_s1'].append(sm.metrics(cl, s, c, hc))
            cl, hc = hybrid_removal(noisy, 2); agg['hybrid_s2'].append(sm.metrics(cl, s, c, hc))
        print(f"  plane {p}:")
        for k, v in agg.items():
            f0, nr, cohl = np.array(v).mean(0)
            print(f"     {k:>10}  F0 {f0:.4f}  noise {nr:.3f}  coh_left {cohl:.4f}")


if __name__ == '__main__':
    main()
