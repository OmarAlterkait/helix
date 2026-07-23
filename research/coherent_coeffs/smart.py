"""Smart coherent removal: level-aware common-mode gating across wires & blocks.

The figure (fig8) shows coherent is a SMALL, DENSE block-constant common-mode
(present in every block at ~sigma_coh per level), while signal is a LARGE, SPARSE
common-mode (only at tracks). Naive subtraction of the block common-mode injects
the signal common-mode (F0 collapses). Smart fix, per band (level-aware):

  1. across wires (within block): robust common-mode m[blk,pos] = k-sigma masked
     mean over the 64 wires (exclude per-position signal-outlier wires).
  2. across blocks: estimate the coherent SCALE sigma_coh = MAD over all (blk,pos)
     of m (robust to the sparse signal). This is the per-level amplitude the simple
     mask ignored.
  3. GATE: coherent_est = m where |m| < kgate*sigma_coh  (consistent with coherent),
     else 0 (a large common-mode is signal -> keep it). Subtract the gated estimate.

So we remove the small dense coherent everywhere and never inject the large sparse
signal common-mode. Compared with helix and the naive coeff-space remover.
"""
import os
import sys
import numpy as np
import pywt
import cc_common as cc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import common as wd  # noqa: E402
from helix.core import backend as _be  # noqa: E402
_be.set_backend('numpy')
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402
GS = cc.GROUP_SIZE


def block_common_mode(band, nw, ksig=3.0, return_occ=False):
    """Masked-mean common-mode per (block,pos): (nblk, L). Excludes per-position wires
    deviating > ksig*MAD (signal outliers). return_occ also returns the per-(block,pos)
    signal-occupancy fraction (flagged wires / block size) for the reliability map."""
    nblk = cc.n_groups(nw)
    L = band.shape[-1]
    M = np.zeros((nblk, L), np.float32)
    occ = np.zeros((nblk, L), np.float32) if return_occ else None
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        blk = band[lo:hi]
        med = np.median(blk, axis=0)
        resid = blk - med
        sg = max(np.median(np.abs(resid)) / 0.6745, 1e-6)
        uf = np.abs(resid) <= ksig * sg
        nuf = uf.sum(0)
        mean = (blk * uf).sum(0) / np.maximum(nuf, 1)
        M[g] = np.where(nuf > 0, mean, med)
        if return_occ:
            occ[g] = 1.0 - nuf / (hi - lo)
    if return_occ:
        return M, occ
    return M


def broadcast_blocks(M, nw):
    nblk, L = M.shape
    idx = np.minimum(np.arange(nw) // GS, nblk - 1)
    return M[idx]


def occupancy_map(occ_bands, nw, nt):
    """Reduce per-band signal-occupancy to a (nblk, nt) reliability proxy: max over bands,
    each band upsampled to the time axis. High = signal-dense = common-mode less reliable (#2)."""
    nblk = occ_bands[0].shape[0]
    out = np.zeros((nblk, nt), np.float32)
    for occ in occ_bands:
        Lb = occ.shape[1]
        f = max(int(round(nt / Lb)), 1)
        up = np.repeat(occ, f, axis=1)
        if up.shape[1] < nt:
            up = np.pad(up, ((0, 0), (0, nt - up.shape[1])), mode='edge')
        out = np.maximum(out, up[:, :nt])
    return out


def smart_removal(noisy, wavelet=cc.WAVELET, level=cc.LEVEL, mode=cc.MODE,
                  kgate=4.0, ksig=3.0, gate_soft=False, return_info=False):
    """Coefficient-space level-aware gated common-mode removal (the production method).

    kgate default = 4.0 (the value every fig/eval here used: smart_figs, oracle, de_*,
    induction) — minimizes leftover coherent. (k~3 is the max-F0 point but leaves small
    coherent strips; see RESULTS 6c.)

    return_info=True also returns {'occ_map'}: a per-(block,tick) signal-occupancy reliability
    map (high = dense = common-mode least reliable), computed ~free from the gate's clean-wire
    counts. (Two variants were tested and dropped -- noise-anchored sigma was a no-op; a coeff-
    space spatial mask-dilation recovered <20% of de2's edge at a coeff cost. See RESULTS 6p.)
    """
    nw, nt = noisy.shape
    bands = pywt.wavedec(noisy.astype(np.float32), wavelet, level=level, mode=mode, axis=-1)
    est = []
    occ_bands = []
    for b in bands:
        if return_info:
            M, occ = block_common_mode(b, nw, ksig, return_occ=True)
            occ_bands.append(occ)
        else:
            M = block_common_mode(b, nw, ksig)
        sigc = max(float(np.median(np.abs(M)) / 0.6745), 1e-6)   # coherent scale (robust)
        t = kgate * sigc
        if gate_soft:
            Mc = np.sign(M) * np.minimum(np.abs(M), t)       # clip large (signal) to +/-t
        else:
            Mc = np.where(np.abs(M) < t, M, 0.0)             # keep small (coherent), drop large
        est.append(broadcast_blocks(Mc, nw))
    cleaned = pywt.waverec([b - e for b, e in zip(bands, est)], wavelet, mode=mode, axis=-1)[..., :nt]
    coh_hat = pywt.waverec(est, wavelet, mode=mode, axis=-1)[..., :nt]
    cleaned = cleaned.astype(np.float32)
    coh_hat = coh_hat.astype(np.float32)
    if return_info:
        return cleaned, coh_hat, {'occ_map': occupancy_map(occ_bands, nw, nt)}
    return cleaned, coh_hat


def metrics(cleaned, signal, coherent, coh_hat):
    sig = np.abs(signal) > 0
    tc = float(np.abs(signal)[sig].sum())
    f0 = 1.0 - float(np.abs(cleaned - signal)[sig].sum()) / max(tc, 1e-9)
    nrms = float(np.sqrt(np.mean((cleaned - signal)[~sig] ** 2)))
    cl = float(np.sqrt(np.mean((coh_hat - coherent) ** 2)))
    return f0, nrms, cl


def main():
    ptype = sys.argv[1] if len(sys.argv) > 1 else 'Y'
    n_ev = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    events = list(range(0, n_ev * 37, 37))
    cfg = DetectorConfig(group_size=64)

    print(f"=== plane {ptype}, {n_ev} events, {cc.WAVELET} L{cc.LEVEL} ===")
    methods = {'raw': None, 'helix': 'helix'}
    for kg in (3.0, 4.0, 5.0, 6.0):
        methods[f'smart_hard_k{kg}'] = ('hard', kg)

    agg = {k: [] for k in methods}
    for e in events:
        signal, coherent, intrinsic = cc.components(ptype, e)
        noisy = wd.digitize(signal + coherent + intrinsic, cc.PLANES[ptype]['pedestal'])
        agg['raw'].append(metrics(noisy, signal, coherent, np.zeros_like(coherent)))
        hel = np.asarray(remove_coherent(noisy, cfg))
        agg['helix'].append(metrics(hel, signal, coherent, noisy - hel))
        for k, spec in methods.items():
            if not isinstance(spec, tuple):
                continue
            kind, kg = spec
            cl_img, hc = smart_removal(noisy, kgate=kg, gate_soft=(kind == 'soft'))
            agg[k].append(metrics(cl_img, signal, coherent, hc))

    print(f"   {'method':>18}  {'F0':>8}  {'noise_rms':>10}  {'coh_left':>9}")
    print(f"   {'-'*18}  {'-'*8}  {'-'*10}  {'-'*9}")
    for k in methods:
        f0, nr, cl = np.array(agg[k]).mean(0)
        print(f"   {k:>18}  {f0:>8.4f}  {nr:>10.3f}  {cl:>9.4f}")


if __name__ == '__main__':
    main()
