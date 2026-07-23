"""Systematic per-level structure: same coefficients across wires, within vs
outside a block, at EVERY level of a deep decomposition.

For each band (level) we quantify, per 64-wire block:
  within_dev   max |coeff_wire - coeff_wire0| across the block (coherent)  -> 0 expected
  xblk_lag1    mean correlation of a block's coherent coeff-vector with its neighbor
  coh_cm_rms   RMS of the coherent common-mode (= the block's coherent coeff value)
  sig_cm_naive RMS of the SIGNAL common-mode = mean_over_block(signal coeff) -> the
               contamination a naive block-mean would inject
  sig_cm_mask  same after per-position k-sigma masking (exclude deviating wires)
  ratio_naive  sig_cm_naive / coh_cm_rms   (>~1 => signal swamps coherent at this level)
  ratio_mask   sig_cm_mask  / coh_cm_rms   (residual after masking)

The crossover level (ratio jumps above ~1, masking stops helping) is where signal
becomes common-mode-like and coefficient-space coherent estimation breaks down.
"""
import sys
import numpy as np
import cc_common as cc

GS = cc.GROUP_SIZE


def block_common_modes(band, nw, ksig=3.0):
    """Return (coh-agnostic) naive block-mean and k-sigma-masked block-mean per
    position, shape (n_full_blocks, L). Operates on whatever band is passed."""
    ngf = cc.full_groups(nw)
    L = band.shape[-1]
    blk = band[:ngf * GS].reshape(ngf, GS, L)
    naive = blk.mean(axis=1)
    med = np.median(blk, axis=1, keepdims=True)
    resid = blk - med
    sig = np.maximum(np.median(np.abs(resid), axis=(1, 2)) / 0.6745, 1e-6)[:, None, None]
    unflag = np.abs(resid) <= ksig * sig
    nuf = unflag.sum(axis=1)
    masked = (blk * unflag).sum(axis=1) / np.maximum(nuf, 1)
    masked = np.where(nuf > 0, masked, med[:, 0, :])
    return naive, masked


def main():
    ptype = sys.argv[1] if len(sys.argv) > 1 else 'Y'
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    wavelet = sys.argv[3] if len(sys.argv) > 3 else 'sym4'
    event = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    signal, coherent, intrinsic = cc.components(ptype, event)
    nw = signal.shape[0]
    sig_b = cc.dwt_bands(signal, wavelet, level)
    coh_b = cc.dwt_bands(coherent, wavelet, level)
    labels = cc.band_labels(level)
    ngf = cc.full_groups(nw)

    print(f"=== plane {ptype}, {wavelet} L{level}, event {event} (nw={nw}, {ngf} full blocks) ===")
    print(f"{'band':>5} {'len':>6} {'within_dev':>10} {'xblk_lag1':>10} "
          f"{'coh_cm':>8} {'sig_naive':>9} {'sig_mask':>9} {'ratio_nv':>8} {'ratio_mk':>8}")
    for bi, lab in enumerate(labels):
        sb, cb = sig_b[bi], coh_b[bi]
        L = cb.shape[-1]
        # within-block deviation (coherent)
        wdev = 0.0
        for g in range(ngf):
            blk = cb[g * GS:(g + 1) * GS]
            wdev = max(wdev, float(np.max(np.abs(blk - blk[0:1]))))
        # across-block lag-1 correlation of coherent reps
        reps = np.stack([cb[g * GS] for g in range(ngf)])
        reps = reps - reps.mean(1, keepdims=True)
        reps /= np.maximum(np.linalg.norm(reps, axis=1, keepdims=True), 1e-30)
        C = reps @ reps.T
        lag1 = float(np.mean([C[g, g + 1] for g in range(ngf - 1)])) if ngf > 1 else np.nan
        # coherent common-mode rms = the block coherent value itself
        coh_cm = float(np.sqrt(np.mean(reps_raw_rms(cb, ngf))))
        # signal common-mode (naive + masked)
        s_naive, s_mask = block_common_modes(sb, nw)
        sig_nv = float(np.sqrt(np.mean(s_naive ** 2)))
        sig_mk = float(np.sqrt(np.mean(s_mask ** 2)))
        print(f"{lab:>5} {L:>6} {wdev:>10.2e} {lag1:>10.3f} "
              f"{coh_cm:>8.3f} {sig_nv:>9.3f} {sig_mk:>9.3f} "
              f"{sig_nv/max(coh_cm,1e-9):>8.2f} {sig_mk/max(coh_cm,1e-9):>8.2f}")


def reps_raw_rms(cb, ngf):
    """Per-block coherent coeff value (wire0 of each block); return squared values."""
    return np.array([cb[g * GS] ** 2 for g in range(ngf)]).ravel()


if __name__ == '__main__':
    main()
