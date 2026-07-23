"""Probe: confirm coherent within-block coefficient identity + per-level stats.

Answers the central questions before we build figures:
  Q1  Within one block, are the coherent wavelet coefficients EXACTLY identical
      across the 64 wires? (Expected yes: same waveform -> same linear DWT.)
  Q2  How does each component (signal/coherent/intrinsic) distribute its energy
      across decomposition levels?  (Expected: coherent -> coarse, intrinsic ->
      spread/finer, signal -> mid.)
  Q3  Cross-block correlation of coherent coefficients ("bleeding" via beta).
"""
import sys
import numpy as np
import cc_common as cc


def q1_within_block_identity(coh_bands, nw):
    labels = cc.band_labels()
    ng = cc.full_groups(nw)
    print(f"\n[Q1] within-block coherent coefficient identity ({ng} full blocks)")
    worst = 0.0
    for b, lab in zip(coh_bands, labels):
        # for each full group, max abs deviation of each wire from wire 0 of group
        devs = []
        for g in range(ng):
            blk = b[g * cc.GROUP_SIZE:(g + 1) * cc.GROUP_SIZE]      # (64, len)
            dev = np.max(np.abs(blk - blk[0:1]))
            devs.append(dev)
        m = float(np.max(devs))
        worst = max(worst, m)
        scale = float(np.median(np.abs(b)))
        print(f"   {lab:>3}: max |coeff - coeff_wire0| over all blocks = {m:.3e}"
              f"   (band median|coeff| = {scale:.3e})")
    print(f"   --> worst deviation anywhere = {worst:.3e}  "
          f"({'IDENTICAL to float precision' if worst < 1e-3 else 'NOT identical'})")


def per_level_stats(name, bands):
    labels = cc.band_labels()
    print(f"\n[Q2] per-level statistics — {name}")
    print(f"   {'band':>4} {'len':>6} {'var':>12} {'energy%':>9} {'MADsig':>9} {'kurtosis':>9}")
    tot_e = sum(float(np.sum(b.astype(np.float64) ** 2)) for b in bands)
    rows = []
    for b, lab in zip(bands, labels):
        v = float(np.var(b))
        e = float(np.sum(b.astype(np.float64) ** 2))
        ms = float(np.median(np.abs(b)) / 0.6745)
        f = b.ravel().astype(np.float64)
        mu = f.mean(); sd = f.std()
        kurt = float(np.mean(((f - mu) / (sd + 1e-30)) ** 4)) if sd > 0 else 0.0
        rows.append((lab, b.shape[-1], v, 100 * e / max(tot_e, 1e-30), ms, kurt))
        print(f"   {lab:>4} {b.shape[-1]:>6} {v:>12.4f} {100*e/max(tot_e,1e-30):>8.1f}% {ms:>9.4f} {kurt:>9.2f}")
    return rows


def q3_cross_block(coh_bands, nw):
    """Correlation of a group's coherent coeff vector with neighbor groups."""
    ng = cc.full_groups(nw)
    flat = cc.flat_coeffs(coh_bands)                                # (nw, n_coeffs)
    # representative coeff vector per group = wire 0 of each group (all identical)
    reps = np.stack([flat[g * cc.GROUP_SIZE] for g in range(ng)])    # (ng, n_coeffs)
    reps = reps - reps.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(reps, axis=1, keepdims=True)
    repn = reps / np.maximum(norm, 1e-30)
    C = repn @ repn.T                                               # (ng, ng) corr
    print(f"\n[Q3] cross-block coherent correlation ('bleeding', beta=0.15)")
    for d in range(0, 4):
        vals = [C[g, g + d] for g in range(ng - d)]
        print(f"   lag {d}: mean corr = {np.mean(vals):+.3f}  (std {np.std(vals):.3f})")
    return C


def main():
    ptype = sys.argv[1] if len(sys.argv) > 1 else 'Y'
    event = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(f"=== plane {ptype}, event {event}, {cc.WAVELET} L{cc.LEVEL} {cc.MODE} ===")
    signal, coherent, intrinsic = cc.components(ptype, event)
    nw = signal.shape[0]
    print(f"shape (nw,T) = {signal.shape}; full 64-blocks = {cc.full_groups(nw)}, "
          f"partial tail = {nw % cc.GROUP_SIZE} wires")
    print(f"RMS: signal(on-sig)={np.abs(signal)[signal!=0].std():.3f}  "
          f"coherent={coherent.std():.3f}  intrinsic={intrinsic.std():.3f}")

    sig_b = cc.dwt_bands(signal)
    coh_b = cc.dwt_bands(coherent)
    int_b = cc.dwt_bands(intrinsic)

    q1_within_block_identity(coh_b, nw)
    per_level_stats('signal', sig_b)
    per_level_stats('coherent', coh_b)
    per_level_stats('intrinsic', int_b)
    q3_cross_block(coh_b, nw)


if __name__ == '__main__':
    main()
