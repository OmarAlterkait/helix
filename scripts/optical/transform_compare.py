"""Is the DWT even the right basis? Compare DWT vs DCT vs FFT on the optical
chunks, all with noise-relative (universal) thresholding + quantization, scored
identically (event-total compression vs peak/rms). CPU/numpy, subset of events.

Usage: python scripts/optical/transform_compare.py [n_events]
"""
import sys, glob, json, math
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
import numpy as np
import pywt
from scipy.fft import dct, idct, rfft, irfft
from helix.optical import config_from_file, list_events
from helix.optical.io import read_event_chunks, pad_batch

PATH = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
NOISE = 2.566


def storage_bits_flat(coeffs2d, bits):
    """coeffs2d: (n_chunks, Ncoef) thresholded. Event-total bits (sparse/dense, per-row)."""
    n, L = coeffs2d.shape
    K = (coeffs2d != 0).sum(1)
    idx = math.ceil(math.log2(max(L, 2)))
    dense = L * bits
    per = np.where(K <= 0.5 * L, K * (bits + idx), dense)
    return per.sum()


def quant(c, bits):
    if bits >= 32:
        return c
    amax = np.abs(c).max(1, keepdims=True); amax[amax == 0] = 1
    lv = (1 << (bits - 1)) - 1
    step = amax / lv
    return np.round(c / step) * step


def universal_hard(coef, sigma, kappa):
    """coef (n, L) real; sigma scalar per row; universal hard threshold."""
    N = coef.shape[1]
    t = kappa * sigma[:, None] * np.sqrt(2 * np.log(N))
    return coef * (np.abs(coef) >= t)


def run_transform(x, lengths, which, kappa, bits, sig):
    n, Lpad = x.shape
    if which == "dwt":
        co = pywt.wavedec(x, "coif3", level=10, mode="periodization", axis=1)
        flat = np.concatenate(co, axis=1)
        th = universal_hard(flat, sig, kappa)
        thq = quant(th, bits)
        # reconstruct: split back
        sizes = [c.shape[1] for c in co]; off = np.cumsum([0] + sizes)
        col = [thq[:, off[i]:off[i + 1]] for i in range(len(co))]
        rec = pywt.waverec(col, "coif3", mode="periodization", axis=1)[:, :Lpad]
        coeffs = thq
    elif which == "dct":
        d = dct(x, type=2, norm="ortho", axis=1)     # orthonormal -> noise sigma preserved
        th = universal_hard(d, sig, kappa)
        thq = quant(th, bits)
        rec = idct(thq, type=2, norm="ortho", axis=1)
        coeffs = thq
    elif which == "fft":
        f = rfft(x, axis=1)                          # complex
        t = kappa * sig[:, None] * np.sqrt(2 * np.log(f.shape[1]))
        mag = np.abs(f)
        keep = mag >= t
        fr = f * keep
        re, im = quant(fr.real, bits), quant(fr.imag, bits)
        rec = irfft(re + 1j * im, n=Lpad, axis=1)
        coeffs = np.concatenate([re, im], axis=1)     # store both parts
    # metrics
    valid = np.arange(Lpad)[None, :] < lengths[:, None]
    xm, rm = x * valid, rec * valid
    sigmask = np.abs(x).max(1) > 10 * NOISE
    pk = xm.min(1)
    peak = np.abs(rm.min(1) - pk) / (np.abs(pk) + 1e-9)
    storage = storage_bits_flat(coeffs, bits)
    comp = float((lengths * 16).sum() / max(storage, 1))
    return comp, float(peak[sigmask].mean())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cfg0 = config_from_file(PATH)
    events = list_events(PATH)[:n]
    out = {}
    for which in ("dwt", "dct", "fft"):
        for kappa in (1.0, 1.5, 2.0, 3.0):
            for bits in (8, 12):
                comps, peaks = [], []
                for ev in events:
                    ec = read_event_chunks(PATH, ev, cfg0)
                    bnp, lengths = pad_batch(ec.chunks, 12)        # event-max length (no 65536 over-pad)
                    sig = np.array([np.median(np.abs(pywt.wavedec(c.astype(np.float64), "db1", level=1,
                          mode="periodization")[-1])) / 0.6745 for c in ec.chunks])  # padding-free sigma
                    c, p = run_transform(bnp.astype(np.float64), lengths, which, kappa, bits, sig)
                    comps.append(c); peaks.append(p)
                out[(which, kappa, bits)] = (float(np.mean(comps)), float(np.mean(peaks)))
        print(f"{which} done")
    print(f"\n{'transform':10s} {'kappa':>6s} {'bits':>5s} {'compress':>9s} {'peak_err':>9s}")
    for (w, k, b), (c, p) in sorted(out.items()):
        print(f"{w:10s} {k:>6.1f} {b:>5d} {c:>8.1f}x {p:>9.4f}")
    json.dump({f"{w}|{k}|{b}": v for (w, k, b), v in out.items()},
              open("/sdf/group/neutrino/omara/helix/temp/figures/big/transform_compare.json", "w"))


if __name__ == "__main__":
    main()
