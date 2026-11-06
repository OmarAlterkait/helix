"""Quick facts: sizes, totals, time lag hits->sensor, the example kernel vs our sim."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sp.data import Event, registry  # noqa: E402

REF = os.path.join(HERE, "reference", "uboone_deconv")


def xcorr_t(a, b, maxlag=200):
    """sum over wires of cross-correlation along time, lag = shift of b relative to a."""
    n = a.shape[1]
    F = np.fft.rfft(a, 2 * n, axis=1).conj() * np.fft.rfft(b, 2 * n, axis=1)
    c = np.fft.irfft(F.sum(0), 2 * n)
    lags = np.r_[np.arange(0, maxlag + 1), np.arange(-maxlag, 0)]
    return lags, c[lags]


reg = registry()
for ev in (0, 1):
    E = Event(ev)
    print(f"ev{ev:03d} {E.sensor} event {E.event}")
    for g in range(6):
        C = E.clean(g, reg)
        H = E.truth(g, C.shape)
        nz = C[C != 0]
        lags, c = xcorr_t(H.astype(np.float64), C.astype(np.float64))
        print(f"  g{g} {reg[g]['label']}: shape {C.shape} clean sum {C.sum():.4g} absmax {np.abs(C).max():.0f} "
              f"min|nz| {np.abs(nz).min() if nz.size else 0:.1f} nnz {nz.size} | hits sum {H.sum():.4g} "
              f"| sum clean / sum hits {C.sum() / max(H.sum(), 1):.4g} | xcorr peak lag {lags[np.argmax(c)]} "
              f"min lag {lags[np.argmin(c)]}")

for p in "UVY":
    d = np.load(os.path.join(REF, f"avg_response_{p}.npz"))
    R = d["R_avg"]
    print(p, R.shape, d["wire_offsets"][[0, -1]], int(d["time_zero_bin"]), float(d["electrons_per_adc"]),
          "sum", R.sum(), "central wire peak t", R[:, R.shape[1] // 2].argmax(), "min t", R[:, R.shape[1] // 2].argmin(),
          "peak", R.max(), R.min())
