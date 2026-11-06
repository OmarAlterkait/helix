"""The 2D (wire offset x time) response kernel: the example's position-averaged
R_avg, and a least-squares fit of our simulation's own response from
(hits, clean sensor) pairs.

Kernel convention (the example's, wiener_2d.pad_and_shift_kernel): ``R[k, j]``
is the signal, in ADC per electron, on wire ``w + wire_offsets[j]`` at tick
``t + (k - tzb)`` from charge arriving at (w, t). Translation invariant.
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(HERE, "reference", "uboone_deconv")
FITTED = os.path.join(HERE, "response")


def load_kernel(plane, source="example"):
    """-> (R (n_t, n_w) float64, wire_offsets int, tzb int).

    source 'example': the shipped avg_response_<plane>.npz (JAXTPC kernel,
    trapezoid-averaged over the pitch by build_avg_response.py).
    source 'fit': response/fit_<plane>.npz from fit_response.py.
    source 'exact' / 'exact0': response/exact[0]_<plane>.npz from v2/exact_kernel.py (JAXTPC-derived).
    """
    if source == "example":
        d = np.load(os.path.join(REF, f"avg_response_{plane}.npz"))
    elif source == "fit":
        d = np.load(os.path.join(FITTED, f"fit_{plane}.npz"))
    elif source in ("exact", "exact0"):            # v2/exact_kernel.py: from JAXTPC's own code (production commit)
        d = np.load(os.path.join(FITTED, f"{source}_{plane}.npz"))
    else:
        raise ValueError(source)
    return d["R_avg"].astype(np.float64), d["wire_offsets"].astype(np.int64), int(d["time_zero_bin"])


def kernel_full(R, wire_offsets, tzb, nw, nt):
    """Vectorised twin of the example's pad_and_shift_kernel (same array)."""
    K = np.zeros((nw, nt), np.float64)
    k = np.arange(R.shape[0])
    for j, wo in enumerate(wire_offsets):
        np.add.at(K[int(wo) % nw], (k - tzb) % nt, R[:, j])
    return K


def forward(H, R, wire_offsets, tzb):
    """hits (electrons) -> predicted sensor (ADC), cyclic 2D convolution."""
    nw, nt = H.shape
    Kf = np.fft.rfft2(kernel_full(R, wire_offsets, tzb, nw, nt))
    return np.fft.irfft2(np.fft.rfft2(H.astype(np.float64)) * Kf, s=(nw, nt))


def store_like_sensor(S):
    """What the sensor writer does to a float waveform: integer ADC, |v| < 2 dropped."""
    v = np.round(S)
    v[np.abs(v) < 2] = 0.0
    return v


class LSAccumulator:
    """Exact least squares for a compact kernel K[dw, dt], dw in [-Wk, Wk],
    dt in [-Tb, Ta], from many (H, S) planes: accumulates the auto-correlation
    of H and the cross-correlation H x S at the needed lags (zero padded, so no
    wrap-around), then solves the normal equations."""

    def __init__(self, Wk=10, Tb=150, Ta=60):
        self.Wk, self.Tb, self.Ta = Wk, Tb, Ta
        self.dw = np.arange(-Wk, Wk + 1)
        self.dt = np.arange(-Tb, Ta + 1)
        self.ac = None
        self.cc = None

    def add(self, H, S):
        nw, nt = H.shape
        lw, lt = 2 * self.Wk + 1, self.Tb + self.Ta + 1
        P = (nw + 2 * lw, nt + 2 * lt)
        FH = np.fft.rfft2(H.astype(np.float64), s=P)
        FS = np.fft.rfft2(S.astype(np.float64), s=P)
        ac = np.fft.irfft2(FH.conj() * FH, s=P)          # ac[l] = sum_x H[x] H[x+l]
        cc = np.fft.irfft2(FH.conj() * FS, s=P)          # cc[l] = sum_x H[x] S[x+l]
        # keep only lags we need: ac at |lw| <= 2Wk, lt in [-(Tb+Ta), Tb+Ta]; cc at the kernel support
        LW = np.arange(-2 * self.Wk, 2 * self.Wk + 1)
        LT = np.arange(-(self.Tb + self.Ta), self.Tb + self.Ta + 1)
        a = ac[np.ix_(LW % P[0], LT % P[1])]
        c = cc[np.ix_(self.dw % P[0], self.dt % P[1])]
        self.ac = a if self.ac is None else self.ac + a
        self.cc = c if self.cc is None else self.cc + c

    def solve(self, ridge=1e-9):
        DW, DT = np.meshgrid(self.dw, self.dt, indexing="ij")
        dw, dt = DW.ravel(), DT.ravel()
        iw = (dw[:, None] - dw[None, :]) + 2 * self.Wk
        it = (dt[:, None] - dt[None, :]) + (self.Tb + self.Ta)
        A = self.ac[iw, it]
        b = self.cc.ravel()
        A = A + ridge * np.trace(A) / len(A) * np.eye(len(A))
        K = np.linalg.solve(A, b).reshape(len(self.dw), len(self.dt))
        # to the example's layout: R[k, j], k = dt + Tb, tzb = Tb
        return K.T.copy(), self.dw.copy(), self.Tb
