"""Verify the pywt-exact periodization filter bank (forward + adjoint inverse)
that the torch backend will use, against pywt.wavedec / pywt.waverec, multilevel.

Forward:  cA = roll(cconv(x, dec_lo), -L//2)[0::2]   (non-reversed filter, shift L//2)
Inverse (adjoint, exact for orthonormal): upsample into [0::2], ccorr w/ dec, roll +L//2, sum.
"""
import warnings
import numpy as np
import pywt

warnings.filterwarnings('ignore')


def cconv(x, h):
    N = x.shape[-1]
    hp = np.zeros(N); hp[:len(h)] = h
    return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(hp), n=N)


def ccorr(u, h):
    N = u.shape[-1]
    hp = np.zeros(N); hp[:len(h)] = h
    return np.fft.irfft(np.fft.rfft(u) * np.conj(np.fft.rfft(hp)), n=N)


def dwt1(x, w):
    L = w.dec_len
    lo = np.roll(cconv(x, np.asarray(w.dec_lo)), -(L // 2))[0::2]
    hi = np.roll(cconv(x, np.asarray(w.dec_hi)), -(L // 2))[0::2]
    return lo, hi


def idwt1(cA, cD, w):
    L = w.dec_len
    N = 2 * len(cA)
    ua = np.zeros(N); ua[0::2] = cA
    ud = np.zeros(N); ud[0::2] = cD
    xlo = np.roll(ccorr(ua, np.asarray(w.dec_lo)), L // 2)
    xhi = np.roll(ccorr(ud, np.asarray(w.dec_hi)), L // 2)
    return xlo + xhi


for name in ('coif3', 'sym4', 'db4', 'haar', 'db2', 'coif1', 'sym8'):
    w = pywt.Wavelet(name)
    L = w.dec_len
    for N in (256, 4096, 36864):
        lev = min(8, pywt.dwt_max_level(N, L))
        rng = np.random.default_rng(0)
        x = rng.standard_normal(N)
        cp = pywt.wavedec(x, w, level=lev, mode='periodization')
        a = x; mine = []
        for _ in range(lev):
            a, d = dwt1(a, w); mine.append(d)
        mine.append(a); mine = mine[::-1]
        ef = max(np.abs(cp[i] - mine[i]).max() for i in range(len(cp)))
        # inverse from MY coeffs, vs pywt.waverec of MY coeffs (should both = x)
        a = mine[0]
        for d in mine[1:]:
            a = idwt1(a[: len(d)], d, w)
        ei = np.abs(a - x).max()
        ew = np.abs(pywt.waverec(mine, w, mode='periodization')[:N] - a).max()
        print(f"{name:6} N={N:6} lev={lev}  fwd={ef:.2e}  inv(PR)={ei:.2e}  inv_vs_pywt={ew:.2e}")
