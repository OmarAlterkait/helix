"""Validation of the optical codec against injected truth (the same for every light system).

Known light is added to noise windows (real light-free windows of the detector, or synthetic noise
of a known model): single photons in even rows, a flash in odd rows (N photons, a fast fraction
with ``fast_ns`` and the rest with ``slow_ns`` exponential delays: argon singlet / triplet), each
photon the given response at a sub-sample time with an amplitude from ``amplitude(rng)``. The
codec runs on the result exactly as on data; per window, with the codec's own noise sigma_j:

  error  = sum over kept differences (d - s)^2 + over dropped s^2, in sigma_j^2 (s: the true
           light's difference) - signal dropped plus noise kept, the quantity the storage rule
           minimises; the ORACLE keeps exactly |s| >= sigma_j (its floor). sigma_j is a fixed
           reference: the clipped rms of each Haar level of the noise windows BEFORE the light is
           added (median over windows), so different codecs and settings are scored in the same
           units (the codec's own sigma_j moves with its light mask);
  values = stored differences + coarse sums.

Use a response different from the codec's for the injection (``h_inject``) to test robustness.

    rep = validate.run(chain, "pmt", noise_windows, channels, tick_ns=2.0, amplitude=...)
"""
from __future__ import annotations

import numpy as np

from helix.optical import codec
from helix.core import robust


def inject(noise, h, peak, polarity, tick_ns, amplitude, rng, *, flash=(100, 500), fast=0.3, fast_ns=6.0,
           slow_ns=1500.0, spacing=450, start=300):
    """(windows with light, true light) for rows of ``noise`` (float, raw ADC): singles every
    ~``spacing`` samples in even rows, a flash at ``start`` in odd rows (sizes cycling ``flash``)."""
    m, n = noise.shape
    truth = np.zeros((m, n)); off = np.arange(len(h)) - peak

    def add(r, t):
        i = int(np.floor(t)); lo, hi = max(0, i + off[0] - 1), min(n, i + off[-1] + 2)
        if hi > lo:
            truth[r, lo:hi] += amplitude(rng) * np.interp(np.arange(lo, hi) - t, off, h, left=0, right=0)
    for r in range(m):
        if r % 2 == 0:
            for p in np.arange(start, n - 100, spacing):
                add(r, p + rng.integers(-spacing // 9, spacing // 9 + 1) + rng.uniform())
        else:
            N = flash[(r // 2) % len(flash)]
            d = np.where(rng.uniform(size=N) < fast, rng.exponential(fast_ns, N), rng.exponential(slow_ns, N)) / tick_ns
            for t in start + d[start + d < n - 5]:
                add(r, t)
    return noise + polarity * truth, polarity * truth


def run(chain, system, noise, channels, *, tick_ns, amplitude, h_inject=None, seed=0, **kw):
    """Inject, encode, score: dict(single=..., flash=...) each with values, error, oracle values,
    oracle error per window (means), and n windows."""
    spec = chain.specs[system]
    h = chain.templates[spec.response] if h_inject is None else np.asarray(h_inject, np.float64)
    rng = np.random.default_rng(seed)
    raw, sig = inject(np.asarray(noise, np.float64), h, int(np.argmax(h)), spec.polarity, tick_ns, amplitude, rng, **kw)
    m, n = raw.shape
    raw = np.round(raw)
    wf = dict(channel=np.asarray(channels, np.int32), t_us=np.zeros(m), length=np.full(m, n), offset=np.arange(m) * n,
              buf=raw.astype(np.int64).ravel())
    res = chain.process(system, wf)
    lev = spec.level
    nz = np.round(np.asarray(noise, np.float64)); nz = nz[:, :n // (1 << lev) * (1 << lev)]
    _, dn = codec.haar(nz - np.median(nz, 1, keepdims=True), lev)
    ref = []
    for d in dn:                                                  # clipped rms per window (integer codes), median
        x = d.astype(np.float32)
        ref.append(float(np.nanmedian(robust.clipped_stats(x, center=np.zeros(len(x)))[1].numpy())))
    ref = np.asarray(ref)
    acc = {k: dict(values=0.0, error=0.0, oracle_values=0.0, oracle_error=0.0, n=0) for k in ("single", "flash")}
    nco = -(-n // (1 << lev))
    for g in res["groups"]:
        idx = np.asarray(g["index"]); L = g["L"]
        S = np.zeros((len(idx), L)); S[:, :n] = sig[idx]; S[:, n:] = S[:, n - 1:n]
        _, ds = codec.haar(S, lev)
        for k, i in enumerate(idx):
            a = acc["single" if i % 2 == 0 else "flash"]; sj = ref
            e = eo = 0.0; v = vo = 0
            for li in range(lev):
                nb = n >> (lev - li)
                d = np.asarray(codec._host(g["details"][li][k]), np.float64)[:nb]
                st = np.asarray(codec._host(g["stored"][li][k]), bool)[:nb]
                s = ds[li][k, :nb]; s2 = sj[li] ** 2
                e += (((d - s) ** 2)[st].sum() + (s ** 2)[~st].sum()) / s2
                ok = np.abs(s) >= sj[li]
                eo += (((d - s) ** 2)[ok].sum() + (s ** 2)[~ok].sum()) / s2
                v += int(st.sum()); vo += int(ok.sum())
            a["values"] += v + nco; a["error"] += e; a["oracle_values"] += vo + nco; a["oracle_error"] += eo; a["n"] += 1
    return {k: {q: (v / max(a["n"], 1) if q != "n" else v) for q, v in a.items()} for k, a in acc.items()}
