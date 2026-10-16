"""Calibrate the optical codec for a light system from its own data.

    spec, template, report = calibrate(events, tick_ns)

``events``: windows of a few events, each a dict like ``io_sbnd.SBNDEvents.optical`` (channel,
length, offset, buf). Everything the codec needs is measured; only ``q`` (precision inside the
light, :class:`helix.optical.codec.OpticalSpec`) is left as a dial:

  1. noise sigma per window: clipped rms of the finest differences (/ sqrt 2) over the window's
     median 256-sample stretch (light inflates any whole-window estimate);
  2. polarity: the sign of each window's most extreme sample, majority;
  3. pulse template: strictly isolated multi-photon pulses (peak 10-60 sigma: the shape is linear
     in photons, and at that size selecting and aligning them adds no noise bias - single photons
     come out too narrow), nothing else beyond 4 sigma from 200 ns before to 6 pulse lengths
     after, each on its own pre-pulse baseline, re-aligned by the matched filter of a first mean,
     averaged (peak 1); the codec's kernel is its positive lobe above 2% of the peak;
  4. footprint: the lobe above 5% of the peak, one sample more before and x1.5 after (SBND
     PMT -4/+11, X-ARAPUCA -8/+27; studies chose -4/+12 and -8/+40); gaps up to twice the
     footprint closed; coarse block ~4 footprints (level 4-7); q on the levels finer than the
     template's energy peak (PMT D1-D3);
  5. photon amplitude: ``spe_adc`` from the experiment's calibration (SBND PDS DB: 12.4 ADC, 40%
     spread) - PASS IT: the fallback (Gaussian fit of matched-filter amplitudes in quiet
     stretches, signal minus flipped polarity, above 5 noise widths) is not reliable on real data
     yet (SBND PMT collapses, goop reads ~1.6x high; right only on the synthetic test), and k
     follows from it;
  6. k: photons of that amplitude and the measured full response injected into quiet stretches
     of the data itself; the largest k whose support covers ``target`` of them (default 90%),
     so k means the same single-photon efficiency on every experiment. The report gives the
     efficiency and the noise regions per us (flipped polarity) at each k.
"""
from __future__ import annotations

import numpy as np

from helix.optical import codec

STRETCH = 256


def _crms(d):
    c = d
    for _ in range(6):
        c = d[np.abs(d) < 3.5 * max(c.std() * 1.04, 1e-3)]
    return float(c.std() * 1.04) if c.size > 9 else float("nan")


def _window_sigma(y):
    """Clipped rms of the finest differences / sqrt 2 over the median 256-sample stretch."""
    k = len(y) // STRETCH
    if k == 0:
        return _crms((y[0:len(y) // 2 * 2:2] - y[1:len(y) // 2 * 2:2]) / np.sqrt(2))
    r = [_crms((s[0::2] - s[1::2]) / np.sqrt(2)) for s in y[:k * STRETCH].reshape(k, STRETCH)]
    return float(np.nanmedian(r))


def _windows(events):
    out = []
    for wf in events:
        for c, o, n in zip(np.asarray(wf["channel"]), np.asarray(wf["offset"]), np.asarray(wf["length"])):
            if n >= 64:
                out.append((int(c), np.asarray(wf["buf"][o:o + n], np.float64)))
    return out


def _isolated(W, sig, pol, pre, post, width, band, kern=None, mf_select=False):
    """Isolated pulses (peak in ``band`` sigma, nothing beyond 4 sigma from ``pre`` before to
    ``post`` after except the pulse itself, ``width`` samples), in sigma units on their pre-pulse
    baseline, peak at index ``pre``; with ``kern`` re-aligned by its matched filter (+-3), and with
    ``mf_select`` found as matched-filter maxima above 5 instead of by their peak sample (a peak
    selection near the photon amplitude narrows the mean; the filter's only along the template)."""
    from scipy.signal import fftconvolve
    segs = []
    if kern is not None:
        u = kern - kern.mean(); u /= np.sqrt((u ** 2).sum()); pk = int(np.argmax(kern)); lag = codec.template_lag(kern)
    for (_, x), s in zip(W, sig):
        y = pol * (x - np.median(x)) / s
        loud = np.abs(y) > 4
        if mf_select:
            mf = fftconvolve(y, u[::-1], mode="same")
            cand = np.where((mf[1:-1] > 5) & (mf[1:-1] >= mf[:-2]) & (mf[1:-1] > mf[2:]))[0] + 1 - lag
            cand = cand[(cand > 0) & (cand < len(y))]
            cand = cand[y[cand] < band[1]]
        else:
            cand = np.where((y[1:-1] > band[0]) & (y[1:-1] < band[1]) & (y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
        for j in cand:
            if not (pre + 3 <= j < len(y) - post - 3) or loud[j - pre:j - 3].any() or loud[j + width:j + post].any():
                continue
            if kern is not None:
                c = [np.dot(y[j + d - pk:j + d - pk + len(kern)], u) for d in range(-3, 4)]
                j = j + int(np.argmax(c)) - 3
            seg = y[j - pre:j + post]
            segs.append(seg - seg[:pre - 3].mean())
    return np.array(segs)


def _lobe(t, p, f):
    st = p
    while st > 0 and t[st - 1] >= f:
        st -= 1
    en = p
    while en < len(t) - 1 and t[en + 1] >= f:
        en += 1
    return st, en


def _energy_peak_level(t, levels=8):
    """Haar level holding most of the template's energy (averaged over alignment)."""
    E = np.zeros(levels)
    for sh in range(1 << levels):
        x = np.zeros(4 << levels); n = min(len(t), len(x) - sh); x[sh:sh + n] = t[:n]; s = x
        for j in range(levels):
            a, b = s[0::2], s[1::2]; E[j] += (((a - b) / np.sqrt(2)) ** 2).sum(); s = (a + b) / np.sqrt(2)
    return int(np.argmax(E)) + 1


def _near(m, h):
    """Samples within h of a True sample (any length)."""
    c = np.r_[0, np.cumsum(m)]
    i = np.arange(len(m))
    return (c[np.minimum(i + h + 1, len(m))] - c[np.maximum(i - h, 0)]) > 0


def _amplitude(W, sig, pol, kern, quiet_n):
    """Photon amplitude (sigma units) from matched-filter maxima in quiet stretches."""
    from scipy.signal import fftconvolve
    u = kern - kern.mean(); nu = np.sqrt((u ** 2).sum())
    if len(kern) < 3 or nu <= 0:
        return float("nan"), float("nan")
    u /= nu
    gain = float(kern @ u)                                    # filter output per unit peak amplitude
    amps = {1: [], -1: []}; mfn = []
    for (_, x), s in zip(W, sig):
        y = (x - np.median(x)) / s
        quiet = ~_near(np.abs(y) > 20, quiet_n)
        for sg in (1, -1):
            mf = fftconvolve(sg * pol * y, u[::-1], mode="same")
            if sg == -1 and quiet.sum() > 64:
                mfn.append(_crms(mf[quiet]))
            lm = np.where((mf[1:-1] > mf[:-2]) & (mf[1:-1] >= mf[2:]) & quiet[1:-1] & (mf[1:-1] > 1.5))[0] + 1
            amps[sg] += list(mf[lm] / gain)
    # noise maxima (both polarities alike) end ~5 widths of the amplitude noise above 0 (measured:
    # the filter output's rms / gain - coloured noise makes it wider than white); the filter's
    # sidelobes add low maxima around each photon, so the photons are fit above that: a Gaussian
    # to (signal - flipped) over amplitudes > 5 noise widths
    cut = 5.0 * float(np.nanmedian(mfn)) / gain
    bins = np.arange(cut, cut + 40, 0.1); c = 0.5 * (bins[1:] + bins[:-1])
    hp = np.histogram(amps[1], bins)[0].astype(float); hm = np.histogram(amps[-1], bins)[0].astype(float)
    d = hp - hm
    from scipy.optimize import least_squares
    p0 = (max(d.max(), 1.0), c[np.argmax(d)], max(0.3 * c[np.argmax(d)], 0.3))
    fit = least_squares(lambda p: (p[0] * np.exp(-0.5 * ((c - p[1]) / p[2]) ** 2) - d) / np.sqrt(hp + hm + 1), p0,
                        bounds=([0, 0, 0.1], [np.inf, 60, 30])).x
    return float(fit[1]), float(fit[2] / fit[1])


def calibrate(events, tick_ns, *, polarity=None, spe_adc=None, spe_spread=None, target=0.9, q=1.0,
              ks=(2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0), n_inject=3000, seed=0, name="optical"):
    """-> (OpticalSpec, template dict(kernel, response, tick_ns), report dict)."""
    rng = np.random.default_rng(seed)
    W = _windows(events)
    sig = np.maximum(np.array([_window_sigma(x - np.median(x)) for _, x in W]), 1e-3)
    sigma = float(np.nanmedian(sig))
    if polarity is None:
        # the sign of each window's most extreme sample, majority (counting samples is fooled by
        # a long overshoot: goop's +13-sigma recovery outlasts its flash)
        ext = [np.sign((x - np.median(x))[np.argmax(np.abs(x - np.median(x)))]) for _, x in W]
        polarity = 1 if np.sum(ext) >= 0 else -1
    pre_n, post_n, width = int(np.ceil(200 / tick_ns)), int(np.ceil(800 / tick_ns)), max(int(np.ceil(40 / tick_ns)), 3)
    # template from multi-photon pulses (10-60 sigma): the shape is linear in photons and the
    # noise bias of selecting and aligning them is negligible at that size (single photons at
    # 4-5 sigma come out too narrow); a first mean sets the pulse length, which sizes the
    # isolation, then the pulses are re-aligned by its matched filter
    band = (10.0, 60.0)
    S = _isolated(W, sig, polarity, pre_n, post_n, width, band)
    single = len(S) < 50                                      # few multi-photon pulses: single photons,
    if single:                                                # found by the matched filter (2 passes)
        band = (5.0, 60.0); S = _isolated(W, sig, polarity, pre_n, post_n, width, band)
    if len(S) < 20:
        raise ValueError(f"calibrate: {len(S)} isolated pulses; give more events")
    for _ in range(2 if single else 1):
        t1 = S.mean(0); t1 /= t1[pre_n]
        s1, e1 = _lobe(t1, pre_n, 0.05)
        width = max(e1 - pre_n + 2, 3); post_n = max(post_n, 6 * width)
        S2 = _isolated(W, sig, polarity, pre_n, post_n, width, (0.0, 60.0) if single else band, kern=t1[s1:e1 + 1],
                       mf_select=single)
        if len(S2) >= 20:
            S = S2
    resp = S.mean(0)
    pk = pre_n - 3 + int(np.argmax(resp[pre_n - 3:pre_n + 4]))
    resp = np.roll(resp, pre_n - pk); resp /= resp[pre_n]
    st, en = _lobe(resp, pre_n, 0.02)
    kern = resp[st:en + 1]
    s5, e5 = _lobe(resp, pre_n, 0.05)
    pre, post = pre_n - s5 + 1, int(np.ceil(1.5 * (e5 - pre_n)))
    level = int(np.clip(np.round(np.log2(4 * (pre + post))), 4, 7))
    q_levels = max(_energy_peak_level(kern) - 1, 1)
    gap = 2 * (pre + post)
    if spe_adc is not None:
        amp_sig, spread = spe_adc / sigma, (0.4 if spe_spread is None else spe_spread)
    else:
        amp_sig, spread = _amplitude(W, sig, polarity, kern, int(np.ceil(2000 / tick_ns)))
    # k: inject photons into quiet stretches of the data, pick the largest k reaching the target
    # windows cropped to a common length (the median, at most 32768 samples), photons spaced by
    # at least two isolation lengths in the quiet parts (a first support at k 3 marks the light)
    n0 = int(min(np.median([len(x) for _, x in W]), 32768))
    sel = [i for i, (_, x) in enumerate(W) if len(x) >= n0][:400]
    X = np.stack([W[i][1][:n0] for i in sel]); ch = np.array([W[i][0] for i in sel], np.int32); sr = sig[sel]
    shape = resp
    tt = np.arange(n0)
    Y = X.copy(); truth = []
    base_spec = codec.OpticalSpec(name, polarity, k=3.0, pre=pre, post=post, gap=gap, level=level, q=q, q_levels=q_levels)
    oc0 = codec.OpticalChain({name: base_spec}, templates={name: kern})
    wf0 = dict(channel=ch, t_us=np.zeros(len(X)), length=np.full(len(X), n0), offset=np.arange(len(X)) * n0,
               buf=np.round(X).astype(np.int64).ravel())
    r0 = oc0.process(name, wf0)
    busy = np.zeros(X.shape, bool)
    for g in r0["groups"]:
        busy[g["index"]] = np.asarray(g["roi"])[:, :n0]
    per = max(1, n_inject // len(X))
    for i in range(len(X)):
        free = ~_near(busy[i], post_n); free[:pre_n] = False; free[n0 - post_n:] = False
        for p in rng.permutation(np.where(free)[0]):
            if len([1 for (ii, _) in truth[-per:] if ii == i]) >= per:
                break
            if not free[p]:
                continue
            a = max(rng.normal(amp_sig, spread * amp_sig), 0.2 * amp_sig) * sr[i]
            Y[i] += polarity * a * np.interp(tt - p - rng.uniform(), np.arange(len(shape)) - pre_n, shape, left=0, right=0)
            truth.append((i, int(p)))
            free[max(p - 2 * post_n, 0):p + 2 * post_n] = False
    wfi = dict(wf0, buf=np.round(Y).astype(np.int64).ravel())
    curve = []
    for k in ks:
        sp = codec.OpticalSpec(name, polarity, k=k, pre=pre, post=post, gap=gap, level=level, q=q, q_levels=q_levels)
        res = codec.OpticalChain({name: sp}, templates={name: kern}).process(name, wfi)
        roi = np.zeros(X.shape, bool)
        for g in res["groups"]:
            roi[g["index"]] = np.asarray(g["roi"])[:, :n0]
        flip = codec.OpticalChain({name: codec.OpticalSpec(name, -polarity, k=k, pre=pre, post=post, gap=gap, level=level)},
                                  templates={name: kern}).process(name, wf0)
        nr = 0
        for g in flip["groups"]:
            r_ = np.asarray(g["roi"])[:, :n0]
            nr += int((np.diff(np.c_[np.zeros(len(r_)), r_.astype(int)], axis=1) == 1).sum())
        curve.append(dict(k=k, efficiency=float(np.mean([roi[i, p] for i, p in truth])),
                          noise_regions_per_us=nr / (len(X) * n0 * tick_ns / 1000), values_per_window=res["n_kept"] / len(X)))
    ok = [c for c in curve if c["efficiency"] >= target]
    k = max(c["k"] for c in ok) if ok else min(ks)
    spec = codec.OpticalSpec(name, polarity, k=k, pre=pre, post=post, gap=gap, level=level, q=q, q_levels=q_levels)
    report = dict(tick_ns=tick_ns, sigma_adc=sigma, polarity=polarity, isolated_pulses=len(S),
                  photon_amplitude_sigma=amp_sig, photon_amplitude_spread=spread, photon_amplitude_adc=amp_sig * sigma,
                  kernel_samples=len(kern), footprint=(pre, post), level=level, q_levels=q_levels, gap=gap,
                  injected=len(truth), k_curve=curve, k=k, target=target)
    return spec, dict(kernel=kern, response=resp, tick_ns=tick_ns, pre=pre_n), report
