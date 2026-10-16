"""A light system's single-photon response, measured from its own data the way the codec uses it.

    resp = calibrate.fit_response(events, chain, system, pre=10, core=40, length=700)

``events``: waveform dicts of the system (channel, length, offset, buf; as for
:meth:`helix.optical.codec.OpticalChain.process`); ``chain``: an OpticalChain whose current
response for ``system`` is the starting point (one pass from it; iterating trades a long tail
against correlated late light and drifts). Lengths in samples from the peak.

  1. photons: the codec's own deconvolution at significance ``k_est`` (5: few noise peaks); a
     photon = a cluster of consecutive a > 0 (amplitude: its sum, time: its largest entry);
     windows with a sample at the digitizer's full scale are skipped (saturated pulses);
  2. core (offsets -pre..core): regression on isolated single photons (no other photon within
     -pre-core..+2 core; amplitude 0.3-3 x the median), each against its own local level (the
     ``core`` samples before it: earlier light, undershoot), with equal weight for every arrival
     time mod 8 samples (noise locked to the sample clock cancels although aligning on the
     largest entry favours its positive phases); 0 before the rise;
  3. tail (offsets core+1..length-1): least squares over ALL photons with the core fixed
     (Toeplitz normal equations from the auto- / cross-correlations of a and y);
  4. the response: unconstrained samples (error > 5x the median near the core) interpolated; the
     tail as 1 or 2 exponentials starting where the core ends if that describes it (chi^2 / ndf
     <= 10), else smoothed over a window growing with time (10% of the delay: structure is kept
     at its own scale); ended, with a taper, where it stops being significant;
  5. errors: 8 interleaved subsets of windows.

A stack of matched-filter maxima (the earlier method) selects noise peaks in quiet windows and
returns the noise's autocorrelation (digitizer clock lines); bunched photons bias its core.
Validated by recovering a known response injected on real noise (``inject=``): goop core within
0.04 of the peak and tail within 0.001, SBND PMT 0.09 / 0.005; the 16 ns X-ARAPUCA is not
recovered reliably (short self-triggered windows, few isolated photons).
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.optimize import curve_fit

from helix.optical import codec


def _photons(a_row):
    nz = np.nonzero(a_row > 0)[0]
    if not len(nz):
        return []
    return [(int(c[np.argmax(a_row[c])]), float(a_row[c].sum())) for c in np.split(nz, np.nonzero(np.diff(nz) > 1)[0] + 1)]


def _inject(wf, h, polarity, amplitude, rng, every=300):
    """``wf`` with photons of response ``h`` added: one every ~``every`` samples per window."""
    buf = np.asarray(wf["buf"]).astype(np.float64).copy(); p = int(np.argmax(h)); off = np.arange(len(h)) - p
    for o, n in zip(wf["offset"], wf["length"]):
        x = np.zeros(n)
        for t in np.arange(50, n - 50, every) + rng.uniform(0, every / 2, len(np.arange(50, n - 50, every))):
            i = int(t); lo, hi = max(0, i + off[0] - 1), min(n, i + off[-1] + 2)
            x[lo:hi] += amplitude(rng) * np.interp(np.arange(lo, hi) - t, off, h, left=0, right=0)
        buf[o:o + n] += polarity * x
    return dict(wf, buf=np.round(buf).astype(np.int64))


def fit_response(events, chain, system, *, pre, core, length, k_est=5.0, full_scale=None, inject=None,
                 amplitude=None, seed=0, passes=1):
    """dict(offsets, h (final, unit peak), h_ls, sem, tail, n_isolated). ``full_scale``: the
    digitizer's maximum code (windows reaching it are skipped). ``inject``: a known response
    added to every window first (with ``amplitude(rng)`` per photon) - the recovery check.
    ``passes``: each pass starts from the previous result. Without correlated light (injected
    photons) passes converge to the noise floor (goop's response from a 30% wider start: core
    error 0.091, 0.054, 0.044, 0.042 of the peak); on data, late light correlated with each photon
    (argon's slow component) is traded against a longer tail and the passes drift - use 1."""
    events = list(events)
    out = None
    for i in range(passes):
        out = _fit_once(events, chain, system, P=pre, C=core, K=length, k_est=k_est, full_scale=full_scale,
                        inject=inject, amplitude=amplitude, seed=seed)
        if i + 1 < passes:
            t = dict(chain.templates); t[chain.specs[system].response] = out["h"]
            chain = codec.OpticalChain(chain.specs, device=chain.device, templates=t)
    return out


def _fit_once(events, chain, system, *, P, C, K, k_est, full_scale, inject, amplitude, seed):
    P, C, K = int(P), int(C), int(K)
    spec = dataclasses.replace(chain.specs[system], k=k_est)
    ch = codec.OpticalChain({system: spec}, device=chain.device, templates=chain.templates)
    rng = np.random.default_rng(seed)
    caa = np.zeros((8, K + P)); cay = np.zeros((8, K + P)); segs = []; sub = 0
    for wf in events:
        if full_scale is not None:
            keep = [i for i, (o, n) in enumerate(zip(wf["offset"], wf["length"])) if np.asarray(wf["buf"][o:o + n]).max() < full_scale]
            L = np.asarray(wf["length"])[keep]
            wf = dict(channel=np.asarray(wf["channel"])[keep], t_us=np.asarray(wf["t_us"])[keep], length=L,
                      offset=np.r_[0, np.cumsum(L)[:-1]].astype(np.int64),
                      buf=np.concatenate([np.asarray(wf["buf"][wf["offset"][i]:wf["offset"][i] + wf["length"][i]]) for i in keep]))
        if inject is not None:
            wf = _inject(wf, np.asarray(inject, np.float64), spec.polarity, amplitude, rng)
        ch.capture = []
        ch.process(system, wf)
        for idx, y, valid, a in ch.capture:
            for r in range(len(idx)):
                if idx[r] < 0:
                    continue
                n = int(valid[r].sum()); yr, ar = y[r, :n], a[r, :n]
                Lf = codec._fast_len(n + K + P)
                A = np.fft.rfft(ar, Lf); Y = np.fft.rfft(yr, Lf)
                s = sub % 8; sub += 1
                caa[s] += np.fft.irfft(np.abs(A) ** 2, Lf)[:K + P]
                c_ = np.fft.irfft(np.conj(A) * Y, Lf); cay[s] += np.r_[c_[Lf - P:], c_[:K]]
                ph = _photons(ar); ts = np.array([t for t, _ in ph])
                for t, amp in ph:
                    if t - P - C < 0 or t + 2 * C >= n or ((ts > t - P - C) & (ts < t + 2 * C)).sum() > 1:
                        continue
                    segs.append((s, amp, yr[t - P:t + C + 1] - yr[t - P - C:t - P].mean(), t % 8))
    if not segs:
        raise ValueError("fit_response: no isolated photons; give more events")
    amps = np.array([x[1] for x in segs]); med = np.median(amps)
    num = np.zeros((8, 8, P + C + 1)); den = np.zeros((8, 8)); n_iso = 0
    for s, amp, g, ph in segs:
        if 0.3 * med <= amp <= 3 * med:
            num[s, ph] += amp * g; den[s, ph] += amp * amp; n_iso += 1
    cores = np.mean(num / np.maximum(den, 1e-12)[:, :, None], 1)
    core_ = np.mean(num.sum(0) / np.maximum(den.sum(0), 1e-12)[:, None], 0)
    cores /= core_[P]; core_ = core_ / core_[P]
    rise = P - int(np.argmax(core_[:P + 1][::-1] < 0.02))
    core_[:rise] = 0.0; cores[:, :rise] = 0.0

    def tail(caa_, cay_, cr):
        oc = np.arange(-P, C + 1); ot = np.arange(C + 1, K)
        rhs = cay_[P + C + 1:] - caa_[np.abs(ot[:, None] - oc[None, :])] @ cr
        c = caa_[:K - C - 1].copy(); c[0] *= 1 + 1e-6
        return solve_toeplitz(c, rhs)
    hs = np.stack([np.r_[cores[s], tail(caa[s], cay[s], cores[s])] for s in range(8)])
    h_ls = np.r_[core_, tail(caa.sum(0), cay.sum(0), core_)]
    sem = hs.std(0, ddof=1) / np.sqrt(8)
    off, h, info = _final(h_ls, sem, P, C, K)
    return dict(offsets=off, h=h, h_ls=h_ls, sem=sem, tail=info, n_isolated=n_iso, ls_offsets=np.arange(-P, K))


def _final(h_ls, sem, P, C, K):
    """The response from the least-squares estimate (see the module docstring, step 4)."""
    o = np.arange(-P, K); h = h_ls.copy(); se = sem.copy()
    near = slice(0, min(len(o), P + 3 * C + 1))
    med = np.median(se[near][se[near] > 0])
    bad = np.zeros(len(o), bool); bad[near] = se[near] > 5 * med; bad[P] = False
    good = np.nonzero(~bad)[0]
    h[bad] = np.interp(np.nonzero(bad)[0], good, h[good]); se[bad] = med
    m = o > C; t = o[m].astype(float) - C; y = h[m]; s = np.maximum(se[m], 1e-6)
    c = h[P + C - 2:P + C + 1].mean()
    cands = []
    for name, f, p0, bd in (("1exp", lambda t, tau: c * np.exp(-t / tau), (K / 6,), ([2], [5 * K])),
                            ("2exp", lambda t, w, t1, t2: c * (w * np.exp(-t / t1) + (1 - w) * np.exp(-t / t2)),
                             (0.5, 3.0, K / 4), ([0, 2, 6], [1, 5 * K, 5 * K])),
                            ("free1exp", lambda t, a, tau: a * np.exp(-t / tau), (np.clip(y[:5].mean(), -0.9, 0.9), K / 6),
                             ([-1, 2], [1, 5 * K]))):
        try:
            p_, _ = curve_fit(f, t, y, p0=p0, sigma=s, bounds=bd, maxfev=40000)
            cands.append((name, p_, float((((y - f(t, *p_)) / s) ** 2).sum()), f))
        except Exception:
            pass
    best = min([x for x in cands if x[0] != "2exp"], key=lambda x: x[2])
    two = [x for x in cands if x[0] == "2exp"]
    if two and best[2] - two[0][2] > 25:
        best = two[0]
    chi = best[2] / max(int(m.sum()) - len(best[1]), 1)
    out = h.copy()
    if chi <= 10:                                              # the tail is an exponential model
        out[m] = best[3](t, *best[1])
        end = int(np.nonzero(np.abs(out) > 1e-4)[0].max()) + 1
    else:                                                      # structure: smoothed in log time
        sm = se.copy()
        for i in range(P + C + 1, len(o)):
            w = max(1, int(round(0.1 * o[i]))); lo, hi = max(P + C + 1, i - w), min(len(o), i + w + 1)
            out[i] = h[lo:hi].mean(); sm[i] = np.sqrt((se[lo:hi] ** 2).sum()) / (hi - lo)
        sig = np.abs(out) > 3 * sm; sig[:P + C + 1] = True
        end = int(np.nonzero(sig)[0].max()) + 1
        taper = max(1, int(0.1 * o[end - 1]))
        out[end:] = 0.0
        lo = max(P + C + 1, end - taper); out[lo:end] *= np.linspace(1, 0, end - lo)
    rise = P - int(np.argmax(out[:P + 1][::-1] < 0.02)); out[:rise] = 0.0
    return o[:end], out[:end], dict(kind=best[0], params=np.asarray(best[1]).tolist(), chi2_ndf=chi, model=chi <= 10)
