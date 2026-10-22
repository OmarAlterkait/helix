"""Optical codec: one method for every light system (SBND PMT and X-ARAPUCA, goop, ...).

A system is described by its measured single-photon response h (unit peak, pulse, undershoot and
recovery; helix.optical.calibrate) and the photon significance k; everything else follows from h.

Per event and system:
  1. noise per CHANNEL from where there is no light: the matched-filter statistic (the window,
     minus a local baseline, correlated with the zero-mean unit-norm core of h - its shortest
     span around the peak holding 98% of h's energy), a first light mask from its rms over each
     window's median 256-sample stretch, then the clipped rms of the statistic and of every Haar
     level, and the window's clipped mean, outside the light (triggers at k, widened by the
     core span); medians over the channel's windows (the pedestal too: a median of integer codes
     is off by up to half a count, and light below the trigger lifts a lit window's own value);
  2. the light: photon amplitudes a >= 0 minimising |y - h*a|^2 / 2 + lambda sum(a), y the window
     minus its channel's pedestal, lambda = k x the statistic's noise x |h| (a photon enters where
     its matched-filter significance exceeds k), refitted without the penalty on its support
     (FISTA); light = h*a - isolated photons, pile-up and slow dense light alike;
  3. stored: the integer Haar transform (sum / difference) of the RAW ADC samples to ``level``:
     the coarse sums of every 2^level block always, and a difference where the light's difference
     reaches q sigma_j (q 1: keep exactly where the signal is worth a noise sigma - nothing is
     chosen by its own noisy value) or where the data depart from the light model by
     |d - light's d| >= residual sigma_j (5: a noise coefficient ~6e-7; a wrong response on big
     pulses, artefacts that are not light).
Decoding is the exact inverse: a stored difference is exact, a missing one splits its block's sum
evenly. Validation against injected truth: helix.optical.validate; numbers: docs/OPTICAL.md.

Batching: windows of one length class are one batch - lengths within 25% padding of the
batch's longest (Haar and baseline blocks round them up); on the GPU rows are padded to a power
of two (multiples of 256 above), so shapes repeat without padding short batches to 256 rows.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from helix.core import backend
from helix.tpc import robust

_DATA = Path(__file__).resolve().parent / "data"
BLOCK = 64                      # local-baseline block (samples)
PAD_FRACTION = 0.25             # batch windows whose length is within this padding of the batch's longest
ENERGY = 0.98                   # the response's core: the shortest span around the peak with this energy fraction


@dataclass(frozen=True)
class OpticalSpec:
    response: str               # template key of the measured single-photon response (unit peak)
    polarity: int               # sign of a pulse: -1 negative-going (PMT), +1 positive (X-ARAPUCA)
    k: float = 3.0              # photon significance (matched-filter noise units)
    level: int = 6              # integer Haar levels: coarse block 2^level samples
    q: float = 1.0              # keep a difference where the light's difference reaches q sigma_j
    residual: float = 5.0       # ... or where the data depart from the light by residual sigma_j (0: off)


SBND = {
    "pmt": OpticalSpec("pmt_response", -1),
    "xarapuca": OpticalSpec("xarapuca_response", 1),
}
# goop (HELIX's simulated light, 1 ns ticks): its own response (data/goop_optical.npz), 128 ns
# coarse blocks. OpticalChain(GOOP, templates="goop").
GOOP = {"goop": OpticalSpec("goop_response", -1, level=7)}


def make_specs(overrides=None, base=None):
    """``base`` (default :data:`SBND`) with per-system field overrides ``{system: {field: value}}``."""
    out = dict(SBND if base is None else base)
    for s, o in (overrides or {}).items():
        if s not in out:
            raise ValueError(f"unknown optical system {s!r}")
        out[s] = dataclasses.replace(out[s], **o)
    return out


def load_templates(name="sbnd"):
    d = np.load(_DATA / f"{name}_optical.npz", allow_pickle=False)
    return {k: d[k].astype(np.float64) for k in d.files if k != "meta"}, json.loads(str(d["meta"]))


def core_span(h, energy=ENERGY):
    """(first, last) index of the shortest span of ``h`` around its peak holding ``energy`` of its
    energy (sum of squares)."""
    h = np.asarray(h, np.float64); p = int(np.argmax(h)); e = h ** 2; tot = e.sum(); best = (0, len(h) - 1)
    for a in range(p + 1):
        c = np.cumsum(e[a:])
        if c[-1] < energy * tot:
            break
        b = a + int(np.searchsorted(c, energy * tot))
        if b >= p and b - a < best[1] - best[0]:
            best = (a, b)
    return best


def matched_filter(h):
    """(zero-mean unit-norm kernel of the response's core, lag of its output after the pulse
    peak, core samples before / after the peak)."""
    from scipy.signal import convolve
    h = np.asarray(h, np.float64); p = int(np.argmax(h)); a, b = core_span(h)
    t = h[a:b + 1]
    u = t - t.mean(); u = u / np.sqrt(np.sum(u ** 2))
    z = np.zeros(4 * len(t)); s = len(t); z[s:s + len(t)] = t
    lag = int(np.argmax(convolve(z, u[::-1], mode="same"))) - (s + p - a)
    return u, lag, p - a, b - p


def batches(lengths, level, torch_=False):
    """Windows grouped for batching: [(padded length, row indices (-1: padding row))]. Sorted by
    length; a batch takes windows whose padded length is within PAD_FRACTION of its longest's."""
    unit = max(1 << level, BLOCK)
    Lp = (np.maximum(np.asarray(lengths, np.int64), 1) + unit - 1) // unit * unit
    order = np.argsort(-Lp, kind="stable"); out = []; i = 0
    while i < len(order):
        L = int(Lp[order[i]]); j = i
        while j < len(order) and Lp[order[j]] * (1 + PAD_FRACTION) >= L:
            j += 1
        idx = np.sort(order[i:j])
        if torch_:
            nb = len(idx); r = 1 << int(np.ceil(np.log2(nb))) if nb < 256 else -(-nb // 256) * 256
            idx = np.concatenate([idx, np.full(r - nb, -1, np.int64)])
        out.append((L, idx)); i = j
    return out


# ---------------------------------------------------------------- array helpers (numpy / torch)
def _is_torch(x):
    return hasattr(x, "detach")


def _cat(xs, axis=1):
    if _is_torch(xs[0]):
        import torch
        return torch.cat(xs, axis)
    return np.concatenate(xs, axis)


def _host(x):
    return x.detach().cpu().numpy() if _is_torch(x) else np.asarray(x)


def haar(x, level):
    """Integer Haar of the rows of ``x`` (length a multiple of 2^level): (coarse sums, [d_level, ..., d_1])."""
    s, ds = x, []
    for _ in range(level):
        a, b = s[:, 0::2], s[:, 1::2]
        ds.append(a - b); s = a + b
    return s, ds[::-1]


def unhaar(coarse, details, stored=None):
    """Inverse of :func:`haar` (numpy, float64); ``stored`` masks: a missing difference is 0."""
    s = np.asarray(coarse, np.float64)
    for j, d in enumerate(details):
        d = np.asarray(d, np.float64) if stored is None else np.where(stored[j], d, 0.0)
        x = np.empty(s.shape[:-1] + (2 * s.shape[-1],))
        x[..., 0::2] = (s + d) / 2; x[..., 1::2] = (s - d) / 2
        s = x
    return s



def _baseline(y):
    """Local baseline: block means, linearly interpolated between block centres (edges flat)."""
    n, L = y.shape
    bm = y.reshape(n, L // BLOCK, BLOCK).mean(-1)
    pos = (np.arange(L) - (BLOCK - 1) / 2) / BLOCK
    i0 = np.clip(np.floor(pos), 0, L // BLOCK - 1).astype(np.int64)
    i1 = np.minimum(i0 + 1, L // BLOCK - 1)
    w = np.clip(pos - i0, 0, 1).astype(np.float32)
    if _is_torch(y):
        import torch
        i0, i1, w = (torch.as_tensor(a, device=y.device) for a in (i0, i1, w))
    return bm[:, i0] * (1 - w) + bm[:, i1] * w


def _correlate(z, u, lag):
    """'same' correlation with ``u`` (as scipy's convolution with the reversed kernel), shifted
    ``lag`` samples earlier (0 where the shift runs out); K shifted multiply-adds (no per-shape
    FFT / cuDNN setup on the GPU)."""
    n, L = z.shape
    K = len(u)
    a, b = K - 1 - (K - 1) // 2, (K - 1) // 2
    if _is_torch(z):
        import torch
        import torch.nn.functional as F
        zp = F.pad(z, (a, b))
        y = zp[:, 0:L] * float(u[0])
        for j in range(1, K):
            y.add_(zp[:, j:j + L], alpha=float(u[j]))
        out = torch.zeros_like(y)
    else:
        zp = np.pad(z, ((0, 0), (a, b)))
        y = zp[:, 0:L] * np.float32(u[0])
        for j in range(1, K):
            y += zp[:, j:j + L] * np.float32(u[j])
        out = np.zeros_like(y)
    if lag > 0:
        out[:, :L - lag] = y[:, lag:]
    elif lag < 0:
        out[:, -lag:] = y[:, :lag]
    else:
        out = y
    return out


DECONV_ITER = (25, 8)           # FISTA iterations, penalised then refitted (truth: 25 as 300 on photons, flashes +2-4%)


def _fast_len(n):
    """Smallest 2^a 3^b 5^c >= n (a fast FFT length for numpy and cuFFT)."""
    best = 1 << int(np.ceil(np.log2(n)))
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best


def _deconvolve(y, w, lam, h, peak, iters=DECONV_ITER, return_a=False):
    """Light in rows ``y`` (signal-positive, baseline 0) as h*a: the photon train a >= 0 minimising
    |w (y - h*a)|^2 / 2 + sum(lam a) (FISTA), refitted without the penalty on its support. ``h``:
    the response, its peak at index ``peak``; ``w``: 1 on valid samples; ``lam``: (rows, 1).
    ``return_a``: also the photon train a (rows of y; for diagnostics and figures)."""
    m, L = y.shape
    n = len(h)
    Lf = _fast_len(L + n)                                         # linear convolution: no wrap-around
    torch_ = _is_torch(y)
    hc = np.zeros(Lf); hc[(np.arange(n) - peak) % Lf] = h       # circular kernel, peak at 0
    lip = float((np.abs(np.fft.rfft(hc)) ** 2).max())
    if torch_:
        import torch
        H = torch.fft.rfft(torch.as_tensor(hc, dtype=y.dtype, device=y.device))
        conv = lambda a, K=H: torch.fft.irfft(torch.fft.rfft(a, Lf, dim=1) * K, Lf, dim=1)[:, :L]  # noqa: E731
        relu_ = lambda x: x.clamp_(min=0)  # noqa: E731
        lerp = torch.lerp
    else:
        H = np.fft.rfft(hc)
        conv = lambda a, K=H: np.fft.irfft(np.fft.rfft(a, Lf, axis=1) * K, Lf, axis=1)[:, :L]  # noqa: E731
        relu_ = lambda x: np.maximum(x, 0, out=x)  # noqa: E731
        lerp = lambda a, b, c: a + c * (b - a)  # noqa: E731
    Hc = H.conj()
    y = y * w
    act = (conv(y, Hc) > lam).any(1)                              # rows where a photon can enter (else a = 0)
    out = y * 0
    if not bool(act.any()):
        return (out, out * 0) if return_a else out
    y, w, lam = y[act], w[act], (lam[act] if getattr(lam, "ndim", 0) else lam)
    Hs = Hc / lip                                                 # gradient step folded into the adjoint

    def fista(a, lam_, supp, nit):                                # few, in-place passes per iteration
        z, t = a, 1.0
        for _ in range(nit):
            r = conv(z); r -= y; r *= w
            an = z - conv(r, Hs)
            if lam_ is not None:
                an -= lam_
            relu_(an)
            if supp is not None:
                an *= supp
            tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
            z = lerp(a, an, 1 + (t - 1) / tn); a, t = an, tn
        return a
    a = fista(y * 0, lam / lip, None, iters[0])
    a = fista(a, None, (a > 0) * 1.0, iters[1])
    out[act] = conv(a)
    if return_a:
        a_full = out * 0; a_full[act] = a
        return out, a_full
    return out


QUIET = 256                     # noise of the statistic: rms over stretches of this many samples


def _quiet_rms(stat, n):
    """Noise of the detection statistic per window: the median of its rms over the window's
    whole 256-sample stretches (a window shorter than one stretch: its own rms). Light inflates
    any whole-window estimate (goop: every window holds a ~1000-sigma flash and its slow light;
    X-ARAPUCA: self-triggered, each window holds a pulse); the median stretch is quiet while
    fewer than half are lit, and unbiased on noise (a lower quantile is biased low)."""
    m, L = stat.shape
    k = L // QUIET
    nn = _host(n)
    if _is_torch(stat):
        import torch
        if k == 0:                                               # a batch of windows all shorter than a stretch
            return (stat.pow(2).sum(1) / torch.clamp(torch.as_tensor(nn, device=stat.device), min=1)).sqrt()
        seg = stat[:, :k * QUIET].reshape(m, k, QUIET)
        r = seg.pow(2).mean(-1).sqrt()
        full = torch.as_tensor(np.arange(1, k + 1)[None, :] * QUIET <= nn[:, None], device=stat.device)
        r = torch.where(full, r, torch.full_like(r, float("inf")))
        nf = full.sum(1)
        q = torch.sort(r, 1).values.gather(1, torch.clamp((nf - 1) // 2, min=0)[:, None])[:, 0]
        own = (stat.pow(2).sum(1) / torch.clamp(torch.as_tensor(nn, device=stat.device), min=1)).sqrt()
        return torch.where(nf > 0, q, own)
    if k == 0:
        return np.sqrt((stat.astype(np.float64) ** 2).sum(1) / np.maximum(nn, 1)).astype(np.float32)
    seg = stat[:, :k * QUIET].reshape(m, k, QUIET)
    r = np.sqrt((seg.astype(np.float64) ** 2).mean(-1))
    full = np.arange(1, k + 1)[None, :] * QUIET <= nn[:, None]
    r = np.where(full, r, np.inf)
    nf = full.sum(1)
    q = np.sort(r, 1)[np.arange(m), np.maximum((nf - 1) // 2, 0)]
    own = np.sqrt((stat.astype(np.float64) ** 2).sum(1) / np.maximum(nn, 1))
    return np.where(nf > 0, q, own).astype(np.float32)


def _channel_median(v, ch):
    """Rows of ``v`` (host; columns: statistic, levels coarse..fine; NaN = no estimate) replaced
    by the median over rows of the same channel. A level no window of the channel can estimate
    (fewer than 10 coefficients, e.g. level 6 of 700-sample windows) takes the next finer
    level's sigma times sqrt(2) (white noise)."""
    v = np.asarray(v, np.float64)
    out = np.full_like(v, np.nan)
    if len(v):                                                    # one sort per column (no loop over channels)
        uc, inv = np.unique(ch, return_inverse=True)
        for col in range(v.shape[1]):
            x = v[:, col]; nan = np.isnan(x)
            o = np.lexsort((np.where(nan, np.inf, x), inv))       # by channel, then value; NaN last
            xs = x[o]
            start = np.searchsorted(inv[o], np.arange(len(uc)))
            k = np.bincount(inv[~nan], minlength=len(uc))         # estimates per channel
            lo = start + np.maximum(k - 1, 0) // 2; hi = np.minimum(start + k // 2, len(xs) - 1)
            out[:, col] = np.where(k > 0, 0.5 * (xs[lo] + xs[hi]), np.nan)[inv]
    for col in range(v.shape[1] - 2, 0, -1):
        out[:, col] = np.where(np.isnan(out[:, col]), out[:, col + 1] * np.sqrt(2), out[:, col])
    return np.nan_to_num(out, nan=1.0)


class OpticalChain:
    """Encode one event's waveforms of a system (:meth:`process`) and decode them
    (:meth:`reconstruct`). ``specs``: {system: :class:`OpticalSpec`} or overrides of :data:`SBND`;
    ``templates``: a template file name (``data/<name>_optical.npz``) or {key: response}."""

    def __init__(self, specs=None, device=None, templates="sbnd"):
        self.specs = specs if specs is not None and all(isinstance(v, OpticalSpec) for v in specs.values()) \
            else make_specs(specs)
        if isinstance(templates, dict):                           # e.g. from helix.optical.calibrate
            self.templates, self.template_meta = {k: np.asarray(v, np.float64) for k, v in templates.items()}, {}
        else:
            self.templates, self.template_meta = load_templates(templates)
        self.device = device
        self.kind = "torch" if device is not None else "numpy"
        self.wa = backend.import_backend_module("helix.core.hysteresis_ops", self.kind).window_any
        self._mf = {}
        self.capture = None                     # a list: (window indices, y, valid, photon train a) per batch
                                                # are appended to it (helix.optical.calibrate)

    def _filter(self, spec):
        if spec.response not in self._mf:
            self._mf[spec.response] = matched_filter(self.templates[spec.response])
        return self._mf[spec.response]

    def _store(self, g, cg, spec):
        """Stored masks (where the light's, or the departure from it, difference reaches its
        threshold) and the samples that mark them for the shard (a stored D1 block, the first
        sample - valid whenever the block is - of a coarser stored block no finer mark touches)."""
        h = self.templates[spec.response]
        valid = g["valid"]; m = valid.shape[0]; lev = spec.level
        w = valid.to(g["y"].dtype) if self.kind == "torch" else valid.astype(np.float64)
        y = g["y"] if self.kind == "torch" else g["y"].astype(np.float64)
        lam = spec.k * cg[:, :1] * float(np.sqrt(np.sum(h ** 2)))     # (rows, 1)
        if self.capture is not None:
            light, a = _deconvolve(y, w, lam, h, int(np.argmax(h)), return_a=True)
            self.capture.append((g["index"], _host(y * w), _host(valid), _host(a)))
        else:
            light = _deconvolve(y, w, lam, h, int(np.argmax(h)))
        _, dl = haar(spec.polarity * light, lev)
        roi = valid & False
        stored = [None] * lev
        for li in range(lev - 1, -1, -1):                            # D1 first
            B = 1 << (lev - li); nbk = dl[li].shape[1]
            sj = cg[:, 1 + li:2 + li]; ok = valid.reshape(m, nbk, B).any(-1)
            a = dl[li].abs() if self.kind == "torch" else np.abs(dl[li])
            s_ = (a >= spec.q * sj) & ok
            if spec.residual:                                        # what the light model does not explain
                d = g["details"][li]
                r_ = (d.to(dl[li].dtype) if self.kind == "torch" else d.astype(np.float64)) - dl[li]
                r_ = r_.abs() if self.kind == "torch" else np.abs(r_)
                s_ = s_ | ((r_ >= spec.residual * sj) & ok)
            stored[li] = s_
            r3 = roi.reshape(m, nbk, B)
            if B == 2:
                r3 |= s_[:, :, None]
            else:
                r3[:, :, 0] |= s_ & ~r3.any(-1)
        return roi & valid, stored

    def _batch(self, wf, idx, L, buf):
        """Rows ``idx`` (-1: empty padding row) as raw integer samples padded to L with the last
        sample, their valid mask, lengths and medians."""
        pad = np.asarray(idx) < 0
        n = np.where(pad, 0, np.asarray(wf["length"])[np.where(pad, 0, idx)]).astype(np.int64)
        o = np.where(pad, 0, np.asarray(wf["offset"])[np.where(pad, 0, idx)]).astype(np.int64)
        if self.kind == "numpy":
            ar = np.arange(L)
            pos = o[:, None] + np.minimum(ar[None, :], np.maximum(n - 1, 0)[:, None])
            X = np.where(n[:, None] > 0, buf[np.minimum(pos, max(len(buf) - 1, 0))], 0).astype(np.int32)
            valid = ar[None, :] < n[:, None]
            Xs = np.sort(np.where(valid, X, np.iinfo(np.int32).max), axis=1).astype(np.float64)
            r = np.arange(len(idx)); lo, hi = np.maximum((n - 1) // 2, 0), np.maximum(n // 2, 0)
            med = np.where(n > 0, 0.5 * (Xs[r, lo] + Xs[r, hi]), 0.0).astype(np.float32)
            return X, valid, n, med
        import torch
        from helix.tpc.coherent_gate_labels_ops_torch import _masked_median
        dev = self.device
        ar = torch.arange(L, device=dev)
        nt = torch.as_tensor(n, device=dev); ot = torch.as_tensor(o, device=dev)
        pos = ot[:, None] + torch.minimum(ar[None, :], torch.clamp_min(nt - 1, 0)[:, None])
        X = torch.where(nt[:, None] > 0, buf[torch.clamp(pos, max=max(buf.shape[0] - 1, 0))].to(torch.int32),
                        torch.zeros((), dtype=torch.int32, device=dev))
        valid = ar[None, :] < nt[:, None]
        med = torch.nan_to_num(_masked_median(X.to(torch.float32), valid, dim=1), nan=0.0)
        return X, valid, nt, med

    def process(self, system, wf):
        """One system's waveforms of one event (channel, t_us, length, offset, buf) -> dict(system,
        spec, channel, t_us, length, groups, noise, n_kept, n_samples). Each group (a batch of
        windows): index, L, roi (marks of the stored blocks), coarse, details [d_level..d_1],
        stored masks; ``noise``: per channel, the statistic's noise and each level's sigma."""
        spec = self.specs[system]
        u, lag, pre, post = self._filter(spec)
        n_all = np.asarray(wf["length"], np.int64)
        chans = np.asarray(wf["channel"])
        lev = spec.level
        if len(n_all) == 0:                                      # no window of this system in the event
            return dict(system=system, spec=dataclasses.asdict(spec), channel=chans, t_us=np.asarray(wf["t_us"]),
                        length=n_all, groups=[], n_kept=0, n_samples=0,
                        noise=dict(channel=np.zeros(0, np.int32), stat=np.zeros(0, np.float32),
                                   level=np.zeros((0, lev), np.float32)))
        buf = np.asarray(wf["buf"])
        if self.kind == "torch":
            import torch
            buf = torch.as_tensor(buf.view(np.int16) if buf.dtype == np.uint16 else buf.astype(np.int32)).to(self.device)
            if buf.dtype == torch.int16:                         # 14/15-bit codes: int16 on the way, int32 here
                buf = buf.to(torch.int32)
        # pass 1: statistic and transform per batch; noise per window
        groups, wnoise, rows = [], [], []
        for L, idx in batches(n_all, lev, self.kind == "torch"):
            nb = int((idx >= 0).sum())
            X, valid, n, med = self._batch(wf, idx, L, buf)
            y = X.to(torch.float32) - med[:, None] if self.kind == "torch" else X.astype(np.float32) - med[:, None]
            z = spec.polarity * (y - _baseline(y)) * valid
            stat = _correlate(z, u, lag) * valid
            coarse, details = haar(X, lev)
            groups.append(dict(index=idx, L=L, nb=nb, X=X, n=n, valid=valid, stat=stat, coarse=coarse, details=details,
                               y=spec.polarity * y))
            wnoise.append(_quiet_rms(stat, n)); rows.append(idx)
        idx_all = np.concatenate(rows)
        ch_rows = np.where(idx_all >= 0, chans[np.maximum(idx_all, 0)], -1)
        dead = (idx_all < 0) | (n_all[np.maximum(idx_all, 0)] == 0)

        def channel(per_group):                                   # one host round trip per estimate
            v = _host(_cat(per_group, 0)).astype(np.float64)
            v = v.reshape(len(v), -1)
            v[dead] = np.nan
            return _channel_median(v, ch_rows)
        # noise and pedestal from where there is no light: triggers at k (first from the
        # median-stretch noise) widened by the response's core span
        cn0, wnoise, wbase, off = channel(wnoise), [], [], 0
        for g in groups:
            m = len(g["index"]); c0 = cn0[off:off + m, :1]; off += m
            if self.kind == "torch":
                c0 = torch.as_tensor(c0, dtype=torch.float32, device=self.device)
            lit = self.wa((g["stat"] > spec.k * c0) & g["valid"], post, pre) | ~g["valid"]
            zero = torch.zeros(m, device=self.device) if self.kind == "torch" else np.zeros(m)
            est = [robust.clipped_stats(g["stat"], mask=lit, center=zero)[1]]
            for li, d in enumerate(g["details"]):
                j = lev - li
                cm = lit.reshape(m, d.shape[1], 1 << j).any(-1)
                pad = np.arange(d.shape[1])[None, :] >= (_host(g["n"]) >> j)[:, None]
                cm = cm | (torch.as_tensor(pad, device=self.device) if self.kind == "torch" else pad)
                est.append(robust.clipped_stats(d.to(torch.float32) if self.kind == "torch" else d.astype(np.float32),
                                                mask=cm, center=zero)[1])
            wnoise.append(torch.stack(est, 1) if self.kind == "torch" else np.stack(est, 1))
            wbase.append(robust.clipped_stats(g["y"], mask=lit)[0][:, None])
        cn = channel(wnoise)
        cbase = channel(wbase)
        uch = np.unique(chans)
        first = {c: i for i, c in reversed(list(enumerate(ch_rows)))}
        noise = dict(channel=uch.astype(np.int32), stat=np.array([cn[first[c], 0] for c in uch], np.float32),
                     level=np.array([cn[first[c], 1:] for c in uch], np.float32).reshape(len(uch), lev))
        # pass 2: the light and the stored masks
        kept, off = [], 0
        out = []
        for g in groups:
            m = len(g["index"])
            cg = cn[off:off + m]; cb = cbase[off:off + m]; off += m
            if self.kind == "torch":
                cg = torch.as_tensor(cg, dtype=torch.float32, device=self.device)
                cb = torch.as_tensor(cb, dtype=torch.float32, device=self.device)
            g["y"] = (g["y"] - cb) * g["valid"]
            roi, stored = self._store(g, cg, spec)
            sl = np.nonzero(g["index"] >= 0)[0]
            if self.kind == "torch":
                sl = torch.as_tensor(sl, device=self.device)
            idx = g["index"][g["index"] >= 0]
            roi, coarse = roi[sl], g["coarse"][sl]
            details, stored = [d[sl] for d in g["details"]], [s_[sl] for s_ in stored]
            kept.append(sum(s_.sum() for s_ in stored))
            out.append(dict(index=idx, L=g["L"], level=lev, roi=roi, coarse=coarse, details=details,
                            stored=stored, lengths=n_all[idx]))
        n_kept = int(sum(int(k) for k in kept)) if self.kind == "numpy" else int(torch.stack(kept).sum())
        n_kept += int(sum(-(-n_all // (1 << lev))))               # coarse sums within each window
        return dict(system=system, spec=dataclasses.asdict(spec), channel=chans, t_us=np.asarray(wf["t_us"]),
                    length=n_all, groups=out, noise=noise, n_kept=n_kept, n_samples=int(n_all.sum()))

    @staticmethod
    def reconstruct(res):
        """Decoded waveforms (float64, original lengths, raw ADC scale)."""
        out = [None] * len(res["length"])
        for g in res["groups"]:
            r = unhaar(_host(g["coarse"]), [_host(d) for d in g["details"]], [_host(s) for s in g["stored"]])
            for k, i in enumerate(g["index"]):
                out[i] = r[k, :g["lengths"][k]]
        return out
