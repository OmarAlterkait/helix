"""A faithful Python port of MicroBooNE's Wire-Cell signal processing ROI chain.

Ported from Wire-Cell Toolkit master @ b7bd2a1a (reference/wirecell/, fetched
2026-10-08), with the MicroBooNE configuration (cfg/pgrapher/experiment/uboone:
sp.jsonnet overrides no ROI parameter, so OmnibusSigProc.h defaults apply;
sp-filters.jsonnet sets the filters). Citations are to those files:
  OSP = sigproc_src_OmnibusSigProc.cxx, OSPh = ..._OmnibusSigProc.h,
  RF = sigproc_src_ROI_formation.cxx, RR = sigproc_src_ROI_refinement.cxx,
  SR = sigproc_src_SignalROI.cxx, PF = sigproc_src_PeakFinding.cxx,
  FLT = cfg_pgrapher_experiment_uboone_sp-filters.jsonnet.

What is ported (per plane; planes are independent in Wire-Cell):
  decon      M / R (2D), x Wire filter (FLT Wire_ind / Wire_col, applied once,
             OSP:1310-1333); per output a time filter: Wiener_tight (ROI),
             x ROI_tight_lf / ROI_tighter_lf / ROI_loose_lf (induction ROI
             finding), Gaus_wide (charge); restore_baseline (two-pass median,
             OSP:1032-1087) on every ROI-finding output and on the collection
             Gaussian output. Both axes zero-padded before the FFT as Wire-Cell
             does (OSP:863-891; pad sizes here: 200 ticks, 20 wires -- the
             intent is verified, the exact amounts are ASSUMED).
  formation  tight ROIs: per-wire cal_RMS (two-pass, RF:364-405) of the
             tight-LF output (collection: Wiener_tight), signed threshold
             th*rms+1, OR with the tighter-LF output (RF:407-549); pad 5
             (extend_ROI_self, RF:167-250); i/i+2 interpolation
             (create_ROI_connect_info, RF:252-362). Loose ROIs (induction):
             rebin x6, 3.5*6*rms capped at 10000, 0.7 edges, local-minimum walk,
             prominence / +-3-bin rules, gap merge, x6 back (find_ROI_loose,
             RF:705-957), extend to tight ROIs (extend_ROI_loose, RF:556-617).
  refinement load_data with the 3-sigma re-check on baseline-subtracted no-LF
             contents and the link graph (RR:280-700); CleanUpROIs (RR:704-876);
             generate_merge_ROIs (RR:878-1162); 2 x {BreakROIs (BreakROI +
             BreakROI1, RR:1878-2460, with a line-by-line port of PeakFinding::
             SearchHighRes, PF:65-479), CheckROIs (RR:1164-1287), CleanUpROIs};
             ShrinkROIs (RR:1614-1876), CheckROIs, CleanUpROIs;
             CleanUpInductionROIs (RR:1371-1612) / CleanUpCollectionROIs
             (RR:1289-1369); ExtendROIs (RR:2971-3156).
  charge     apply_roi on the Gaus_wide output (induction: straight line over
             the extended range, int truncation; collection: copy, RR:118-215);
             negatives set to 0 per sample (save_data, OSP:515-531).

Not ported (labelled where it matters): bad/noisy-channel handling (we have
none), multi-plane protection (off for MicroBooNE, OSPh:273), L1SP (a
follow-on for shorted-wire regions only), the pre-decon rebase (master
default; whether MicroBooNE used it is unverified), the coarse time offset
(our kernel is aligned to the hit tick).

Quirk reproduced on purpose: in CleanUpInductionROIs the U-plane branch
collects its bad ROIs but never removes them (RR:1460-1520 vs 1580-1606 for V).
``fix_u_cleanup=True`` applies the removal to U as well (a labelled variant).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

# ---------------- filters (FLT; Response.cxx:435-444) ----------------
WIENER_TIGHT = {"U": (57.5416 / 800 * 2, 4.10358), "V": (59.9306 / 800 * 2, 4.20820), "Y": (58.8802 / 800 * 2, 4.17455)}
GAUS_WIDE = (0.111408, 2.0)
WIRE_SIGMA = {"U": 1.4 / math.sqrt(math.pi), "V": 1.4 / math.sqrt(math.pi), "Y": 3.0 / math.sqrt(math.pi)}
LF_TIGHT, LF_TIGHTER, LF_LOOSE = 0.02, 0.1, 0.0025

DEFAULTS = dict(th_factor_ind=3.0, th_factor_col=5.0, pad=5, asy=0.1, rebin=6, l_factor=3.5, l_max_th=10000.0,
                l_factor1=0.7, l_short_length=3, l_jump_one_bin=0, r_th_factor=3.0, fake_low=500.0, fake_high=1000.0,
                r_pad=5, r_break_roi_loop=2, r_th_peak=3.0, r_sep_peak=6.0, r_low_peak_sep_threshold_pre=1200.0,
                r_max_npeaks=200, r_sigma=2.0, r_th_percent=0.1, fix_u_cleanup=False, fake_removal=True)
PAD_T, PAD_W = 200, 20


def hf(f, sigma, power, flag):
    out = np.exp(-0.5 * (np.abs(f) / sigma) ** power)
    if flag:
        out = np.where(f == 0, 0.0, out)
    return out


def lf(f, tau):
    return 1.0 - np.exp(-(np.abs(f) / tau) ** 2)


def restore_baseline(arr):
    """OSP:1032-1087, row by row, in place: median of nonzero samples, then the
    median of those within 500 of it, subtracted from nonzero samples."""
    for i in range(arr.shape[0]):
        row = arr[i]
        nz = row != 0
        if not nz.any():
            continue
        s = row[nz]
        b = np.median(s)
        t = s[np.abs(s - b) < 500]
        if t.size:
            b = np.median(t)
        row[nz] -= b
    return arr


def decon_outputs(M, plane, kernel, dt_us=0.5):
    """Wire-Cell decon outputs for one plane (see module doc)."""
    nw, nt = M.shape
    X = np.pad(M.astype(np.float64), ((PAD_W, PAD_W), (0, PAD_T)))
    NW, NT = X.shape
    R = kernel.rfft(NW, NT)
    Rabs = np.abs(R)
    eps = 1e-6 * Rabs.max()                      # same floor as the example; Wire-Cell divides raw (OSP:1274)
    Rs = np.where(Rabs > eps, R, eps * np.exp(1j * np.angle(R)))
    fw = 2 * np.abs(np.fft.fftfreq(NW))          # Wire-Cell wire 'frequency' = 2 x cycles/wire
    base = np.fft.rfft2(X) / Rs * hf(fw, WIRE_SIGMA[plane], 2.0, False)[:, None]
    ft = np.abs(np.fft.rfftfreq(NT, d=dt_us))    # MHz
    sw, pw = WIENER_TIGHT[plane]
    Fw = hf(ft, sw, pw, True)

    def out(F, rb=True):
        y = np.fft.irfft2(base * F[None, :], s=(NW, NT))[PAD_W:PAD_W + nw, :nt]
        y = np.ascontiguousarray(y)
        return restore_baseline(y) if rb else y

    d = dict(wiener_tight=out(Fw), gauss=out(hf(ft, *GAUS_WIDE, True), rb=(plane == "Y")))
    if plane != "Y":
        d["tight_lf"] = out(Fw * lf(ft, LF_TIGHT))
        d["tighter_lf"] = out(Fw * lf(ft, LF_TIGHTER))
        d["loose_lf"] = out(Fw * lf(ft, LF_LOOSE))
    return d


# ---------------- ROI formation (RF) ----------------
def cal_rms(x):
    """RF:364-405 (percentile spread, then a 5-sigma-clipped RMS about zero)."""
    p0, p1, p2 = np.percentile(x, [16, 50, 84])
    r = math.sqrt(((p2 - p1) ** 2 + (p1 - p0) ** 2) / 2.0)
    sel = np.abs(x) < 5.0 * r
    return float(np.sqrt(np.mean(x[sel] ** 2))) if sel.any() else 0.0


def runs(mask):
    """Contiguous True runs -> list of (begin, end) inclusive."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    b = np.nonzero(d == 1)[0]
    e = np.nonzero(d == -1)[0] - 1
    return list(zip(b.tolist(), e.tolist()))


def find_tight(sig1, sig2, th_factor, pad):
    """find_ROI_by_decon_itself (RF:407-549) + extend_ROI_self (RF:167-250).
    Returns per-wire lists and the per-wire rms."""
    nw, nt = sig1.shape
    rms = np.array([cal_rms(sig1[w]) for w in range(nw)])
    rois = []
    for w in range(nw):
        thr = th_factor * rms[w] + 1
        above = (sig1[w, :nt - 1] > thr) | (sig2[w, :nt - 1] > thr)   # j < nbins-1 (RF:485)
        temp = []
        for b, e in runs(above):
            b2, e2 = max(b - pad, 0), min(e + pad, nt - 1)
            if temp and b2 < temp[-1][1]:
                if e2 > temp[-1][1]:
                    temp[-1] = (temp[-1][0], e2)
            else:
                temp.append((b2, e2))
        rois.append(temp)
    return rois, rms


def connect_info(rois, asy):
    """create_ROI_connect_info (RF:252-362): ROIs on wires i and i+2 of similar
    length create the averaged ROI on i+1 unless it overlaps one there."""
    nw = len(rois)
    for i in range(nw - 2):
        for (s1, e1) in list(rois[i]):
            L1 = e1 - s1 + 1
            for (s2, e2) in list(rois[i + 2]):
                L2 = e2 - s2 + 1
                if abs(L2 - L1) < (L2 + L1) * asy:
                    s3, e3 = int((s1 + s2) / 2.0), int((e1 + e2) / 2.0)
                    if s3 < e3 and s3 <= e1 and s3 <= e2 and e3 >= s1 and e3 >= s2:
                        if not any(max(s3, a) < min(e3, b) for a, b in rois[i + 1]):
                            rois[i + 1].append((s3, e3))
    return rois


def _local_ave(s, b):
    n = len(s)
    return (s[(b - 1) % n] + s[b % n] + s[(b + 1) % n]) / 3.0


def _roi_end(s, b, th):
    n = len(s)
    end = b
    c = s[end]
    while c > th:
        end += 1
        c = s[end - n] if end >= n else s[end]
        if end >= n - 1:
            end = n - 1
            break
    while _local_ave(s, end + 1) < _local_ave(s, end) + 25:
        end += 1
        if end >= n - 1:
            end = n - 1
            break
    return end


def _roi_begin(s, b, th):
    n = len(s)
    beg = b
    c = s[beg]
    while c > th:
        beg -= 1
        c = s[beg + n] if beg < 0 else s[beg]
        if beg <= 0:
            beg = 0
            break
    while _local_ave(s, beg - 1) < _local_ave(s, beg) + 25:
        beg -= 1
        if beg <= 0:
            beg = 0
            break
    return beg


def find_loose(sig, P):
    """find_ROI_loose (RF:705-957) for every wire (bins in original ticks)."""
    nw, nt = sig.shape
    rb = P["rebin"]
    nb = nt // rb
    out = []
    for w in range(nw):
        s1 = sig[w]
        s2 = s1[:nb * rb].reshape(nb, rb).sum(1).tolist()
        th = min(cal_rms(s1) * rb * P["l_factor"], P["l_max_th"])
        th1 = th * P["l_factor1"]
        R1, mx1 = [], []
        j = 1
        while j < nb - 1:
            c, pc, nc = s2[j], s2[j - 1], s2[j + 1]
            flag = False
            if c > th:
                beg, end = _roi_begin(s2, j, th1), _roi_end(s2, j, th1)
                mb = beg + int(np.argmax(s2[beg:end + 1]))
                flag = True
            elif c > pc and c > nc:
                beg, end = _roi_begin(s2, j, pc), _roi_end(s2, j, nc)
                mb = beg + int(np.argmax(s2[beg:end + 1]))
                if s2[mb] - s2[beg] + s2[mb] - s2[end] > th * 2:
                    flag = True
                tb, te = max(mb - P["l_short_length"], beg), min(mb + P["l_short_length"], end)
                if s2[mb] - s2[tb] > th1 and s2[mb] - s2[te] > th1:
                    flag = True
            if flag:
                if R1 and beg <= R1[-1][1]:
                    R1[-1] = (R1[-1][0], end)
                    if s2[mb] > s2[mx1[-1]]:
                        mx1[-1] = mb
                else:
                    R1.append((beg, end)); mx1.append(mb)
                j = end if end < nb else nb
            j += 1
        if len(R1) > 1:                                   # gap merge (RF:880-926)
            rep = True
            while rep:
                rep = False
                for k in range(len(R1) - 1):
                    b, e = R1[k][0], R1[k + 1][1]
                    bc, ec = s2[b], s2[e]
                    ok = True
                    for jj in range(R1[k][1], R1[k + 1][0] + 1):
                        if s2[jj] - ((ec - bc) * (jj - b) / (e - b) + bc) < th1:
                            ok = False
                            break
                    if ok:
                        R1[k] = (R1[k][0], R1[k + 1][1]); del R1[k + 1]; rep = True
                        break
        out.append([(b * rb, e * rb + rb - 1) for b, e in R1])
    return out


def extend_loose(loose, tight):
    """extend_ROI_loose (RF:556-617)."""
    for i in range(len(loose)):
        temp = []
        for (s, e) in loose[i]:
            for (ts, te) in tight[i]:
                if ts < s < te:
                    s = ts
                if ts < e < te:
                    e = te
            if temp and s < temp[-1][1]:
                temp[-1] = (temp[-1][0], e)
            else:
                temp.append((s, e))
        loose[i] = temp
    return loose


# ---------------- PeakFinding::SearchHighRes (PF:65-479), line-by-line ----------------
def search_high_res(source, sigma=2.0, threshold=0.1, max_peaks=200, decon_iter=3, markov=True, aver_window=3):
    """Peak positions (float) as TSpectrum::SearchHighRes with backgroundRemove=False."""
    ssize = len(source)
    if ssize < 2 * sigma + 1 or sigma < 1 or not (0 < threshold < 100):
        return []
    n_it = int(7 * sigma + 0.5)
    size_ext = ssize + 2 * n_it
    shift = n_it
    k = int(2 * sigma + 0.5)
    l1low = 0.0
    if k >= 2:
        a = np.arange(k, dtype=float); b = np.asarray(source[:k], float)
        m0, m1, m2, l0, l1 = k, a.sum(), (a * a).sum(), b.sum(), (a * b).sum()
        det = m0 * m2 - m1 * m1
        l1low = (-l0 * m1 + l1 * m0) / det if det != 0 else 0.0
        if l1low > 0:
            l1low = 0.0
    ext = np.empty(size_ext)
    for i in range(size_ext):
        if i < shift:
            v = source[0] + l1low * (i - shift)
            ext[i] = v if v > 0 else 0.0
        elif i >= ssize + shift:
            v = source[ssize - 1]
            ext[i] = v if v > 0 else 0.0
        else:
            ext[i] = source[i - shift]
    orig = ext.copy()                                     # working_space[6*size_ext]
    if markov:
        ws2 = ext.copy()
        maxch = ws2.max()
        plocha = ws2.sum()
        if maxch <= 0:
            return []
        xmin, xmax = 0, size_ext - 1
        y = ws2 / maxch
        ii = np.arange(xmin, xmax)
        nip, nim = y[ii], y[ii + 1]
        sp = np.zeros(len(ii)); sm = np.zeros(len(ii))
        for l in range(1, aver_window + 1):
            a = y[np.minimum(ii + l, xmax)]
            den = np.where(a + nip <= 0, 1.0, np.sqrt(np.maximum(a + nip, 0)))
            sp += np.exp((a - nip) / den)
            a = y[np.maximum(ii - l + 1, xmin)]
            den = np.where(a + nim <= 0, 1.0, np.sqrt(np.maximum(a + nim, 0)))
            sm += np.exp((a - nim) / den)
        w = np.concatenate([[1.0], np.cumprod(sp / sm)])
        nom = w.sum()
        ext = w / nom * plocha
    # response (Gaussian, integer-truncated 1000*exp)
    idx = np.arange(size_ext, dtype=float)
    resp = np.floor(1000 * np.exp(-((idx - 3 * sigma) ** 2) / (2 * sigma * sigma)))
    nzr = np.nonzero(resp != 0)[0]
    lh = int(nzr[-1]) + 1 if len(nzr) else 0
    area = resp.sum()
    posit = int(np.argmax(resp))
    h = resp[:lh]
    y = np.abs(ext)
    # toeplitz-style correlations as in PF:318-361
    auto = np.correlate(h, h, mode="full")                 # length 2*lh-1, index i-imin with imin=-(lh-1)
    # cross[m] = sum_j h[j] y[i+j], i = m - (lh-1), m in [0, size_ext + lh - 1)
    cross = np.correlate(np.concatenate([np.zeros(lh - 1), y, np.zeros(lh - 1)]), h, mode="valid")
    yb = cross[:size_ext]                                   # working_space[2*size_ext + i], i in [0, size_ext)
    x = np.ones(size_ext)
    ws3 = np.zeros(size_ext)                                # working_space[3*size_ext]: starts at 0, keeps old values
    for _ in range(decon_iter):
        lda = np.convolve(x, auto, mode="same")             # sum_j auto[j+lh-1] x[i+j] (auto is symmetric)
        ok = (np.abs(yb) > 1e-5) & (np.abs(x) > 1e-5)
        ws3 = np.where(ok, np.where(lda != 0, yb / np.where(lda != 0, lda, 1.0), 0.0) * x, ws3)
        x = ws3.copy()
    rolled = np.zeros(size_ext)
    for i in range(size_ext):
        rolled[(i + posit) % size_ext] = x[i]
    dec = np.zeros(size_ext)
    maximum = maximum_decon = 0.0
    j = lh - 1
    for i in range(size_ext - j):
        if shift <= i < ssize + shift:
            dec[i] = area * rolled[i + j]
            maximum_decon = max(maximum_decon, dec[i])
            maximum = max(maximum, orig[i])
    lda = min(1.0, threshold) / 100.0
    pos = []
    for i in range(1, size_ext - 1):
        if dec[i] > dec[i - 1] and dec[i] > dec[i + 1] and shift <= i < ssize + shift:
            if dec[i] > lda * maximum_decon and orig[i] > threshold * maximum / 100.0:
                bsum = dec[i - 1] + dec[i] + dec[i + 1]
                a = ((i - 1 - shift) * dec[i - 1] + (i - shift) * dec[i] + (i + 1 - shift) * dec[i + 1]) / bsum
                a = min(max(a, 0.0), ssize - 1)
                if len(pos) < max_peaks:
                    pos.append(a)
    return pos


# ---------------- ROI refinement (RR) ----------------
class ROI:
    __slots__ = ("wire", "start", "end", "contents", "ext_start", "ext_end")

    def __init__(self, wire, start, end, signal):
        """SignalROI(plane, chid, start, end, signal) (SR:8-22): straight line
        between the end bins subtracted."""
        self.wire, self.start, self.end = wire, int(start), int(end)
        seg = np.asarray(signal[self.start:self.end + 1], np.float64)
        n = self.end - self.start
        base = seg[0] + (seg[-1] - seg[0]) * np.arange(n + 1) / n if n > 0 else np.full(1, seg[0])
        self.contents = seg - base
        self.ext_start, self.ext_end = self.start, self.end

    @classmethod
    def copy_of(cls, r):
        o = cls.__new__(cls)
        o.wire, o.start, o.end, o.contents = r.wire, r.start, r.end, r.contents.copy()
        o.ext_start, o.ext_end = r.start, r.end
        return o

    def overlap(self, o):
        return min(self.end, o.end) > max(self.start, o.start)

    def overlap_th(self, o, th, th1):
        a, b = max(self.start, o.start), min(self.end, o.end)
        if b <= a:
            return False
        c = self.contents[a - self.start:b - self.start + 1]
        c1 = o.contents[a - o.start:b - o.start + 1]
        return bool(np.any((c > th) & (c1 > th1)))

    def above(self, th):
        return runs(self.contents > th)


class Refiner:
    """ROI_refinement for one plane ('U', 'V' or 'Y')."""

    def __init__(self, plane, nw, rms, P):
        self.plane, self.nw, self.rms, self.P = plane, nw, rms, P
        self.tight = [[] for _ in range(nw)]
        self.loose = [[] for _ in range(nw)]
        self.front, self.back, self.contained = defaultdict(list), defaultdict(list), defaultdict(list)

    # links
    def link(self, a, b):           # a on wire w, b on wire w+1
        if b not in self.front[a]:
            self.front[a].append(b)
        if a not in self.back[b]:
            self.back[b].append(a)

    def unlink(self, a, b):
        if b in self.front.get(a, []):
            self.front[a].remove(b)
        if a in self.back.get(b, []):
            self.back[b].remove(a)

    def _drop_links(self, r):
        for n in list(self.front.get(r, [])):
            self.unlink(r, n)
        for p in list(self.back.get(r, [])):
            self.unlink(p, r)
        self.front.pop(r, None); self.back.pop(r, None)

    def load(self, tight_rois, loose_rois, ref):
        """load_data (RR:280-700)."""
        thf = self.P["r_th_factor"]
        for w in range(self.nw):
            th = self.rms[w] * thf
            for s, e in tight_rois[w]:
                r = ROI(w, s, e, ref[w])
                if not r.above(th):
                    continue
                self.tight[w].append(r)
                if self.plane == "Y" and w > 0:
                    for p in self.tight[w - 1]:
                        if r.overlap(p):
                            self.link(p, r)
            if self.plane != "Y":
                for s, e in loose_rois[w]:
                    r = ROI(w, s, e, ref[w])
                    if not r.above(th):
                        continue
                    self.loose[w].append(r)
                    if w > 0:
                        for p in self.loose[w - 1]:
                            if r.overlap(p):
                                self.link(p, r)
                    for t in self.tight[w]:
                        if t.overlap(r):
                            self.contained[r].append(t)

    def cleanup(self):
        """CleanUpROIs (RR:704-876): keep loose ROIs connected (front/back) to one containing a tight ROI."""
        saved = set()
        for w in range(self.nw):
            for r in self.loose[w]:
                if r in saved or not self.contained.get(r):
                    continue
                stack = [r]; saved.add(r)
                while stack:
                    t = stack.pop()
                    for n in self.front.get(t, []) + self.back.get(t, []):
                        if n not in saved:
                            saved.add(n); stack.append(n)
        for w in range(self.nw):
            for r in [r for r in self.loose[w] if r not in saved]:
                self._drop_links(r)
                self.loose[w].remove(r)
                self.contained.pop(r, None)

    def generate_merge(self):
        """generate_merge_ROIs (RR:878-1162)."""
        for w in range(self.nw):
            cov = set()
            for r in self.loose[w]:
                cov.update(self.contained.get(r, []))
            for t in [t for t in self.tight[w] if t not in cov]:
                n = ROI.copy_of(t)
                self.loose[w].append(n)
                self.contained[n] = [t]
                if w < self.nw - 1:
                    for x in self.loose[w + 1]:
                        if n.overlap(x):
                            self.link(n, x)
                if w > 0:
                    for x in self.loose[w - 1]:
                        if n.overlap(x):
                            self.link(x, n)

    def check(self):
        """CheckROIs (RR:1164-1287)."""
        thf = self.P["r_th_factor"]
        for w in range(self.nw):
            for r in self.loose[w]:
                th = thf * self.rms[w]
                for n in list(self.front.get(r, [])):
                    if not r.overlap_th(n, th, thf * self.rms[n.wire]):
                        self.unlink(r, n)
                for p in list(self.back.get(r, [])):
                    if not r.overlap_th(p, th, thf * self.rms[p.wire]):
                        self.unlink(p, r)

    def _replace(self, r, news):
        w = r.wire
        self.loose[w].remove(r)
        self.loose[w].extend(news)
        for n in list(self.front.get(r, [])):
            self.unlink(r, n)
            for x in news:
                if x.overlap(n):
                    self.link(x, n)
        for p in list(self.back.get(r, [])):
            self.unlink(p, r)
            for x in news:
                if x.overlap(p):
                    self.link(p, x)
        self.front.pop(r, None); self.back.pop(r, None)
        tl = self.contained.pop(r, [])
        for t in tl:
            for x in news:
                if x.overlap(t):
                    self.contained[x].append(t)

    def break_roi(self, r, rms):
        """BreakROI (RR:1878-2342); multi-plane protection off -> no protected sections."""
        P = self.P
        s0, e0 = r.start, r.end
        sig = r.contents.copy()
        th3 = P["r_th_factor"] * rms
        low_sep = max(P["r_low_peak_sep_threshold_pre"], P["r_sep_peak"] * rms)
        pos = search_high_res(sig.tolist(), P["r_sigma"], P["r_th_percent"], P["r_max_npeaks"])
        if len(pos) > 1:
            heights = [sig[int(round(p))] for p in pos]
            if sum(h > P["r_th_peak"] * rms for h in heights) > 1:
                order = sorted(int(p) + s0 for p in pos)    # vector<int>: double -> int truncation (RR:1925)
                npk = len(order)
                valley = {0: s0}
                mn = 1e9
                for k in range(order[0] - s0):
                    if sig[k] < mn:
                        mn = sig[k]; valley[0] = k + s0
                if valley[0] != s0:
                    order = [s0] + order; npk += 1
                    for j in range(s0, valley[0]):
                        if sig[j - s0] > sig[order[0] - s0]:
                            order[0] = j
                    valley[0] = s0
                for j in range(npk - 1):
                    mn = 1e9; valley[j + 1] = order[j]
                    for k in range(order[j] - s0, order[j + 1] - s0):
                        if sig[k] < mn:
                            mn = sig[k]; valley[j + 1] = k + s0
                valley[npk] = e0
                mn = 1e9
                for k in range(order[npk - 1] - s0, e0 - s0 + 1):
                    if sig[k] < mn:
                        mn = sig[k]; valley[npk] = k + s0
                if valley[npk] != e0:
                    npk += 1
                    valley[npk] = e0
                    order.append(e0)
                    for j in range(valley[npk - 1], valley[npk]):
                        if sig[j - s0] > sig[order[npk - 1] - s0]:
                            order[npk - 1] = j
                v1, p1 = {0: valley[0]}, {}
                n1 = 0
                j = 0
                while j < npk:
                    if n1 > 0 and sig[valley[j] - s0] < sig[v1[n1] - s0]:
                        v1[n1] = valley[j]
                    if sig[order[j] - s0] - sig[v1[n1] - s0] > low_sep:
                        p1[n1] = order[j]; n1 += 1
                        flag1 = False
                        for k in range(j + 1, npk + 1):
                            if sig[order[k - 1] - s0] > sig[p1[n1 - 1] - s0]:
                                p1[n1 - 1] = order[k - 1]
                            if sig[p1[n1 - 1] - s0] - sig[valley[k] - s0] > low_sep:
                                v1[n1] = valley[k]; j = k - 1; flag1 = True
                                break
                        if not flag1:
                            v1[n1] = valley[npk]; j = npk
                    j += 1
                v1[n1] = valley[npk]
                if n1 == 0:
                    n1 = 1; v1 = {0: valley[0], 1: valley[npk]}
                old = sig.copy(); sig = np.zeros_like(old)
                for j in range(n1):
                    a_, b_ = int(v1[j]), int(v1[j + 1])
                    ca, cb = old[a_ - s0], old[b_ - s0]
                    ks = np.arange(a_, b_ + 1)
                    base = ca + (cb - ca) * (ks - a_) / (b_ - a_) if b_ != a_ else np.full(len(ks), ca)
                    sig[a_ - s0:b_ - s0 + 1] = old[a_ - s0:b_ - s0 + 1] - base
                for t in self.contained.get(r, []):        # restore a wiped contained tight ROI (RR:2183-2207)
                    lo, hi = max(t.start, s0), min(t.end, e0)
                    if hi >= lo and not np.any(sig[lo - s0:hi - s0 + 1] != 0):
                        sig[lo - s0:hi - s0 + 1] = t.contents[lo - t.start:hi - t.start + 1]
        for _ in range(2):                                   # RR:2221-2331
            mins = []
            for b, e in runs(sig < th3):
                seg = sig[b:e + 1]
                z = np.nonzero(np.abs(seg) < 1e-3)[0]
                mins.append(b + (int(z[0]) if len(z) else int(np.argmin(seg))))
            old = sig; sig = np.zeros_like(old)
            for j in range(len(mins) - 1):
                a, b = mins[j], mins[j + 1]
                ks = np.arange(a, b + 1)
                base = old[a] + (old[b] - old[a]) * (ks - a) / (b - a) if b != a else np.full(len(ks), old[a])
                sig[a:b + 1] = old[a:b + 1] - base
        r.contents = sig

    def break_roi1(self, r):
        """BreakROI1 (RR:2344-2460): split at every |content| < 1e-3."""
        zs = (np.nonzero(np.abs(r.contents) < 1e-3)[0] + r.start).tolist()
        sigfull = np.zeros(r.end + 1)
        sigfull[r.start:] = r.contents
        news = []
        for i in range(len(zs) - 1):
            a, b = zs[i], zs[i + 1]
            if b > a:
                s = np.zeros(r.end + 1); s[a:b + 1] = sigfull[a:b + 1]
                news.append(ROI(r.wire, a, b, s))
        self._replace(r, news)

    def break_all(self):
        allr = []
        for w in range(self.nw):
            for r in list(self.loose[w]):
                self.break_roi(r, self.rms[w])
                allr.append(r)
        for r in allr:
            self.break_roi1(r)

    def shrink(self, r):
        """ShrinkROI (RR:1614-1853)."""
        thf = self.P["r_th_factor"]
        s0, e0 = r.start, r.end
        th1 = self.rms[r.wire] * thf
        flag = np.zeros(e0 - s0 + 1, bool)
        for t in self.contained.get(r, []):
            a, b = max(t.start, s0), min(t.end, e0)
            if b >= a:
                flag[a - s0:b - s0 + 1] = True
        for n in self.front.get(r, []) + self.back.get(r, []):
            th = self.rms[n.wire] * thf
            for b_, e_ in n.above(th):
                for j in range(b_, e_ + 1):
                    k = j + n.start - s0
                    if 0 <= k < len(flag) and r.contents[k] > th1:
                        flag[k] = True
        ns, ne = s0, e0
        f = np.nonzero(flag)[0]
        if len(f):
            ns, ne = int(f[0]) + s0, int(f[-1]) + s0
        ns, ne = max(ns - self.P["r_pad"], s0), min(ne + self.P["r_pad"], e0)
        news = []
        if ns >= 0 and ne > ns:
            s = np.zeros(e0 + 1); s[ns:ne + 1] = r.contents[ns - s0:ne - s0 + 1]
            news.append(ROI(r.wire, ns, ne, s))
        self._replace(r, news)

    def shrink_all(self):
        for r in [r for w in range(self.nw) for r in self.loose[w]]:
            self.shrink(r)

    def fake_cleanup(self, lists):
        """CleanUp{Induction,Collection}ROIs (RR:1289-1612)."""
        hi, lo = self.P["fake_high"], self.P["fake_low"]
        good = set(r for w in range(self.nw) for r in lists[w]
                   if (r.contents > hi).any() or r.contents.mean() > lo)
        bad = [r for w in range(self.nw) for r in lists[w] if r not in good
               and not any(n in good for n in self.front.get(r, []))
               and not any(p in good for p in self.back.get(r, []))]
        if self.plane == "U" and not self.P["fix_u_cleanup"]:
            return len(bad), 0                           # Wire-Cell master never removes U bad ROIs (RR:1460-1520)
        for r in bad:
            self._drop_links(r)
            lists[r.wire].remove(r)
            self.contained.pop(r, None)
        return len(bad), len(bad)

    def extend(self, lists):
        """ExtendROIs (RR:2971-3156)."""
        for w in range(self.nw):
            lists[w].sort(key=lambda r: (r.start, r.end))
            for r in lists[w]:
                r.ext_start, r.ext_end = r.start, r.end
                for n in self.front.get(r, []) + self.back.get(r, []):
                    r.ext_start = min(r.ext_start, n.start)
                    r.ext_end = max(r.ext_end, n.end)
            prev = None
            for r in lists[w]:
                if prev is not None and prev.ext_end > r.ext_start:
                    m = int(prev.end * 0.5 + r.start * 0.5)
                    prev.ext_end = m; r.ext_start = m
                prev = r


def roi_chain(d, plane, P=None, stages=None):
    """The whole chain on one plane's decon outputs ``d`` (decon_outputs).
    Returns (final ROI list per wire, rms per wire, stage masks if requested)."""
    P = dict(DEFAULTS, **(P or {}))
    nw, nt = d["wiener_tight"].shape
    keep = {}

    def snap(name, lists):
        if stages is not None:
            keep[name] = [[(r.start, r.end) if isinstance(r, ROI) else r for r in lst] for lst in lists]

    if plane == "Y":
        tight, rms = find_tight(d["wiener_tight"], d["wiener_tight"], P["th_factor_col"], P["pad"])
        tight = connect_info(tight, P["asy"])
        snap("tight", tight)
        Rf = Refiner(plane, nw, rms, P)
        Rf.load(tight, None, d["wiener_tight"])
        snap("load_check", Rf.tight)
        if P["fake_removal"]:
            Rf.fake_cleanup(Rf.tight)
        snap("fake_cleanup", Rf.tight)
        Rf.extend(Rf.tight)
        return Rf.tight, rms, keep
    tight, rms = find_tight(d["tight_lf"], d["tighter_lf"], P["th_factor_ind"], P["pad"])
    tight = connect_info(tight, P["asy"])
    snap("tight", tight)
    loose = extend_loose(find_loose(d["loose_lf"], P), tight)
    snap("loose", loose)
    Rf = Refiner(plane, nw, rms, P)
    Rf.load(tight, loose, d["wiener_tight"])
    snap("load_check", Rf.loose)
    Rf.cleanup(); Rf.generate_merge()
    snap("cleanup_merge", Rf.loose)
    for _ in range(P["r_break_roi_loop"]):
        Rf.break_all(); Rf.check(); Rf.cleanup()
    snap("break", Rf.loose)
    Rf.shrink_all(); Rf.check(); Rf.cleanup()
    snap("shrink", Rf.loose)
    if P["fake_removal"]:
        Rf.fake_cleanup(Rf.loose)
    snap("fake_cleanup", Rf.loose)
    Rf.extend(Rf.loose)
    return Rf.loose, rms, keep


def apply_roi(gauss, rois, plane):
    """apply_roi (RR:118-215) + save_data negatives -> 0 (OSP:515-531)."""
    nw, nt = gauss.shape
    out = np.zeros_like(gauss)
    mask = np.zeros(gauss.shape, bool)
    for w in range(nw):
        row = np.zeros(nt)
        for r in rois[w]:
            a, b = r.ext_start, r.ext_end
            seg = gauss[w, a:b + 1].astype(np.float64)
            if plane != "Y":
                base = seg[0] + (seg[-1] - seg[0]) * np.arange(b - a + 1) / (b - a) if b != a else np.full(1, seg[0])
                seg = seg - base
            row[a:b + 1] = np.trunc(seg)
            mask[w, a:b + 1] = True
        out[w] = row
    return np.maximum(out, 0.0), mask


def rois_mask(lists, shape):
    m = np.zeros(shape, bool)
    for w, lst in enumerate(lists):
        for it in lst:
            a, b = (it.ext_start, it.ext_end) if isinstance(it, ROI) else it
            m[w, a:b + 1] = True
    return m
