"""Large-scale rate-distortion campaign — many methods x wavelets x levels x bits.

Multi-GPU, 100 events, incremental save (per-worker jsonl, flushed per event so a
long run never loses progress). Frontier philosophy (settled): anchor at the
denoising point, drive compression with quantization; measure SIGNAL fidelity
(peak vs input on signal chunks; rms vs a common denoised reference), report rate
honestly. Cranking the threshold is included only to MAP where it leaves the
frontier, not as the primary knob.

Stages (argv[1]):
  breadth : wavelets x levels x {core methods} x bits  (find good region)
  depth   : winners x fine knobs (kappa, bits) x {hard/soft/garrote, sure, ...}
Usage: python scripts/optical/sweep_big.py <stage> [n_events]
"""
from __future__ import annotations
import sys, os, json, time, math
import numpy as np
import torch.multiprocessing as mp

PATH = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
OUT = "/sdf/group/neutrino/omara/helix/temp/figures/big"
NOISE = 2.566
REF_WL, REF_LEVEL, REF_KAPPA = "coif3", 10, 1.0   # common denoised reference
COMMON_PAD = 65536                                 # all events padded to this length
                                                   # (>= max chunk; zeros -> ~0 coeffs, thresholded away;
                                                   #  baseline uses true lengths so padding is cost-free)

WAVELETS_BREADTH = ["db2", "db4", "db6", "db8", "db10", "coif1", "coif2", "coif3",
                    "coif4", "coif5", "sym4", "sym5", "sym6", "sym8",
                    "bior2.4", "bior3.5", "bior4.4", "bior6.8", "dmey"]
LEVELS_BREADTH = [6, 8, 10, 12]


# ---------------- threshold methods (all operate on torch coeff lists) ----------------
def _apply(c, t, func, torch):
    a = c.abs()
    if func == "hard":
        return c * (a >= t)
    if func == "soft":
        return torch.sign(c) * (a - t).clamp_min(0)
    return torch.where(a >= t, c - t * t / torch.where(a == 0, torch.ones_like(c), c), torch.zeros_like(c))  # garrote


def _sure_t(c, sig, torch):
    """Per-row SURE threshold (in absolute units). c:(n,L), sig:(n,)."""
    y = (c / sig[:, None]).abs()
    a, _ = torch.sort(y, dim=1)                       # ascending
    n, L = a.shape
    cs = torch.cumsum(a ** 2, dim=1)
    i = torch.arange(L, device=a.device, dtype=a.dtype)[None, :]
    risk = L - 2 * (i + 1) + cs + (L - 1 - i) * (a ** 2)
    imin = risk.argmin(1)
    return (sig * a.gather(1, imin[:, None]).squeeze(1))[:, None]


def threshold(coeffs, sig, cfg, torch):
    det = coeffs[1:]
    out = [coeffs[0]]
    m, func = cfg["method"], cfg.get("func", "hard")
    kap = cfg.get("kappa", 1.0)
    if m == "visu":
        N = sum(c.shape[1] for c in coeffs)
        t = (kap * sig * math.sqrt(2 * math.log(N)))[:, None]
        for c in det:
            out.append(_apply(c, t, func, torch))
    elif m == "leveldep":
        for c in det:
            t = (kap * sig * math.sqrt(2 * math.log(max(c.shape[1], 2))))[:, None]
            out.append(_apply(c, t, func, torch))
    elif m == "minimax":
        N = sum(c.shape[1] for c in coeffs)
        lam = max(0.0, 0.3936 + 0.1829 * math.log2(N))
        t = (kap * sig * lam)[:, None]
        for c in det:
            out.append(_apply(c, t, func, torch))
    elif m == "bayes":
        sn2 = sig ** 2
        for c in det:
            sx = ((c ** 2).mean(1) - sn2).clamp_min(1e-12).sqrt()
            out.append(_apply(c, (sn2 / sx)[:, None] * kap, "hard", torch))
    elif m == "sure":
        for c in det:
            out.append(_apply(c, kap * _sure_t(c, sig, torch), func, torch))
    elif m == "hybrid":      # SURE-hybrid (MATLAB 'heursure'): universal where sparse, else SURE
        for c in det:
            L = c.shape[1]
            test = ((c / sig[:, None]) ** 2).sum(1) - L
            crit = (math.log2(max(L, 2)) ** 1.5) / math.sqrt(L)
            t_u = (sig * math.sqrt(2 * math.log(max(L, 2))))[:, None]
            t = torch.where((test / L < crit)[:, None], t_u, _sure_t(c, sig, torch))
            out.append(_apply(c, kap * t, func, torch))
    elif m == "fdr":         # false-discovery-rate thresholding (Abramovich-Benjamini)
        q = cfg.get("q", 0.05)
        for c in det:
            L = c.shape[1]
            a, _ = torch.sort((c.abs() / sig[:, None]), dim=1, descending=True)
            k = torch.arange(1, L + 1, device=c.device, dtype=c.dtype)
            arg = (1 - q * k / L).clamp(-0.999999, 0.999999)
            z = math.sqrt(2) * torch.erfinv(arg)                       # per-rank critical value
            passm = a >= z[None, :]
            last = (passm * k).max(1).values.long()                    # largest passing rank (0 = none)
            t = torch.where(last > 0, sig * a.gather(1, (last - 1).clamp(min=0)[:, None]).squeeze(1),
                            torch.full_like(sig, float("inf")))
            out.append(_apply(c, t[:, None], "hard", torch))
    elif m == "block":       # block James-Stein (Cai BlockJS)
        bs = cfg.get("block", 8); lam = 4.5057
        for c in det:
            n, L = c.shape
            pad = (-L) % bs
            cc = torch.nn.functional.pad(c, (0, pad)).reshape(n, -1, bs)
            S2 = (cc ** 2).sum(-1).clamp_min(1e-12)
            shrink = (1 - lam * bs * (sig ** 2)[:, None] / S2).clamp_min(0)
            out.append((cc * shrink[..., None]).reshape(n, -1)[:, :L])
    elif m in ("energy", "topk"):
        flat = torch.cat(det, 1); aa = flat.abs(); D = aa.shape[1]
        if m == "topk":
            kk = max(1, int(cfg["keep"] * D)); tv = torch.topk(aa, kk, 1).values[:, -1:]
        else:
            srt, _ = torch.sort(aa, 1, descending=True); csm = torch.cumsum(srt ** 2, 1)
            kc = (csm < cfg["e"] * csm[:, -1:].clamp_min(1e-30)).sum(1).clamp(0, D - 1)
            tv = srt.gather(1, kc[:, None])
        for c in det:
            out.append(_apply(c, tv, "hard", torch))
    return out


def raw_sigma(chunks):
    """Per-chunk noise sigma from the UNPADDED chunk's finest db1 detail (MAD).
    Padding-independent — avoids the zero-padding collapsing the MAD estimate."""
    import pywt
    out = []
    for c in chunks:
        d = pywt.wavedec(c.astype(np.float64), "db1", level=1, mode="periodization")[-1]
        out.append(np.median(np.abs(d)) / 0.6745)
    return np.asarray(out, np.float32)


def quant(coeffs, bits, torch):
    if bits >= 32:
        return coeffs
    out = []
    for c in coeffs:
        amax = c.abs().amax(1, keepdim=True).clamp_min(1e-12)
        lv = (1 << (bits - 1)) - 1
        out.append(torch.round(c / (amax / lv)) * (amax / lv))
    return out


def rate_and_dist(x, valid, sigmask, ref, ref_pk, coeffs, bits, rec, torch):
    n = x.shape[0]
    storage = torch.zeros(n, device=x.device)
    for c in coeffs:
        Lb = c.shape[1]; K = (c != 0).sum(1).float()
        idx = math.ceil(math.log2(max(Lb, 2)))
        storage += torch.where(K <= 0.5 * Lb, K * (bits + idx), torch.full_like(K, float(Lb * bits)))
    comp = float((valid.sum(1).float() * 16).sum() / storage.sum().clamp_min(1))   # event-total (honest)
    xm = x * valid; rm = rec * valid
    pk_in = xm.min(1).values
    peak = float(((rm.min(1).values - pk_in).abs() / pk_in.abs().clamp_min(1e-9))[sigmask].mean()) if sigmask.any() else 0.0
    drf = (rm - ref * valid)
    rms_ref = float((drf.pow(2).sum(1).sqrt() / (ref * valid).pow(2).sum(1).sqrt().clamp_min(1e-9))[sigmask].mean()) if sigmask.any() else 0.0
    rel_noisy = float(((rm - xm).pow(2).sum(1).sqrt() / xm.pow(2).sum(1).sqrt().clamp_min(1e-9)).mean())
    kept = int(sum(int((c != 0).sum()) for c in coeffs))
    return dict(comp=comp, peak_err=peak, rms_vs_ref=rms_ref, rel_vs_noisy=rel_noisy, kept=kept)


def gen_configs(stage):
    import pywt
    cfgs = []
    if stage == "complete_wav":             # ALL 105 PR-valid wavelets x levels, fixed VisuShrink-hard
        allw = [w for w in pywt.wavelist(kind="discrete") if w != "dmey"]
        for wl in allw:
            for lev in (6, 8, 10):
                cfgs.append(dict(method="visu", func="hard", kappa=1.0, wavelet=wl, level=lev, bits=8))
    elif stage == "complete_lev":           # fine level scan on representative wavelets
        for wl in ("haar", "db4", "coif3", "sym6", "bior4.4"):
            for lev in range(1, 14):
                cfgs.append(dict(method="visu", func="hard", kappa=1.0, wavelet=wl, level=lev, bits=8))
    elif stage == "complete_meth":          # all threshold methods on the best wavelet/level
        meths = [dict(method="visu", func="hard"), dict(method="visu", func="soft"),
                 dict(method="visu", func="garrote"), dict(method="leveldep", func="hard"),
                 dict(method="minimax", func="soft"), dict(method="sure", func="soft"),
                 dict(method="sure", func="hard"), dict(method="bayes"),
                 dict(method="hybrid", func="soft"), dict(method="fdr"), dict(method="block")]
        for wl in ("coif3", "sym6"):
            for mth in meths:
                for kap in (0.6, 1.0, 1.5, 2.0):
                    for bits in (6, 8, 12):
                        cfgs.append({**mth, "wavelet": wl, "level": 10, "kappa": kap, "bits": bits})
    elif stage == "breadth":
        for wl in WAVELETS_BREADTH:
            for lev in LEVELS_BREADTH:
                base = [dict(method="visu", func="hard", kappa=1.0),
                        dict(method="sure", func="soft", kappa=1.0),
                        dict(method="bayes", kappa=1.0),
                        dict(method="leveldep", func="hard", kappa=1.0),
                        dict(method="minimax", func="soft", kappa=1.0),
                        dict(method="energy", e=0.99),
                        dict(method="topk", keep=0.01)]
                for b in base:
                    for bits in (8, 12):
                        cfgs.append({**b, "wavelet": wl, "level": lev, "bits": bits})
    elif stage == "depth":
        wls = json.load(open(f"{OUT}/depth_wavelets.json")) if os.path.exists(f"{OUT}/depth_wavelets.json") \
            else ["coif3", "coif4", "sym6", "coif5"]
        for wl in wls:
            for lev in (8, 10, 12):
                for meth in [dict(method="visu", func="hard"), dict(method="visu", func="soft"),
                             dict(method="visu", func="garrote"), dict(method="leveldep", func="hard")]:
                    for kap in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
                        for bits in (6, 8, 10, 12, 16):
                            cfgs.append({**meth, "wavelet": wl, "level": lev, "kappa": kap, "bits": bits})
    return cfgs


def worker(task):
    dev_id, events, stage = task
    import torch
    torch.cuda.set_device(dev_id); dev = f"cuda:{dev_id}"
    from helix.core import backend; backend.set_backend("torch")
    from helix.optical import config_from_file
    from helix.optical.io import read_event_chunks, pad_batch
    from helix.core.wavelet_ops_torch import _wavedec, _waverec
    cfg0 = config_from_file(PATH)
    configs = gen_configs(stage)
    # group by (wavelet, level) to compute DWT once
    from collections import defaultdict
    groups = defaultdict(list)
    for ci, c in enumerate(configs):
        groups[(c["wavelet"], c["level"])].append(ci)
    fout = open(f"{OUT}/{stage}_gpu{dev_id}.jsonl", "w")
    for ev in events:
        t_ev = time.perf_counter()
        ec = read_event_chunks(PATH, ev, cfg0)
        bnp, lengths = pad_batch(ec.chunks, max(LEVELS_BREADTH))
        x = torch.as_tensor(bnp, device=dev); lens = torch.as_tensor(lengths, device=dev)
        if x.shape[1] < COMMON_PAD:                       # common length -> consistent level across events
            x = torch.nn.functional.pad(x, (0, COMMON_PAD - x.shape[1]))
        L = x.shape[1]; valid = torch.arange(L, device=dev)[None, :] < lens[:, None]
        sigmask = x.abs().amax(1) > 10 * NOISE
        sig = torch.as_tensor(raw_sigma(ec.chunks), device=dev).clamp_min(1e-9)  # correct, padding-free
        # common reference
        rc = _wavedec(x, REF_WL, REF_LEVEL)
        ref = _waverec(threshold(rc, sig, dict(method="visu", func="hard", kappa=REF_KAPPA), torch), REF_WL)[:, :L]
        ref_pk = (ref * valid).min(1).values
        recs = []
        for (wl, lev), idxs in groups.items():
            try:
                coeffs = _wavedec(x, wl, lev)
            except Exception as e:
                continue
            for ci in idxs:
                cfg = configs[ci]
                try:
                    tc = threshold(coeffs, sig, cfg, torch)
                    qc = quant(tc, cfg["bits"], torch)
                    rec = _waverec(qc, wl)[:, :L]
                    m = rate_and_dist(x, valid, sigmask, ref, ref_pk, qc, cfg["bits"], rec, torch)
                    m.update(event=ev, **{k: cfg[k] for k in cfg})
                    recs.append(m)
                except Exception as e:
                    recs.append(dict(event=ev, error=str(e)[:80], **cfg))
        for r in recs:
            fout.write(json.dumps(r) + "\n")
        fout.flush()
        print(f"[gpu{dev_id}] {ev} {len(recs)} cfgs {time.perf_counter()-t_ev:.1f}s", flush=True)
    fout.close()
    return f"gpu{dev_id} done {len(events)} events"


def main():
    import torch
    stage = sys.argv[1] if len(sys.argv) > 1 else "breadth"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    os.makedirs(OUT, exist_ok=True)
    from helix.optical import list_events
    events = list_events(PATH)[:n]
    ng = torch.cuda.device_count()
    tasks = [(i, events[i::ng], stage) for i in range(ng) if events[i::ng]]
    print(f"STAGE {stage}: {len(events)} events x {len(gen_configs(stage))} configs over {len(tasks)} GPUs", flush=True)
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        for r in pool.map(worker, tasks):
            print(r, flush=True)
    print(f"STAGE {stage} done in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
