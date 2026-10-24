"""Configurable TPC compression chain for detectors with a readout map (SBND first).

One event, per TPC:
  1. pedestal: from the input (``file``) or a clipped mean per channel;
  2. channel status: rows are excluded (zeroed, and flagged invalid for the model) when
     dead (rms < ``dead_rms`` or not read out), noisy (rms > ``noisy_factor`` x the
     plane median) or with a baseline step (median of the first vs the last
     ``step_ticks`` ticks differing by more than ``step_k`` rms);
  3. DWT (one basis for every plane: periodization, time padded to a multiple of 2^level);
  4. coherent removal (:mod:`helix.tpc.coherent_gate`) on the readout's electronics groups
     (``coherent.groups``; null: none): SBND's front-end boards, across planes; the simulation's
     64-wire blocks, per plane;
  5. threshold (:mod:`helix.tpc.threshold_rows`): sigma on live rows per plane and band;
     hysteresis: seeds at the universal threshold (``kappa``), grown through ``floor``
     sigma. The ``drop_finest`` finest detail bands are zeroed (HELIX's FM tokenizes A4,
     D4, D3, D2: D1 is not used).

Configuration: a nested dict (:data:`SBND_DEFAULT` = the adopted SBND chain). Every
leaf outside ``dwt`` may instead be a per-plane mapping ``{"U": .., "V": .., "Y": ..}``. :func:`make_config`
merges overrides and validates; :func:`config_digest` hashes the canonical JSON so an
output records exactly which chain produced it.

Torch on ``device`` (default CPU): inputs are moved there once; every stage keeps them there.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from helix.core import robust, wavelet
from helix.core.robust import q50
from helix.tpc import coherent_gate as cg
from helix.tpc import threshold_rows as thr
from helix.tpc.readout import load_readout

SBND_DEFAULT = {
    "readout": "sbnd",
    "pedestal": {"method": "file", "clip_k": 3.5},
    "status": {"method": "cuts", "dead_rms": 0.3, "noisy_factor": 4.0, "clip_k": 3.5, "step_k": 10.0, "step_ticks": 500},
    # HELIX's basis (coif3, level 4), so SBND coefficients tokenize with the FM's band set
    "dwt": {"wavelet": "coif3", "level": 4, "mode": "periodization"},
    "coherent": {"groups": "femb", "cross_plane": True, "kgate": 3.0, "ksig": 3.0, "npass": 2, "tau": None,
                 "sigc_scope": "group"},
    "threshold": {"kappa": 1.0, "floor": 2.0, "sigma_scope": "plane",
                  "threshold_approx": True, "drop_finest": 1},
    # stored values rounded to a power-of-two step <= sigma / q of their (plane, band), sigma the
    # event's threshold sigma (helix.tpc.chain_coeff.quantize; 0 = exact float32): the input
    # gains a variance <= sigma^2 / (12 q^2), the noise in its mantissa bits costs no bytes
    "store": {"step_input": 16, "step_target": 32},
}

# The simulation (JAXTPC sensor files): its wire layout as the readout (``sim_config``), pedestals
# already subtracted by the reader, SBND's status cuts (they cut no simulated wire), the gate on the
# noise model's 64-wire blocks per plane (one coherent scale per band; occupancy tolerance 0.05),
# the same threshold and basis as SBND.
SIM_BASE = {
    "readout": {"kind": "layout", "num_wires": [], "group_size": 64},
    "pedestal": {"method": "file", "clip_k": 3.5},
    "status": {"method": "cuts", "dead_rms": 0.3, "noisy_factor": 4.0, "clip_k": 3.5, "step_k": 10.0, "step_ticks": 500},
    "dwt": {"wavelet": "coif3", "level": 4, "mode": "periodization"},
    "coherent": {"groups": "block", "cross_plane": False, "kgate": 3.0, "ksig": 3.0, "npass": 2, "tau": 0.05,
                 "sigc_scope": "global"},
    "threshold": {"kappa": 1.0, "floor": 2.0, "sigma_scope": "plane", "threshold_approx": True, "drop_finest": 1},
    "store": {"step_input": 16, "step_target": 32},
}


def sim_config(num_wires, overrides=None):
    """The simulation chain config for a sensor file's wire layout (``helix.tpc.io.sim_layout``)."""
    o = dict(overrides or {})
    o["readout"] = dict(SIM_BASE["readout"], num_wires=[[int(x) for x in r] for r in np.asarray(num_wires)])
    return make_config(o, base=SIM_BASE)


_CHOICES = {
    ("pedestal", "method"): ("file", "clipped_mean"),
    ("status", "method"): ("cuts", "none"),
    ("threshold", "sigma_scope"): ("plane", "channel", "group"),
    ("coherent", "sigc_scope"): ("global", "group"),
}


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not _is_per_plane(v):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _is_per_plane(v):
    return isinstance(v, dict) and len(v) > 0 and set(v) <= {"U", "V", "Y"}


def per_plane(v, plane):
    """A config leaf's value for ``plane`` ('U' / 'V' / 'Y')."""
    return v[plane] if _is_per_plane(v) else v


def make_config(overrides=None, base=None):
    """Defaults (``base``: SBND_DEFAULT) merged with ``overrides`` (nested dict), validated."""
    base = SBND_DEFAULT if base is None else base
    cfg = _merge(base, overrides)
    for (sec, key), allowed in _CHOICES.items():
        val = cfg[sec][key]
        for v in (val.values() if _is_per_plane(val) else [val]):
            if v not in allowed:
                raise ValueError(f"{sec}.{key} = {v!r}; allowed: {allowed}")
    for sec in cfg:
        if sec != "readout" and isinstance(cfg[sec], dict):
            unknown = set(cfg[sec]) - set(base.get(sec, {}))
            if unknown:
                raise ValueError(f"unknown {sec} keys: {sorted(unknown)}")
    for k, v in cfg["dwt"].items():
        if _is_per_plane(v):
            raise ValueError(f"dwt.{k}: one basis for all planes (per-plane values are not supported)")
    return cfg


def config_json(cfg) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def config_digest(cfg) -> str:
    return hashlib.sha256(config_json(cfg).encode()).hexdigest()[:16]


class TPCChain:
    """The chain for one detector configuration. Reuse it across events."""

    def __init__(self, cfg=None, device=None):
        self.cfg = make_config(cfg)
        self.readout = load_readout(self.cfg["readout"])
        self.device = "cpu" if device is None else device
        self.digest = config_digest(self.cfg)
        self._rows = {t: self.readout.rows(t) for t in range(self.readout.n_tpc)}
        self.timings = None                     # dict stage -> seconds when timing is on

    def enable_timing(self, on=True):
        """Accumulate wall time per stage (synchronising the GPU at stage boundaries)."""
        self.timings = {} if on else None
        self._t0 = None

    def _mark(self, name):
        if self.timings is None:
            return
        import time
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        t = time.perf_counter()
        if self._t0 is not None and name is not None:
            self.timings[name] = self.timings.get(name, 0.0) + t - self._t0
        self._t0 = t

    # ---------------------------------------------------------------- helpers
    def _to(self, x, dtype=None):
        """A tensor on the chain's device (``dtype``: np.float32 or bool)."""
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
        t = t.to(self.device, non_blocking=True)
        return t if dtype is None else t.to({np.float32: torch.float32, bool: torch.bool}[dtype])

    def _where_rows(self, img, ok):
        return torch.where(ok[:, None], img, torch.zeros((), dtype=img.dtype, device=img.device))

    def _dwt(self, img, wav, level):
        nt = img.shape[1]
        pad = (-nt) % (2 ** level)
        if pad:
            img = F.pad(img, (0, pad))
        bands, lev = wavelet.wavedec(img, wav, level, "periodization")
        if lev != level:
            raise ValueError(f"level {level} too deep for {nt} ticks and {wav}")
        return list(bands)

    def _rec(self, bands, wav, level, nt):
        return wavelet.reconstruct(bands, wav, nt)

    def _median(self, x):
        return q50(x, dim=0) if x.numel() else x.new_zeros(())

    # ---------------------------------------------------------------- stages
    def status(self, img, present):
        """Per-row validity and noise (rms) of pedestal-subtracted rows (see the module
        docstring: dead, noisy, baseline step)."""
        c = self.cfg["status"]
        _, sd, _ = robust.clipped_stats(img, k=c["clip_k"])
        sd = sd.nan_to_num(nan=0.0)
        if c["method"] == "none":                               # simulation: every present row is valid
            return present, sd
        n = min(int(c["step_ticks"]), img.shape[1] // 2)
        step = (q50(img[:, -n:], dim=1) - q50(img[:, :n], dim=1)).abs()
        ok = present & (sd > c["dead_rms"]) & (step <= c["step_k"] * sd)
        med = self._median(sd[ok])
        ok = ok & ((sd < c["noisy_factor"] * med) | (med <= 0))           # no host sync: med 0 -> no noisy cut
        return ok, sd

    def process_tpc(self, adc, ped, present, tpc, return_images=False):
        """``adc`` (channels, ticks), ``ped`` (channels,), ``present`` (channels,) bool
        for the WHOLE detector (numpy or torch). Returns one dict per plane."""
        cfg = self.cfg
        self._mark(None)
        imgs, pres = [], []
        for r in self._rows[tpc]:
            ri = self._to(r)
            raw = self._to(adc[ri], np.float32)
            if cfg["pedestal"]["method"] == "file":
                p = self._to(ped[ri], np.float32)
            else:
                p, _, _ = robust.clipped_stats(raw, k=cfg["pedestal"]["clip_k"])
                p = p.nan_to_num(nan=0.0)
            imgs.append(raw - p[:, None]); pres.append(self._to(present[ri], bool))
        return self._planes(imgs, pres, tpc, return_images)

    def process_images(self, images, tpc, present=None, return_images=False):
        """Pedestal-subtracted (wires, ticks) images of one TPC's planes, in plane order
        (simulation: the sensor reader's planes). Returns one dict per plane."""
        self._mark(None)
        imgs = [self._to(im, np.float32) for im in images]
        pres = [self._to(np.ones(im.shape[0], bool) if present is None else present[i], bool) for i, im in enumerate(images)]
        return self._planes(imgs, pres, tpc, return_images)

    def process_planes(self, planes, return_images=False):
        """``{gid: (wires, ticks) image}`` of pedestal-subtracted planes (gid = tpc * n_planes +
        plane), every TPC with all its planes at the readout's wire counts -> one dict per plane."""
        n_planes = len(self.readout.planes)
        by_tpc = {}
        for gid, img in planes.items():
            by_tpc.setdefault(int(gid) // n_planes, {})[int(gid) % n_planes] = img
        out = []
        for tpc in sorted(by_tpc):
            pl = by_tpc[tpc]
            if tpc >= self.readout.n_tpc:
                raise ValueError(f"plane gids of tpc {tpc}, but the readout has {self.readout.n_tpc} tpcs")
            if sorted(pl) != list(range(n_planes)):
                raise ValueError(f"tpc {tpc}: planes {sorted(pl)} present; the chain needs all {n_planes}")
            for p, rows in enumerate(self.readout.rows(tpc)):
                if int(pl[p].shape[0]) != len(rows):
                    raise ValueError(f"gid {tpc * n_planes + p}: image has {int(pl[p].shape[0])} wires, "
                                     f"the readout {len(rows)} — geometry must match")
            out += self.process_images([pl[p] for p in range(n_planes)], tpc, return_images=return_images)
        return out

    def dwt_rows(self, img):
        """The chain's transform of one plane image (no gate, no threshold), the dropped finest
        bands zeroed: the basis of the noise-free target (helix.build)."""
        d = self.cfg["dwt"]
        b = self._dwt(self._to(img, np.float32), d["wavelet"], d["level"])
        nd = int(self.cfg["threshold"]["drop_finest"])
        for i in range(len(b) - nd, len(b)):
            b[i] = b[i] * 0
        return b

    def _planes(self, raw_imgs, pres, tpc, return_images):
        cfg, ro = self.cfg, self.readout
        planes = ro.planes
        rows_bp = self._rows[tpc]
        imgs, valid, sds = [], [], []
        for img, pr in zip(raw_imgs, pres):
            ok, sd = self.status(img, pr)
            imgs.append(self._where_rows(img, ok)); valid.append(ok); sds.append(sd)
        self._mark("pedestal_status")
        nt = imgs[0].shape[1]
        d = cfg["dwt"]
        wav, lev = d["wavelet"], d["level"]
        coh = cfg["coherent"]
        valid_np = [np.asarray(v.cpu() if hasattr(v, "cpu") else v) for v in valid]
        bands = [self._dwt(im, wav, lev) for im in imgs]
        if coh["groups"] is not None:
            gkw = dict(kgate=coh["kgate"], ksig=coh["ksig"], npass=coh["npass"], tau=coh["tau"],
                       sigc_scope=coh["sigc_scope"])
            labels = [ro.labels(coh["groups"], r, v) for r, v in zip(rows_bp, valid_np)]
            if coh["cross_plane"]:
                bands = cg.coherent_gate_planes(bands, labels, **gkw)
            else:
                bands = [cg.coherent_gate_labeled(b, lab, **gkw) for b, lab in zip(bands, labels)]
        self._mark("dwt_gate")
        tc = cfg["threshold"]
        nd = int(tc["drop_finest"])
        jobs = []
        for p, pl in enumerate(planes):
            b = list(bands[p])
            for i in range(len(b) - nd, len(b)):
                b[i] = b[i] * 0
            kw = dict(kappa=per_plane(tc["kappa"], pl), valid=valid[p], sigma_scope=per_plane(tc["sigma_scope"], pl),
                      threshold_approx=tc["threshold_approx"], floor=per_plane(tc["floor"], pl))
            if kw["sigma_scope"] == "group":
                kw["groups"] = ro.labels(coh["groups"], rows_bp[p], valid_np[p])
            jobs.append((b, kw))
        out = []
        for p, (pl, (th, n_kept, n_total, sig)) in enumerate(zip(planes, thr.threshold_planes_rows(jobs))):
            res = dict(plane=pl, tpc=tpc, rows=rows_bp[p], valid=valid[p], noise_rms=sds[p], bands=th,
                       n_kept=n_kept, n_total=n_total, sigma_per_band=sig, wavelet=wav, level=lev,
                       n_ticks=nt, band_lengths=[int(x.shape[1]) for x in th])
            if return_images:
                res["image"] = imgs[p]
                res["cleaned"] = self._rec(bands[p], wav, lev, nt)
            out.append(res)
        self._mark("threshold")
        return out

    def process_event(self, adc, ped, present, return_images=False):
        adc, ped, present = self._to(adc), self._to(ped, np.float32), self._to(present, bool)
        return [pl for t in range(self.readout.n_tpc) for pl in self.process_tpc(adc, ped, present, t, return_images)]
