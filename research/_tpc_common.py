"""Shared helpers for the TPC figures: load real doraemon truth, add the
pimm_data forward noise model, run the helix pipeline, score it.

This is the *current* post-refactor path end-to-end:
  pimm_data  : load clean truth (JAXTPCDataset) + geometry registry + forward
               noise (generate_noise: coherent + incoherent) + digitize
  helix      : remove_coherent  ->  sparsify  ->  reconstruct  (numpy backend)
"""
from __future__ import annotations
import numpy as np

from pimm_data import JAXTPCDataset, load_plane_registry
from pimm_data.jaxtpc import canonical_plane_id
from pimm_data.noise import generate_noise, digitize
from helix.tpc.config import DetectorConfig
from helix.tpc.coherent import remove_coherent
from helix.core.wavelet import sparsify, reconstruct

DATA_ROOT = "/sdf/home/o/omara/data/omara/doraemon"
SPLIT = "run_0026628546"
GROUP_SIZE = 64

_reg = None
_ds = None


def _setup():
    global _reg, _ds
    if _reg is None:
        _reg = load_plane_registry("cubic_wireplane_geometry.json")
        _ds = JAXTPCDataset(data_root=DATA_ROOT, split=SPLIT, modalities=("sensor",))
    return _reg, _ds


def load_clean(event, label):
    """Dense clean-truth image (n_wires, n_ticks) + geometry."""
    reg, ds = _setup()
    g = reg[canonical_plane_id(label)]
    nw, nt = g["n_wires"], g["n_ticks"]
    wl = np.asarray(g["wire_lengths"], np.float32)
    cols = ds.get_data(event)["sensor"]["raw"][label]
    clean = np.zeros((nw, nt), np.float32)
    clean[cols["wire"], cols["time"]] = cols["value"]
    return clean, nt, wl, int(g["pedestal"])


def make_noisy(clean, wl, ped, seed):
    """Forward model: clean + (coherent+incoherent) noise, then digitize."""
    rng = np.random.default_rng(seed)
    noise = generate_noise(clean.shape, rng=rng, wire_lengths_m=wl,
                           incoherent=True, coherent=True, group_size=GROUP_SIZE)
    return digitize(clean + noise, ped)


def intrinsic_sigma(wl):
    """True per-wire intrinsic noise sigma from the detector model.

    remove_coherent MUST get this (production passes it). With a MAD estimate
    instead, dense-signal planes over-estimate sigma -> the signal mask is too
    permissive -> the coherent estimate is signal-contaminated -> coherent
    structure SURVIVES removal (verified: residual coherent content 1.70 vs
    0.61 ADC). See research/diag_coherent.py."""
    return DetectorConfig().wire_sigma_intrinsic(wl)


def f0(clean, recon):
    """Charge-fidelity over signal pixels: 1 - sum|recon-clean|/sum|clean|."""
    sig = np.abs(clean) > 0
    tc = float(np.abs(clean)[sig].sum())
    return 1.0 - float(np.abs(recon - clean)[sig].sum()) / max(tc, 1e-9)


def rms_off(clean, x):
    """Residual RMS on the OFF-signal pixels (the noise that survives)."""
    off = np.abs(clean) <= 0
    return float(np.sqrt(np.mean((x - clean)[off] ** 2))) if off.any() else 0.0
