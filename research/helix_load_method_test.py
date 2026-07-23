"""Thorough helix functional test: load real doraemon events -> add forward noise
(pimm_data.noise) -> run helix's method (remove_coherent -> wavelet sparsify ->
reconstruct) -> report F0 / noise reduction / compression.

Geometry (n_wires, per-wire lengths, pedestal, num_time_steps) comes from the
JAXTPC-exported config registry via pimm_data.load_plane_registry — NOT hardcoded.
End-to-end across loader (pimm_data) + config geometry + noise (pimm_data) +
denoise+compress (helix) on real data.
"""
import numpy as np
from pimm_data import JAXTPCDataset, load_plane_registry
from pimm_data.jaxtpc import canonical_plane_id
from pimm_data.noise import generate_noise, digitize
from helix.tpc.pipeline import process_plane
from helix.tpc.config import DetectorConfig

DATA_ROOT = '/sdf/home/o/omara/data/omara/doraemon'
SPLIT = 'run_0026628546'
EVENTS = [0, 1]
PLANES = ['volume_0_Y', 'volume_0_U']  # collection (clean) + induction (harder)

reg = load_plane_registry('cubic_wireplane_geometry.json')   # config-derived geometry
ds = JAXTPCDataset(data_root=DATA_ROOT, split=SPLIT, modalities=('sensor',))
print(f"loaded {len(ds)} events; geometry registry planes={sorted(reg)}\n")


def f0_noise(clean, recon):
    sig = np.abs(clean) > 0
    tc = float(np.abs(clean)[sig].sum())
    f0 = 1.0 - float(np.abs(recon - clean)[sig].sum()) / max(tc, 1e-9)
    off = ~sig
    nrms = float(np.sqrt(np.mean((recon - clean)[off] ** 2))) if off.any() else 0.0
    return f0, nrms


def rms_off(clean, x):
    off = np.abs(clean) <= 0
    return float(np.sqrt(np.mean((x - clean)[off] ** 2))) if off.any() else 0.0


print(f"{'plane':>12} {'evt':>3} {'nwires':>6} {'nsig':>6} {'n_kept':>8} "
      f"{'n_total':>9} {'kept/wire':>9} {'noise_out':>9} {'reject':>7} {'F0':>7} {'compress':>9}")
print("-" * 100)
for label in PLANES:
    g = reg[canonical_plane_id(label)]
    nw, nt, ped, wl = g['n_wires'], g['n_ticks'], g['pedestal'], g['wire_lengths']
    cfg = DetectorConfig(group_size=64, num_time_steps=nt, plane_labels=(label,))
    for e in EVENTS:
        cols = ds.get_data(e)['sensor']['raw'][label]
        assert int(cols['wire'].max()) < nw
        clean = np.zeros((nw, nt), np.float32)
        clean[cols['wire'], cols['time']] = cols['value']
        rng = np.random.default_rng(1000 + e)
        noise = generate_noise(clean.shape, rng=rng, wire_lengths_m=wl,
                               incoherent=True, coherent=True, group_size=64)
        noisy = digitize(clean + noise, ped)
        out = process_plane(noisy, cfg)               # helix: remove_coherent + sparsify + reconstruct
        recon = np.asarray(out.reconstructed)[:, :nt]
        f0, n_out = f0_noise(clean, recon)
        n_in = rms_off(clean, noisy)
        nsig = int((np.abs(clean) > 0).sum())
        nk, ntot = out.sparse.n_kept, out.sparse.n_total
        print(f"{label:>12} {e:>3} {nw:>6} {nsig:>6} {nk:>8} {ntot:>9} "
              f"{nk / nw:>9.1f} {n_out:>9.3f} {n_in / max(n_out, 1e-9):>6.2f}x "
              f"{f0:>7.4f} {out.sparse.compression:>8.1f}x")
print("\nExpect: noise_out << noise_in (coherent removed + wavelet-denoised); "
      "Y (collection) F0 high; compression >> 1x.")
