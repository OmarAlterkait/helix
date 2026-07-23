"""Confirm both real datasets load and report their schema/shape, so the plot
scripts can be written against verified facts (not memory)."""
import numpy as np

print("=" * 70)
print("DORAEMON wire data (pimm_data loader + geometry registry)")
print("=" * 70)
from pimm_data import JAXTPCDataset, load_plane_registry
from pimm_data.jaxtpc import canonical_plane_id

DATA_ROOT = "/sdf/home/o/omara/data/omara/doraemon"
SPLIT = "run_0026628546"
reg = load_plane_registry("cubic_wireplane_geometry.json")
print(f"registry planes: {sorted(reg)}")
for pid in sorted(reg):
    g = reg[pid]
    wl = np.asarray(g["wire_lengths"])
    print(f"  {pid:>16} label={g.get('label','?'):>12} n_wires={g['n_wires']:>5} "
          f"n_ticks={g['n_ticks']:>5} ped={g['pedestal']:>5} "
          f"wire_len[m] {wl.min():.2f}..{wl.max():.2f}")

ds = JAXTPCDataset(data_root=DATA_ROOT, split=SPLIT, modalities=("sensor",))
print(f"\n#events in split = {len(ds)}")
d0 = ds.get_data(0)["sensor"]["raw"]
print(f"event0 plane labels in raw: {sorted(d0)}")
for label in sorted(d0):
    cols = d0[label]
    v = np.asarray(cols["value"])
    print(f"  {label:>12}: hits={len(v):>7}  wire[{int(cols['wire'].min())}..{int(cols['wire'].max())}]"
          f"  time[{int(cols['time'].min())}..{int(cols['time'].max())}]"
          f"  |val| {np.abs(v).min():.1f}..{np.abs(v).max():.1f}  off-pixel std (clean?)={v.std():.3f}")

print("\n" + "=" * 70)
print("GOOP optical light_output.h5 (helix.optical.io)")
print("=" * 70)
from helix.optical import io as oio
LIGHT = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
cfg = oio.config_from_file(LIGHT)
print(f"config: n_channels={cfg.n_channels} n_pmts/side={cfg.n_pmts_per_side} "
      f"tick_ns={cfg.tick_ns} ped={cfg.pedestal} n_bits={cfg.n_bits} "
      f"baseline_noise_std={cfg.baseline_noise_std}")
evkeys = oio.list_events(LIGHT)
print(f"#events = {len(evkeys)}; first few = {evkeys[:3]}")
ec = oio.read_event_chunks(LIGHT, evkeys[0], cfg)
lens = ec.lengths
print(f"event0: n_chunks={len(ec.chunks)} sides={set(ec.side.tolist())} "
      f"chunk_len {lens.min()}..{lens.max()} (median {int(np.median(lens))})")
sig = oio.chunk_noise_sigma(ec.chunks)
peaks = np.array([np.abs(c).max() for c in ec.chunks])
nsig = int((peaks > 50).sum())
print(f"per-chunk noise sigma (db1 MAD): {sig.mean():.2f} ADC (median {np.median(sig):.2f})")
print(f"chunk peak |adc|: {peaks.min():.1f}..{peaks.max():.1f}; signal chunks(|x|max>50)={nsig}/{len(peaks)}")
print("\nVERIFY OK")
