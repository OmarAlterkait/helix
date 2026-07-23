"""Prototype: materialize per-plane, per-band wavelet-coefficient tensors.

The input contract for the coefficient foundation model. Division of labour
(per the project plan): **pimm_data extracts the sensor data only** (sparse clean
truth + geometry); **everything else is helix** — densify, forward noise, the DWT,
and the per-level coefficient lattice.

Pipeline (all on GPU after extraction):
    pimm_data.JAXTPCDataset.get_data        sparse clean COO  (CPU, only sparse crosses PCIe)
    -> densify (this file)                   {gid: (B, W, T)} clean dense
    -> forward noise (pimm_data.noise)       + coherent + intrinsic, digitize  -> noisy
    -> pad T to a multiple of 2^level
    -> helix torch DWT (pre-threshold)       {gid: {band: (B, W, len_band)}}

Produces the (wire x level x time) dyadic lattice the model consumes, for BOTH
the noisy input and the clean target. Pre-threshold (raw wavedec): removal /
thresholding is the model's job, not the loader's.

Run (env paths are wired in __main__):
    python research/coeff_prototype.py --events 2 --save /tmp/coeff_proto.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
DATA_ROOT = "/sdf/data/neutrino/omara/JAXTPC_Wire/test_00_00_02"
SPLIT = "run_0027575766"
DATASET_NAME = "sim_wire"
GEOM_JSON = ("/sdf/group/neutrino/omara/particle-imaging-models/"
             "libs/pimm-data/src/pimm_data/data/cubic_wireplane_geometry.json")

WAVELET = "coif3"
LEVEL = 4
MODE = "periodization"


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def load_geometry(path=GEOM_JSON):
    """Canonical-plane geometry registry: {label: {n_wires, pedestal, wire_lengths_m, n_ticks}}."""
    import json
    d = json.load(open(path))
    nts = int(d["num_time_steps"])
    planes = {}
    for label, e in d["planes"].items():
        wl = e.get("wire_lengths_m")
        planes[label] = {
            "n_wires": int(e["n_wires"]),
            "pedestal": int(e["pedestal"]),
            "n_ticks": nts,
            "wire_lengths_m": (np.asarray(wl, np.float32) if wl is not None else None),
        }
    return planes, nts, d.get("coherent", {})


# --------------------------------------------------------------------------- #
# helix-side ops (densify / noise / DWT)
# --------------------------------------------------------------------------- #
def densify(raw, n_wires, n_ticks):
    """Sparse plane COO -> dense (n_wires, n_ticks) float32 (clean, pedestal-subtracted)."""
    img = np.zeros((n_wires, n_ticks), np.float32)
    w, t, v = raw["wire"], raw["time"], raw["value"]
    img[w, t] = v
    return img


def add_forward_noise(clean, geo, seed, *, coherent=True, incoherent=True):
    """clean + (coherent + intrinsic) noise, then digitize. JAXTPC forward model.

    The forward physics lives in pimm_data.noise (canonical numpy twin of JAXTPC);
    helix orchestrates. coherent is device-independent / bit-exact; intrinsic needs
    wire_lengths_m.
    """
    from pimm_data.noise import generate_noise, digitize
    rng = np.random.default_rng(seed)
    noise = generate_noise(
        clean.shape, rng=rng, wire_lengths_m=geo["wire_lengths_m"],
        incoherent=incoherent, coherent=coherent)
    return digitize(clean + noise, geo["pedestal"]).astype(np.float32)


def wavedec_torch(img, *, wavelet=WAVELET, level=LEVEL):
    """Pre-threshold per-band coeffs via helix's torch DWT. Pads T to a multiple
    of 2^level (torch periodization requirement). Returns list [cA, cD_L, ..., cD_1]."""
    import torch
    from helix.core import backend
    backend.set_backend("torch")
    ops = backend.ops("helix.core.wavelet_ops")
    T = img.shape[-1]
    pad = (-T) % (2 ** level)
    pad_spec = [(0, 0)] * (img.ndim - 1) + [(0, pad)]   # last axis (time) only
    x = torch.as_tensor(np.pad(img, pad_spec), dtype=torch.float32)
    if torch.cuda.is_available():
        x = x.cuda()
    return ops._wavedec(x, wavelet, level), pad


def band_names(level=LEVEL):
    return ["cA%d" % level] + ["cD%d" % (level - i) for i in range(level)]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=int, default=2, help="batch size (events)")
    ap.add_argument("--volume", type=int, default=0, help="volume to use (0 or 1)")
    ap.add_argument("--no-noise", action="store_true", help="clean coeffs only")
    ap.add_argument("--save", type=str, default=None, help="path to save the coeff dict (.pt)")
    args = ap.parse_args()

    import torch
    from pimm_data import JAXTPCDataset
    from pimm_data.jaxtpc import canonical_plane_id

    geom, nts, coh = load_geometry()
    print(f"geometry: {len(geom)} planes, n_ticks={nts}, coherent={coh}")

    ds = JAXTPCDataset(data_root=DATA_ROOT, split=SPLIT,
                       modalities=("sensor",), dataset_name=DATASET_NAME)
    print(f"dataset: {len(ds.data_list)} events; using {args.events} (volume {args.volume})")

    labels = [f"volume_{args.volume}_{t}" for t in ("U", "V", "Y")]

    # {gid: {'clean': [bands...], 'noisy': [bands...]}}; each band (B, W, len)
    out = {}
    for label in labels:
        gid = canonical_plane_id(label)
        geo = geom[label]
        W, T = geo["n_wires"], geo["n_ticks"]
        clean_imgs, noisy_imgs = [], []
        for ev in range(args.events):
            s = ds.get_data(ev)["sensor"]
            clean = densify(s["raw"][label], W, T)
            clean_imgs.append(clean)
            if not args.no_noise:
                noisy_imgs.append(add_forward_noise(clean, geo, seed=ev))
        clean_batch = np.stack(clean_imgs)              # (B, W, T)
        cA_clean, pad = wavedec_torch(clean_batch)
        entry = {"clean": cA_clean, "pad": pad, "n_ticks": T, "n_wires": W}
        if not args.no_noise:
            noisy_batch = np.stack(noisy_imgs)
            entry["noisy"], _ = wavedec_torch(noisy_batch)
        out[gid] = entry

    # report -----------------------------------------------------------------
    names = band_names()
    print(f"\nper-band coefficient tensors  (wavelet={WAVELET} L{LEVEL} {MODE})")
    print(f"{'plane':>14} {'band':>5} {'shape (B,W,len)':>22} {'dtype':>9} "
          f"{'MB':>7} {'clean>2ADC%':>11} {'noisy>2ADC%':>11}")
    for label in labels:
        gid = canonical_plane_id(label)
        e = out[gid]
        for nm, cc in zip(names, e["clean"]):
            occ_c = 100.0 * (cc.abs() > 2.0).float().mean().item()
            occ_n = float("nan")
            if "noisy" in e:
                ncc = e["noisy"][names.index(nm)]
                occ_n = 100.0 * (ncc.abs() > 2.0).float().mean().item()
            mb = cc.numel() * cc.element_size() / 1e6
            print(f"{label:>14} {nm:>5} {str(tuple(cc.shape)):>22} "
                  f"{str(cc.dtype).replace('torch.',''):>9} {mb:>7.2f} "
                  f"{occ_c:>11.3f} {occ_n:>11.3f}")

    # totals
    tot = sum(c.numel() for e in out.values() for c in e["clean"])
    print(f"\ntotal clean coeffs (one batch, {len(labels)} planes): {tot:,} "
          f"= {tot*4/1e6:.1f} MB float32")
    if args.save:
        torch.save({"coeffs": out, "labels": labels, "wavelet": WAVELET,
                    "level": LEVEL, "band_names": names}, args.save)
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    # env wiring: pimm_data (extraction), helix (algorithms), .pylibs (pywt+hdf5plugin)
    for p in ("/sdf/group/neutrino/omara/helix/.pylibs",
              "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src",
              "/sdf/group/neutrino/omara/helix"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import hdf5plugin  # noqa: F401  (register blosc/zstd HDF5 codecs before any read)
    main()
