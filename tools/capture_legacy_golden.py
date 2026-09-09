"""Freeze the OLD DSP chain's output so `test_legacy_parity` outlives research/.

`tests/test_legacy_parity.py` is the ONLY test that checks helix's coherent
gate + threshold against the ACTUAL historical production algorithm
(`measure_coeffs.smart_gate_bands` / `prod_threshold`) on real detector noise.
Everything else compares helix against helix: `test_coherent_gate.py`'s reference
is an inline transcription written in THIS repo from the intended algorithm
(`_ref_smart_gate_1pass`, "Inline numpy transcription of..."), so a
misunderstanding shared between the transcription and the implementation passes
both. `test_corpus_acceptance`'s F0 bound is a loose physics check that a real
numeric drift can slip under.

So the property worth keeping is: helix reproduces what the historical code
ACTUALLY DID, not what we believe it did.

A frozen golden is the right instrument here for the reason it is the WRONG one
for `test_training_parity`: the legacy DSP is finished. It will never change
again, so "verified once against research, checked forever without it" is the
correct epistemic status. Training semantics, by contrast, we changed this
session and will change again — a research-verified golden there could never be
legitimately refreshed, because research will never move to match.

Stores VALUES, not digests: the parity check is tolerant (support Jaccard,
median relative difference), and a hash cannot express a tolerance.

    python tools/capture_legacy_golden.py --write     # needs GPU + research/
    python tools/capture_legacy_golden.py --check     # needs neither
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tests"))

from _paths import RESEARCH_ROOT, sensor_shard          # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "tests", "goldens_legacy_dsp.npz")
SHARD = sensor_shard("run_0027575766", "sim_wire_sensor_0000.h5")
PLANES = ("volume_0_U", "volume_0_V", "volume_0_Y")
KGATE, KSIG, KAPPA, GS, NOISE_SEED = 4.0, 3.0, 1.0, 64, 12345


def _noisy_plane(label):
    """The exact input test_legacy_parity builds: real image + seeded noise."""
    import torch  # noqa: F401  (import order matters for the backend seam)
    from helix.tpc.config import DetectorConfig
    from helix.tpc.io import config_from_file, read_sensor_plane
    from helix.tpc.pipeline import canonical_plane_gid
    from pimm_data.geometry import load_plane_registry
    from pimm_data.noise import digitize, generate_noise

    cfg0 = config_from_file(SHARD)
    cfg = DetectorConfig(num_time_steps=cfg0.num_time_steps,
                         plane_labels=cfg0.plane_labels,
                         pedestals=cfg0.pedestals, threshold_kappa=KAPPA)
    reg = load_plane_registry("cubic_wireplane_geometry.json")
    ped = cfg.pedestals.get(label.split("_")[-1], 0)
    img = read_sensor_plane(SHARD, 0, label, cfg.num_time_steps, ped)
    gid, nw = canonical_plane_gid(label), img.shape[0]
    wl = np.asarray(reg.get(gid, {}).get("wire_lengths", []), np.float64)
    wl = wl if wl.size == nw else np.full(nw, 2.33)
    noisy = digitize(img + generate_noise(
        img.shape, rng=np.random.default_rng(NOISE_SEED), wire_lengths_m=wl,
        incoherent=True, coherent=True, series_spectrum=None, group_size=GS), ped)
    return cfg, noisy


def old_chain(label):
    """The historical chain, from the research tree, on GPU. Capture-time only."""
    import torch
    if RESEARCH_ROOT not in sys.path:
        sys.path.insert(0, RESEARCH_ROOT)
    import measure_coeffs as M

    from helix.core import backend
    _, noisy = _noisy_plane(label)
    backend.set_backend("torch")
    ops = backend.ops("helix.core.wavelet_ops")
    xt = torch.as_tensor(noisy, device="cuda")
    npad = (-noisy.shape[-1]) % (1 << M.LEVEL)
    if npad:
        xt = torch.nn.functional.pad(xt, (0, npad))
    old = M.prod_threshold(
        M.smart_gate_bands(ops._wavedec(xt, M.WAVELET, M.LEVEL),
                           kgate=KGATE, ksig=KSIG), KAPPA)
    return [np.asarray(b.detach().cpu(), np.float32) for b in old[:4]]


def sparse_of(bands):
    """(band, wire, tau, value) of every surviving coefficient, first 4 bands."""
    bi, wi, ti, vv = [], [], [], []
    for b, arr in enumerate(bands[:4]):
        w, t = np.nonzero(arr)
        bi.append(np.full(w.size, b, np.int16)); wi.append(w.astype(np.int32))
        ti.append(t.astype(np.int32)); vv.append(arr[w, t].astype(np.float32))
    return (np.concatenate(bi), np.concatenate(wi),
            np.concatenate(ti), np.concatenate(vv))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if not (a.write or a.check):
        raise SystemExit("pass --write or --check")

    if a.write:
        if not os.path.isdir(RESEARCH_ROOT):
            raise SystemExit(
                f"research tree absent ({RESEARCH_ROOT}) — this golden certifies "
                "agreement WITH it, so it cannot be written without it. There is "
                "no --no-verify: an unwitnessed legacy golden would freeze "
                "whatever helix happens to do, which is the opposite of the point.")
        out = {}
        for label in PLANES:
            b, w, t, v = sparse_of(old_chain(label))
            out[f"{label}/band"] = b; out[f"{label}/wire"] = w
            out[f"{label}/tau"] = t;  out[f"{label}/value"] = v
            print(f"  {label}: {v.size} coefficients")
        out["_meta"] = np.array(
            [f"shard={os.path.basename(SHARD)}", f"seed={NOISE_SEED}",
             f"kgate={KGATE}", f"ksig={KSIG}", f"kappa={KAPPA}", f"gs={GS}",
             f"research={RESEARCH_ROOT}"], dtype=object)
        np.savez_compressed(GOLDEN, **out)
        print(f"wrote {GOLDEN} ({os.path.getsize(GOLDEN)/1e6:.2f} MB)")
        return 0

    g = np.load(GOLDEN, allow_pickle=True)
    for label in PLANES:
        print(f"  {label}: {g[f'{label}/value'].size} coefficients")
    print("meta:", list(g["_meta"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
