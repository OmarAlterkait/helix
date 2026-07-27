"""Legacy DSP parity — the packaged pipeline vs the OLD production reference.

Feeds the SAME real noisy plane through both chains in legacy configuration
(kgate=4, 1 pass, kappa=1, first 4 bands) and compares the kept support and
values:

  OLD (reference): torch/GPU — ``measure_coeffs``' ``_wavedec`` →
                   ``smart_gate_bands`` → ``prod_threshold``
  NEW (packaged):  numpy — ``wavedec`` → ``coherent_gate`` → ``threshold_bands``

Feeding one shared noisy image isolates the DSP: the numpy and torch noise
generators cannot produce the same RNG draw, so reproducing noise is not the
thing under test here.

Measured on run_0027575766 event 0 (U/V/Y): support match 99.99–100%, median
relative value diff ~5e-7 (float32 epsilon), with <1% of coefficients above
1e-5 — that tail is the deliberate ``sigc`` tie-break (``np.median`` averages the
two middles, ``torch.median`` takes the lower), not a structural difference.

Site-gated: needs a CUDA GPU, the production shard, and the research tree.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

SHARD = ("/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor/"
         "run_0027575766/sim_wire_sensor_0000.h5")
RESEARCH = "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model"
KGATE, KSIG, KAPPA, GS, NB_OLD = 4.0, 3.0, 1.0, 64, 4


def _deps():
    """(ok, reason) — GPU + shard + research reference + pimm-data all reachable."""
    if not os.path.exists(SHARD):
        return False, "production shard not reachable"
    if not os.path.isdir(RESEARCH):
        return False, "research reference tree not reachable"
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA GPU"
    except ImportError:
        return False, "torch not installed"
    try:
        import pimm_data  # noqa: F401
    except ImportError:
        return False, "pimm-data not importable"
    return True, ""


_OK, _WHY = _deps()
pytestmark = pytest.mark.skipif(not _OK, reason=_WHY)


def _rows(bands, nb):
    """-> {(band, wire, tau): value} over the first ``nb`` bands."""
    out = {}
    for b in range(nb):
        c = bands[b]
        c = c.detach().cpu().numpy() if hasattr(c, "detach") else np.asarray(c)
        w, t = np.nonzero(c)
        out.update(zip(zip([b] * w.size, w.tolist(), t.tolist()), c[w, t].tolist()))
    return out


def test_legacy_parity_vs_old_reference():
    import torch
    if RESEARCH not in sys.path:
        sys.path.insert(0, RESEARCH)
    import measure_coeffs as M

    from helix.core import backend
    from helix.core.wavelet import wavedec, threshold_bands
    from helix.tpc.io import config_from_file, read_sensor_plane
    from helix.tpc.config import DetectorConfig
    from helix.tpc.coherent_gate import coherent_gate
    from helix.tpc.pipeline import canonical_plane_gid, _pad_time
    from pimm_data.noise import generate_noise, digitize
    from pimm_data.geometry import load_plane_registry

    cfg0 = config_from_file(SHARD)
    cfg = DetectorConfig(num_time_steps=cfg0.num_time_steps,
                         plane_labels=cfg0.plane_labels, pedestals=cfg0.pedestals,
                         threshold_kappa=KAPPA)
    reg = load_plane_registry("cubic_wireplane_geometry.json")

    for label in ("volume_0_U", "volume_0_V", "volume_0_Y"):
        ped = cfg.pedestals.get(label.split("_")[-1], 0)
        img = read_sensor_plane(SHARD, 0, label, cfg.num_time_steps, ped)
        gid, nw = canonical_plane_gid(label), img.shape[0]
        wl = np.asarray(reg.get(gid, {}).get("wire_lengths", []), np.float64)
        wl = wl if wl.size == nw else np.full(nw, 2.33)
        noisy = digitize(img + generate_noise(
            img.shape, rng=np.random.default_rng(12345), wire_lengths_m=wl,
            incoherent=True, coherent=True, series_spectrum=None, group_size=GS), ped)

        backend.set_backend("torch")                       # OLD reference, on GPU
        ops = backend.ops("helix.core.wavelet_ops")
        xt = torch.as_tensor(noisy, device="cuda")
        npad = (-noisy.shape[-1]) % (1 << M.LEVEL)
        if npad:
            xt = torch.nn.functional.pad(xt, (0, npad))
        old = M.prod_threshold(
            M.smart_gate_bands(ops._wavedec(xt, M.WAVELET, M.LEVEL),
                               kgate=KGATE, ksig=KSIG), KAPPA)

        backend.set_backend("numpy")                       # NEW packaged chain
        nb, _ = wavedec(_pad_time(noisy, cfg.dwt_level), wavelet=cfg.wavelet,
                        level=cfg.dwt_level, mode=cfg.dwt_mode)
        new = threshold_bands(coherent_gate(nb, group_size=GS, kgate=KGATE, ksig=KSIG,
                                            npass=1, sigc_mode="median"),
                              cfg.threshold_spec())[0]

        o, n = _rows(old, NB_OLD), _rows(new, NB_OLD)
        inter = set(o) & set(n)
        support = len(inter) / max(len(set(o) | set(n)), 1)
        rel = np.array([abs(o[k] - n[k]) / max(abs(o[k]), 1e-9) for k in inter])

        assert support > 0.999, f"{label}: support match {support:.4%} (old {len(o)}, new {len(n)})"
        assert np.median(rel) < 1e-5, f"{label}: median rel diff {np.median(rel):.2e}"
        # the >1e-5 tail is the sigc tie-break only — keep it small
        assert (rel > 1e-5).mean() < 0.02, f"{label}: {(rel > 1e-5).mean():.2%} above float32 noise"
