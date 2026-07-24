"""End-to-end: process_plane(gate) → event_coeff_event → CoeffEvent → codec.

Ties the qualified gate (step 2) to the codec identity gate (step 1): a full
compute→write→read round-trip is bit-identical, and the read event reconstructs
to the same images as the direct sparse result.
"""
from __future__ import annotations

import numpy as np

from helix.core.backend import set_backend
from helix.core.wavelet import reconstruct
from helix.tpc.pipeline import process_plane, event_coeff_event
from helix.core.coeff_io import write_coeff_shard, read_coeff_event


def test_gate_compute_to_codec_roundtrip(tmp_path, config, synthetic_plane):
    set_backend("numpy")
    img = synthetic_plane["dig"]
    nt = img.shape[1]
    # two planes → two gids
    results = {0: process_plane(img, config, removal="gate"),
               4: process_plane(img, config, removal="gate")}
    for pp in results.values():
        assert pp.sparse.n_kept > 0

    ce = event_coeff_event(results, config, run="run_000", source_file="s_0000.h5", event=5)
    ce.basis.validate()                                  # band_lengths consistent with basis
    assert ce.n_coeff == sum(pp.sparse.n_kept for pp in results.values())

    path = tmp_path / "s_coeff_0000.h5"
    write_coeff_shard(path, [ce], dataset_name="s")
    ce2 = read_coeff_event(path, 0)

    for name in ("band", "plane_gid", "wire", "tau", "value", "gids", "n_wires", "sigma_threshold"):
        np.testing.assert_array_equal(getattr(ce2, name), getattr(ce, name),
                                      err_msg=f"{name} differs")
    assert ce2.basis.digest() == ce.basis.digest()
    assert ce2.run == "run_000" and ce2.event == 5

    # decode completeness: read event reconstructs to the same images as the
    # direct sparse coeffs of each plane
    recon = ce2.reconstruct_images(nt)
    for gid, pp in results.items():
        np.testing.assert_array_equal(recon[gid], reconstruct(pp.sparse, nt),
                                      err_msg=f"gid {gid} reconstruct differs")


def test_removal_modes_all_run(config, synthetic_plane):
    set_backend("numpy")
    img = synthetic_plane["dig"]
    for mode in ("gate", "multipass", "none"):
        pp = process_plane(img, config, synthetic_plane["sigma_w"], removal=mode)
        assert pp.reconstructed.shape == img.shape
        assert pp.sparse.n_kept > 0
