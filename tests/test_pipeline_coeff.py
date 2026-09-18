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


def test_the_descriptor_records_the_mode_that_RAN_not_the_config_default():
    """A shard must not be able to claim a DSP it did not use.

    `process_plane` takes `removal=` and does `mode = (removal or
    config.removal)`, so an explicit override diverges from the config.
    `basis_from_config` only ever saw the config, so a run with removal='none'
    produced ungated coefficients under a descriptor saying 'gate' -- and
    therefore the same `basis_digest` as a real gated shard. `check_corpus_matches`
    compares exactly that digest, so the two were indistinguishable: a wrong
    number with no error, which is what the descriptor exists to prevent.
    """
    import numpy as np

    from helix.tpc.config import DetectorConfig
    from helix.tpc.pipeline import basis_from_config, process_plane

    cfg = DetectorConfig()
    assert cfg.removal == "gate", "this test assumes the gate default"
    img = np.random.default_rng(0).normal(scale=5.0,
                                          size=(128, cfg.num_time_steps)).astype(np.float32)

    gated = process_plane(img, cfg, removal="gate", with_images=False)
    plain = process_plane(img, cfg, removal="none", with_images=False)
    cat = lambda r: np.concatenate([np.asarray(c).ravel() for c in r.sparse.coeffs])
    assert not np.array_equal(cat(gated), cat(plain)), (
        "gate and none produced identical coefficients; this test cannot detect "
        "the divergence it exists for")

    bl = [np.asarray(c).shape[-1] for c in gated.sparse.coeffs]
    b_gate = basis_from_config(cfg, band_lengths=bl, level=gated.sparse.level,
                               removal="gate")
    b_none = basis_from_config(cfg, band_lengths=bl, level=plain.sparse.level,
                               removal="none")
    assert b_gate.removal["kind"] == "gate"
    assert b_none.removal["kind"] == "none", (
        "the descriptor stamped the CONFIG default, not the mode that ran")
    assert b_gate.digest() != b_none.digest(), (
        "two different DSPs produced the same basis_digest -- corpus identity "
        "cannot tell them apart")
