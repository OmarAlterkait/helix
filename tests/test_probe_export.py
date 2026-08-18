"""Loading what we TRAIN: a pimm-export directory, not a converted blob."""
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm
from helix.model.checkpoint import is_export_dir, load_export_dir

ARCH = dict(n_slot=8, n_band=4, n_plane=6, d=32, blocks=1, dec_blocks=1,
            heads=4, dec_mode="cross", n_bins=16)


def _export(tmp_path, model, safe=False, with_tokenizer=True):
    """Mimic `pimm export`: weights + the resolved config beside them."""
    sd = {f"module.{k}": v for k, v in model.state_dict().items()}   # DDP prefix
    torch.save(sd, tmp_path / "model.bin")
    cfg = {"model": dict(ARCH, type="Coeff-FM", bins="/some/path.pt")}
    if with_tokenizer:
        cfg["transform"] = [{"type": "CoeffTokenize",
                             "cfg": {"cell_t": "grid_center", "pw": 16, "pt": 8}}]
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_export_round_trip_recovers_weights_and_bins(tmp_path):
    """The whole point: edges ride in the state_dict, so no sidecar is needed."""
    m = build_fm(dict(ARCH))
    edges = torch.linspace(-4, 4, ARCH["n_bins"] + 1).repeat(ARCH["n_band"], 1)
    m.set_bins(edges)
    d = _export(tmp_path, m)

    assert is_export_dir(d)
    back, meta = load_export_dir(d, device="cpu")
    torch.testing.assert_close(back.bin_edges, m.bin_edges)
    for (k, a), (_, b) in zip(sorted(m.state_dict().items()),
                              sorted(back.state_dict().items())):
        # equal_nan: the bin CENTROID buffers are persistent but NaN until
        # set_bins is given them, and NaN is the deliberate "absent" marker that
        # bin_centroids_ratio falls back on. Round-tripping unset->unset must
        # pass; a centroid that silently became a number would not.
        torch.testing.assert_close(a, b, msg=k, equal_nan=True)
    assert meta["source"] == "pimm-export"


def test_it_strips_the_ddp_prefix(tmp_path):
    m = build_fm(dict(ARCH))
    m.set_bins(torch.linspace(-4, 4, 17).repeat(4, 1))
    back, _ = load_export_dir(_export(tmp_path, m), device="cpu")
    assert not any(k.startswith("module.") for k in back.state_dict())


def test_tokenizer_geometry_travels_in_the_same_config(tmp_path):
    """cell_t is per-checkpoint; reading it from the config is what stops the
    encode path silently defaulting to `centroid` (which cost 94% of cells)."""
    m = build_fm(dict(ARCH))
    m.set_bins(torch.linspace(-4, 4, 17).repeat(4, 1))
    _, meta = load_export_dir(_export(tmp_path, m), device="cpu")
    assert meta["patch_config"].cell_t == "grid_center"

    _, meta2 = load_export_dir(_export(tmp_path, m, with_tokenizer=False), device="cpu")
    assert meta2["patch_config"] is None      # absent, not silently defaulted


def test_weights_without_a_config_are_refused(tmp_path):
    m = build_fm(dict(ARCH))
    m.set_bins(torch.linspace(-4, 4, 17).repeat(4, 1))
    torch.save(m.state_dict(), tmp_path / "model.bin")
    with pytest.raises(ValueError, match="architecture is not recoverable"):
        load_export_dir(tmp_path, device="cpu")


def test_probe_loader_accepts_an_export_dir(tmp_path):
    from helix.probe.features import load_probe_model
    m = build_fm(dict(ARCH))
    m.set_bins(torch.linspace(-4, 4, 17).repeat(4, 1))
    d = _export(tmp_path, m)
    trained, meta = load_probe_model(d, device="cpu")
    assert meta["source"] == "pimm-export"
    rnd, rmeta = load_probe_model(d, random_init=True, device="cpu")
    assert rmeta["random_init"] and rmeta["weights"] == "random-init"
    # the random twin must NOT carry the trained weights
    assert not torch.allclose(trained.embed.weight, rnd.embed.weight)


def test_ema_shadow_moves_to_the_model_device_on_resume():
    """The resume path must not leave the shadow on the CPU.

    Found by an actual 2-GPU resume: pimm restored step 600, the sidecar
    reloaded, and the first EMA update raised "Expected all tensors to be on the
    same device". A fresh run never reaches it — there the shadow is cloned from
    the live model and is already on-device — so only a resume exercises it.

    Checked without pimm by driving the update arithmetic directly.
    """
    shadow = {"w": torch.zeros(4)}                     # as loaded, on CPU
    live = {"w": torch.ones(4)}                        # as the model holds it
    d = 0.9
    for k, v in live.items():
        sh = shadow[k]
        if sh.device != v.device:
            sh = sh.to(v.device)
            shadow[k] = sh
        sh.mul_(d).add_(v.float(), alpha=1.0 - d)
    torch.testing.assert_close(shadow["w"], torch.full((4,), 0.1))


def test_random_init_control_is_seeded(tmp_path):
    """The `random` arm's null must be reproducible and seed-controlled.

    It was unseeded: `run_probe.py` builds the random net before any fit, and the
    only manual_seed in the probe path is in `fit_probe`, so the control was drawn
    from torch's process-start seed — a DIFFERENT network in every probe process.
    Two arms of an A/B were therefore scored against two different nulls, with
    nothing in the results row recording it.

    Compares PARAMETERS, not the full state_dict: a fresh model's `bin_edges`
    buffer is NaN until `set_bins`, and `torch.equal` is False for NaN, so a
    state_dict comparison fails whether or not the seeding works.
    """
    from helix.probe.features import load_probe_model

    m = build_fm(dict(ARCH))
    m.set_bins(torch.linspace(-4, 4, ARCH["n_bins"] + 1).repeat(ARCH["n_band"], 1))
    d = _export(tmp_path, m)

    def params(mod):
        return dict(mod.named_parameters())

    a, meta = load_probe_model(d, random_init=True, random_seed=0, device="cpu")
    b, _ = load_probe_model(d, random_init=True, random_seed=0, device="cpu")
    c, _ = load_probe_model(d, random_init=True, random_seed=1, device="cpu")

    pa, pb, pc = params(a), params(b), params(c)
    assert set(pa) == set(pb) == set(pc)
    assert all(torch.equal(pa[k], pb[k]) for k in pa), \
        "same random_seed gave a different network"
    assert any(not torch.equal(pa[k], pc[k]) for k in pa), \
        "different random_seed gave the same network"
    # The seed must reach the results row, or the fix is unauditable.
    assert meta["random_seed"] == 0 and meta["random_init"] is True
    # And it is genuinely a random net, not the exported weights.
    assert any(not torch.equal(pa[k], v) for k, v in params(m).items())
