"""The FM's construction and its pimm Trainer contract.

The arithmetic of the forward is pinned in tests/test_serial.py; this file pins
what a caller relies on: ``model(batch)`` returns a loss dict, the head is
categorical and refuses to run without its bin edges, the config is strict, and
the masking policy mixes modes per step.
"""

import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm, FMModel                          # noqa: E402


SMALL = dict(n_slot=8, n_band=4, n_plane=6, d=64, blocks=2, dec_blocks=1, heads=4,
             n_bins=16)


def make_batch(n_cells=32, n_slot=8, n_band=4, n_plane=6, seed=0):
    """A batch with the key contract helix.model.tokenize.to_fm() emits."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    return dict(
        band_id=torch.randint(0, n_band, (n_cells,), generator=g),
        plane_id=torch.randint(0, n_plane, (n_cells,), generator=g),
        t_phys=torch.randn(n_cells, generator=g) * 500,
        wire_pos=r(n_cells) * 1900,
        wirefeat=r(n_cells, 1),
        inp=torch.randn(n_cells, n_slot, generator=g),
        occ=(r(n_cells, n_slot) < 0.3).float(),
        valid=(r(n_cells, n_slot) < 0.9),
        tgt=torch.randn(n_cells, n_slot, generator=g),
        n_cells=n_cells,
    )


def binned(model):
    """``model`` with bin edges registered, as every forward requires."""
    return model.set_bins(torch.linspace(-4, 4, model.n_bins + 1).repeat(SMALL["n_band"], 1))


# --------------------------------------------------------------------------
# the pimm Trainer contract
# --------------------------------------------------------------------------

def test_forward_returns_loss_dict():
    """pimm's Trainer calls ``model(batch)["loss"]``; the loss must carry grad."""
    model = binned(build_fm(SMALL))
    out = model(make_batch(n_slot=SMALL["n_slot"]))
    assert set(out) >= {"loss", "bce", "val", "masked_frac"}
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert out["loss"].requires_grad
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_forward_without_bins_is_refused():
    """A model whose edges were never registered must fail loudly, not silently
    score against a garbage buffer."""
    model = build_fm(SMALL)
    with pytest.raises(AssertionError, match="set_bins"):
        model(make_batch(n_slot=SMALL["n_slot"]))


def test_set_bins_rejects_wrong_k():
    model = build_fm(SMALL)
    with pytest.raises(AssertionError, match="n_bins"):
        model.set_bins(torch.linspace(-4, 4, 9).repeat(4, 1))    # K+1 = 9, not 17


def test_the_head_is_categorical_and_needs_bins():
    assert build_fm(SMALL).val_head.weight.shape[0] == SMALL["n_bins"] * SMALL["n_slot"]
    with pytest.raises(ValueError, match="n_bins"):
        build_fm(dict(SMALL, n_bins=0))
    with pytest.raises(TypeError, match="n_bins"):
        build_fm({k: v for k, v in SMALL.items() if k != "n_bins"})


def test_explicit_mask_is_honoured():
    model = binned(build_fm(SMALL))
    B = make_batch(n_slot=SMALL["n_slot"])
    m = torch.zeros(B["n_cells"], dtype=torch.bool)
    m[:4] = True
    assert model(B, tok_mask=m)["masked_frac"] == pytest.approx(4 / B["n_cells"])


def test_train_opts_are_not_constructor_params():
    """Training policy must stay off the arch signature: the constructor is the
    architecture a checkpoint records, so a training knob there would corrupt
    that record."""
    import inspect
    params = set(inspect.signature(FMModel.__init__).parameters)
    assert not params & {"mask_mode", "mask_ratio", "plane_frac", "compile_blocks"}
    m = build_fm(dict(SMALL, mask_ratio=0.6, mask_mode="plane"))
    assert (m.mask_ratio, m.mask_mode) == (0.6, "plane")


def test_build_fm_refuses_unknown_keys():
    """A key build_fm does not consume would otherwise do nothing, silently."""
    with pytest.raises(TypeError, match="some_trainer_key"):
        build_fm(dict(SMALL, some_trainer_key=1))


def test_missing_batch_keys_name_themselves():
    """A batch lacking what the objective reads must say which keys -- not raise
    a bare KeyError from inside the loss."""
    model = binned(build_fm(SMALL))
    B = make_batch(n_slot=SMALL["n_slot"])
    B.pop("tgt")
    with pytest.raises(KeyError, match="tgt"):
        model(B)


def test_plane_frac_mixes_modes_per_step():
    """`plane_frac` is a per-STEP mix, not a mode.

    A run that only ever masks randomly never has to reconstruct a plane it
    cannot see, so nothing forces cross-plane triangulation -- which is the
    whole reason the option exists.
    """
    B = make_batch(n_slot=SMALL["n_slot"])
    gid = B["plane_id"]

    def whole_plane(msk):
        return all(bool(msk[gid == g].all()) or bool((~msk[gid == g]).all())
                   for g in gid.unique())

    counts = {}
    for pf in (0.0, 0.5, 1.0):
        model = build_fm(dict(SMALL), plane_frac=pf)
        torch.manual_seed(0)
        counts[pf] = sum(whole_plane(model.make_mask(B)) for _ in range(200))

    assert counts[0.0] == 0, "plane_frac=0 must never mask a whole plane"
    assert counts[1.0] == 200, "plane_frac=1 must always mask whole planes"
    assert 60 < counts[0.5] < 140, f"plane_frac=0.5 should mix, got {counts[0.5]}/200"

    # An explicit mode= overrides the policy and skips the draw entirely.
    model = build_fm(dict(SMALL), plane_frac=1.0)
    assert not whole_plane(model.make_mask(B, mode="random"))
