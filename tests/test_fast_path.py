"""The fast path must be the same model, not a lookalike.

``fast_path=True`` selects two replacements at once: the permuted-residual
forward (``helix.model.fastpath``) and the sparse-active categorical head
(``helix.model.head``). They have different equivalence obligations and these
tests hold them to different standards:

* the trunk is **bit-exact** — it reorders memory, not arithmetic, so
  ``max|delta| == 0`` is the acceptance criterion, not a tolerance;
* the head computes the same sum over the same pairs in a different order, so it
  is held to a relative tolerance and the test says so.

Run on CPU in fp32 deliberately. Under bf16 autocast the trunk is still bit-exact
(measured on an A100, ``docs/PERFORMANCE.md`` §7b) but SDPA's backward is
nondeterministic, so a CPU test is the one that can assert equality at all.
"""

import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm                                   # noqa: E402
from helix.model.layers import apply_rope                          # noqa: E402
from helix.model.rope import apply_rope_fused, rope_tables         # noqa: E402
from helix.model.fm import rope_angles                             # noqa: E402


SMALL = dict(n_slot=8, n_band=4, n_plane=6, d=64, blocks=4, dec_blocks=2,
             heads=4, dec_mode="cross", n_bins=16, gp=16, gd=32)


def make_batch(n_cells=96, n_slot=8, n_band=4, n_plane=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    occ = (r(n_cells, n_slot) < 0.3).float()
    tgt = torch.randn(n_cells, n_slot, generator=g)
    cell, slot = occ.nonzero(as_tuple=True)
    return dict(
        band_id=torch.randint(0, n_band, (n_cells,), generator=g),
        plane_id=torch.randint(0, n_plane, (n_cells,), generator=g),
        t_phys=torch.randn(n_cells, generator=g) * 500,
        wire_pos=r(n_cells) * 1900,
        wirefeat=r(n_cells, 1),
        inp=torch.randn(n_cells, n_slot, generator=g),
        occ=occ,
        valid=(r(n_cells, n_slot) < 0.9),
        tgt=tgt,
        target=tgt[cell, slot],
        cell=cell, slot=slot, n_cells=n_cells,
    )


def _model(**kw):
    m = build_fm({**SMALL, **kw})
    edges = torch.linspace(-6, 6, SMALL["n_bins"] + 1).repeat(SMALL["n_band"], 1)
    m.set_bins(edges)
    return m.eval()


# ----------------------------------------------------------------- the RoPE

def test_fused_rope_is_bit_exact():
    """apply_rope_fused reorders the same products; fp32 equality, not closeness."""
    torch.manual_seed(0)
    x = torch.randn(64, 4, 16)
    at = rope_angles(torch.randn(64) * 500, 16, 8.0, 4336.0)
    aw = rope_angles(torch.rand(64) * 1900, 16, 32.0, 2048.0)
    cos, sin = rope_tables(at, aw)
    assert torch.equal(apply_rope_fused(x, cos, sin), apply_rope(x, at, aw))


def test_fused_rope_reproduces_a_disabled_axis():
    """ang_w=None must leave the second half UNROTATED, exactly as apply_rope
    does — that is a defect (docs/REVIEW_FIELD.md §1.3), and reproducing it is
    the point: this module changes cost, not arithmetic."""
    torch.manual_seed(0)
    x = torch.randn(32, 4, 16)
    at = rope_angles(torch.randn(32) * 500, 16, 8.0, 4336.0)
    cos, sin = rope_tables(at, None)
    got, ref = apply_rope_fused(x, cos, sin), apply_rope(x, at, None)
    assert torch.equal(got, ref)
    assert torch.equal(got[..., 8:], x[..., 8:])          # second half untouched


# ---------------------------------------------------------------- the trunk

@pytest.mark.parametrize("rope_split", [False, True])
def test_forward_feat_is_bit_exact(rope_split):
    """The permuted-residual trunk is the same computation, so equality holds."""
    B = make_batch()
    m = _model(rope_split=rope_split)
    mask = torch.zeros(B["n_cells"], dtype=torch.bool)
    mask[::4] = True                                      # deterministic 25% mask
    with torch.no_grad():
        m.fast_path = False
        ref = m.forward_feat(B, mask).clone()
        m.fast_path = True
        got = m.forward_feat(B, mask)
    assert torch.equal(got, ref), f"max|delta| = {(got - ref).abs().max()}"


def test_forward_feat_falls_back_for_return_ctx():
    """return_ctx is not implemented on the fast path; it must fall back rather
    than return something differently shaped."""
    B = make_batch()
    m = _model()
    m.fast_path = True
    mask = torch.zeros(B["n_cells"], dtype=torch.bool); mask[::4] = True
    with torch.no_grad():
        out = m.forward_feat(B, mask, return_ctx=True)
    assert isinstance(out, tuple) and len(out) == 2


# ------------------------------------------------------------------ the head

def test_forward_loss_matches_within_summation_order():
    """Same pairs, same sum, different accumulation order. A tolerance, not
    equality — and a tight one, because nothing else may differ."""
    B = make_batch()
    m = _model()
    mask = torch.zeros(B["n_cells"], dtype=torch.bool); mask[::4] = True
    with torch.no_grad():
        m.fast_path = False
        ref = m(B, tok_mask=mask)
        m.fast_path = True
        got = m(B, tok_mask=mask)
    for k in ("loss", "bce", "val"):
        r, g = float(ref[k]), float(got[k])
        assert abs(g - r) <= 1e-5 * max(abs(r), 1.0), f"{k}: {r} vs {g}"


def test_gradients_match():
    """The gradient is what training consumes, so it is the thing to check."""
    B = make_batch()
    m = _model()
    mask = torch.zeros(B["n_cells"], dtype=torch.bool); mask[::4] = True

    def grads(fast):
        m.zero_grad(set_to_none=True)
        m.fast_path = fast
        m(B, tok_mask=mask)["loss"].backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters()
                if p.grad is not None}

    a, b = grads(False), grads(True)
    assert set(a) == set(b)
    worst = max(float((a[k] - b[k]).abs().max() / a[k].abs().max().clamp(min=1e-12))
                for k in a)
    assert worst < 1e-4, f"max relative gradient difference {worst}"


def test_fast_train_path_declines_what_it_cannot_do():
    """Every gate is a capability, not a preference: anything the fast head does
    not implement is declined rather than trained as a different objective."""
    m = _model()
    m.fast_path = True
    assert m._fast_train_ok()
    m.vis_w = 0.5
    assert not m._fast_train_ok(), "vis_w > 0 needs the complementary pair set"
    m.vis_w = 0.0
    m.n_bins = 0
    assert not m._fast_train_ok(), "the sparse head is categorical-only"


def test_requested_but_ineligible_fast_path_raises():
    """A requested fast path must take effect or refuse. Falling back would run
    at the old speed and memory under a config sized for the fast one."""
    m = _model()
    m.fast_path = True
    m.vis_w = 0.5
    with pytest.raises(ValueError, match="vis_w"):
        m(make_batch())


def test_fm_keys_names_every_option_build_fm_consumes():
    """The training entry point validates config keys against this set, so it
    must include the new option and the serial-only parameters."""
    from helix.model.fm import fm_keys
    k = fm_keys()
    assert {"fast_path", "gp", "gd", "rope_split", "serial", "d", "mask_ratio"} <= k
    assert not ({"self", "args", "kw"} & k)


def test_training_entry_point_refuses_an_unknown_model_key():
    """A key build_fm would ignore must fail the training build, not train the default."""
    pytest.importorskip("pimm")
    from helix.integrations.pimm.model import build_coeff_fm
    with pytest.raises(TypeError, match="fast_pth"):
        build_coeff_fm(**SMALL, fast_pth=True)
