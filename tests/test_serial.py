"""The model's forward must be the per-block formulation, bit for bit.

``FMModel`` carries the residual stream in layout order and permutes once per
block; the plain formulation gathers q, k and v into groups and scatters the
output back inside every block. That formulation is kept HERE, as the
reference, because "the same model, faster" is only a claim while something can
still compute the plain answer:

* the trunk reorders memory, not arithmetic, so ``torch.equal`` is the criterion;
* the masked-only categorical objective sums the same pairs in a different
  order, so it is held to a stated relative tolerance.

CPU, fp32: SDPA's backward is nondeterministic on GPU, so this is where equality
can be asserted at all.
"""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from helix.model import build_fm                                   # noqa: E402
from helix.model.layers import apply_rope_tables, rope_angles, rope_tables  # noqa: E402
from helix.model.loss import losses_cat                            # noqa: E402


SMALL = dict(n_slot=8, n_band=4, n_plane=6, d=64, blocks=4, dec_blocks=2,
             heads=4, n_bins=16, gp=16, gd=32)


def make_batch(n_cells=96, n_slot=8, n_band=4, n_plane=6, seed=0):
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


def _model(**kw):
    m = build_fm({**SMALL, **kw})
    edges = torch.linspace(-6, 6, SMALL["n_bins"] + 1).repeat(SMALL["n_band"], 1)
    m.set_bins(edges)
    return m.eval()


def _mask(n, every=4):
    m = torch.zeros(n, dtype=torch.bool); m[::every] = True
    return m


# ------------------------------------------------ the reference formulation

def _rope_ref(x, ang_t, ang_w):
    """RoPE written independently: strided halves, per-call cos/sin."""
    h2 = x.shape[-1] // 2

    def rot(v, ang):
        c = torch.cos(ang)[:, None, :].repeat_interleave(2, -1)
        s = torch.sin(ang)[:, None, :].repeat_interleave(2, -1)
        v2 = torch.stack([-v[..., 1::2], v[..., 0::2]], -1).reshape_as(v)
        return v * c + v2 * s
    xt = rot(x[..., :h2], ang_t)
    xw = rot(x[..., h2:], ang_w) if ang_w is not None else x[..., h2:]
    return torch.cat([xt, xw], -1)


def _attn_groups(q, k, v, qgroups, kgroups):
    """SDPA within each (query rows, key rows) group pair, in natural order."""
    out = torch.empty_like(q)
    for iq, ik in zip(qgroups, kgroups):
        out[iq] = F.scaled_dot_product_attention(
            q[iq].transpose(0, 1), k[ik].transpose(0, 1), v[ik].transpose(0, 1)).transpose(0, 1)
    return out


def _chunks(order, g):
    """Consecutive g-token groups along ``order``, the last one shorter."""
    return [order[i:i + g] for i in range(0, order.numel(), g)]


def _even(order, nb):
    T = order.numel()
    return [order[(i * T) // nb:((i + 1) * T) // nb] for i in range(nb)]


def _qkn(blk, q, k):
    """QK-norm where the reference applies it: per head, before RoPE. The norm
    modules themselves are the block's -- what is under test is the placement."""
    if blk.qn is None:
        return q, k
    return blk.qn(q), blk.kn(k)


def _self_ref(blk, x, at, aw, order, g):
    T, d = x.shape
    q, k, v = blk.qkv(blk.n1(x)).chunk(3, -1)
    q, k = _qkn(blk, q.view(T, blk.h, blk.hd), k.view(T, blk.h, blk.hd))
    q, k = _rope_ref(q, at, aw), _rope_ref(k, at, aw)
    grp = _chunks(order, g)
    o = _attn_groups(q, k, v.view(T, blk.h, blk.hd), grp, grp)
    x = x + blk.proj(o.reshape(T, d))
    return x + blk.mlp(blk.n2(x))


def _cross_ref(blk, q, kv, qat, qaw, kat, kaw, oq, okv, g):
    Tq, Tk = q.shape[0], kv.shape[0]
    k, v = blk.kv(blk.nk(kv)).chunk(2, -1)
    qh, kh = _qkn(blk, blk.q(blk.nq(q)).view(Tq, blk.h, blk.hd), k.view(Tk, blk.h, blk.hd))
    qh, kh = _rope_ref(qh, qat, qaw), _rope_ref(kh, kat, kaw)
    nb = max(1, min((max(Tq, Tk) + g - 1) // g, Tq, Tk))
    o = _attn_groups(qh, kh, v.view(Tk, blk.h, blk.hd), _even(oq, nb), _even(okv, nb))
    q = q + blk.proj(o.reshape(Tq, blk.h * blk.hd))
    return q + blk.mlp(blk.n2(q))


def _sched(m, plane, t, wire):
    lay = m._layouts(plane, t, wire)
    return [lay[i % len(lay)] for i in range(len(m.enc))]


def _angles_ref(m, B, sel):
    hd = m.d // m.heads
    at = rope_angles(B["t_phys"][sel], hd, *m.lam_t)
    aw = rope_angles(B["wire_pos"][sel], hd, *m.lam_w) if m.wire_rope else None
    return at, aw


def _encode_ref(m, B, sel, layers=()):
    at, aw = _angles_ref(m, B, sel)
    x = m._emb(B, None if isinstance(sel, slice) else sel)
    out = {}
    for i, (blk, (o, g, uw)) in enumerate(
            zip(m.enc, _sched(m, B["plane_id"][sel], B["t_phys"][sel], B["wire_pos"][sel])), 1):
        x = _self_ref(blk, x, at, aw if uw else None, o, g)
        if i in layers:
            out[i] = x
    return x, out


def _forward_feat_ref(m, B, tok_mask):
    vis = ~tok_mask
    vis_idx, mask_idx = vis.nonzero(as_tuple=True)[0], tok_mask.nonzero(as_tuple=True)[0]
    at, aw = _angles_ref(m, B, slice(None))
    sel = lambda a, i: None if a is None else a[i]
    xv, _ = _encode_ref(m, B, vis_idx)
    g_, b_ = m.film(B["band_id"][tok_mask], B["plane_id"][tok_mask], B["wirefeat"][tok_mask])
    qm = g_ * m.mask_tok.expand(mask_idx.numel(), m.d) + b_
    qm = qm + m.band_emb(B["band_id"][tok_mask]) + m.plane_emb(B["plane_id"][tok_mask])
    oq = torch.argsort(B["t_phys"][tok_mask].double())
    okv = torch.argsort(B["t_phys"][vis].double())
    for blk in m.dec:
        qm = _cross_ref(blk, qm, xv, at[tok_mask], sel(aw, tok_mask), at[vis], sel(aw, vis),
                        oq, okv, m.gd)
    x = torch.zeros(B["inp"].shape[0], m.d, dtype=xv.dtype)
    return m.dec_norm(x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm))


# ------------------------------------------------------------------ RoPE

@pytest.mark.parametrize("wire", [True, False])
def test_rope_tables_are_the_strided_formula_exactly(wire):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(50, 4, 16, generator=g)
    at = rope_angles(torch.randn(50, generator=g) * 500, 16)
    aw = rope_angles(torch.rand(50, generator=g) * 1900, 16) if wire else None
    assert torch.equal(apply_rope_tables(x, *rope_tables(at, aw)), _rope_ref(x, at, aw))


# ----------------------------------------------------------------- trunk

CASES = {
    "default": {},
    "rope_split": dict(rope_split=True),
    "no_wire_rope": dict(wire_rope=False),
    "6_blocks_wrap": dict(blocks=6),       # layouts cycle: block 4 follows block 3
    "2_blocks": dict(blocks=2),            # fewer blocks than layouts
    "qk_norm": dict(qk_norm=True),
}


@pytest.mark.parametrize("kw", CASES.values(), ids=CASES.keys())
@pytest.mark.parametrize("n_cells", [96, 100])          # 100: groups of unequal size
def test_forward_feat_is_bit_exact(kw, n_cells):
    B, m = make_batch(n_cells=n_cells), _model(**kw)
    mask = _mask(B["n_cells"])
    with torch.no_grad():
        assert torch.equal(m.forward_feat(B, mask), _forward_feat_ref(m, B, mask))


@pytest.mark.parametrize("kw", CASES.values(), ids=CASES.keys())
@pytest.mark.parametrize("n_cells", [96, 100])
def test_encode_layers_is_bit_exact(kw, n_cells):
    """The probe reads these: every layer, every token, no masking."""
    B, m = make_batch(n_cells=n_cells), _model(**kw)
    layers = set(range(1, len(m.enc) + 1))
    with torch.no_grad():
        got = m.encode_layers(B, layers)
        x, ref = _encode_ref(m, B, slice(None), layers)
        assert set(got) == set(ref) == layers
        for k in layers:
            assert torch.equal(got[k], ref[k]), f"layer {k}"
        assert torch.equal(m.encode(B), x)


def test_masked_only_is_the_masked_rows():
    B, m = make_batch(), _model()
    mask = _mask(B["n_cells"])
    with torch.no_grad():
        full = m.forward_feat(B, mask)
        rows_feat, rows = m.forward_feat(B, mask, masked_only=True)
    assert torch.equal(rows, mask.nonzero(as_tuple=True)[0])
    assert torch.equal(rows_feat, full[rows])


# ------------------------------------------------------------ objective

def _dense(m, B, mask):
    occ, val = m.raw_heads(B, mask)
    bce, v = losses_cat(occ, val, B, mask, m.bin_edges)
    return bce + v, bce, v


def test_loss_matches_the_dense_objective_within_summation_order():
    """Same pairs, same sum, different accumulation order: a tolerance, and a
    tight one, because nothing else may differ."""
    B, m = make_batch(), _model()
    mask = _mask(B["n_cells"])
    with torch.no_grad():
        got = m(B, tok_mask=mask)
        ref = dict(zip(("loss", "bce", "val"), _dense(m, B, mask)))
    for k in ref:
        r, g = float(ref[k]), float(got[k])
        assert abs(g - r) <= 1e-5 * max(abs(r), 1.0), f"{k}: {r} vs {g}"


def test_gradients_match_the_dense_objective():
    """The gradient is what training consumes, so it is the thing to check."""
    B, m = make_batch(), _model()
    mask = _mask(B["n_cells"])

    def grads(loss_fn):
        m.zero_grad(set_to_none=True)
        loss_fn().backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters()
                if p.grad is not None}

    a = grads(lambda: _dense(m, B, mask)[0])
    b = grads(lambda: m(B, tok_mask=mask)["loss"])
    assert set(a) == set(b)
    worst = max(float((a[k] - b[k]).abs().max() / a[k].abs().max().clamp(min=1e-12))
                for k in a)
    assert worst < 1e-4, f"max relative gradient difference {worst}"


# --------------------------------------------------------------- options

def test_fm_keys_names_every_option_build_fm_consumes():
    """The training entry point validates config keys against this set."""
    from helix.model.fm import fm_keys
    k = fm_keys()
    assert {"compile_blocks", "gp", "gd", "rope_split", "wire_rope", "d", "mask_ratio"} <= k
    assert not ({"self", "args", "kw", "fast_path"} & k)


def test_training_entry_point_refuses_an_unknown_model_key():
    """A key build_fm would ignore must fail the training build, not train the default."""
    pytest.importorskip("pimm")
    from helix.integrations.pimm.model import build_coeff_fm
    with pytest.raises(TypeError, match="fast_path"):
        build_coeff_fm(**SMALL, fast_path=True)


def test_compile_blocks_selects_the_compiled_blocks():
    """Creating the wrappers compiles nothing; this checks only the selection."""
    from helix.model import serial
    m = _model(compile_blocks=True)
    s, c = serial._blocks(m)
    assert s is not serial.self_block and c is not serial.cross_block
    m.compile_blocks = False
    assert serial._blocks(m) == (serial.self_block, serial.cross_block)
    # d768+ blocks exceed a DDP bucket; dynamo's DDP graph split then fails to
    # compile under dynamic shapes, so it must be off whenever we compile.
    import torch._dynamo
    assert torch._dynamo.config.optimize_ddp is False


def test_an_eval_rebuild_does_not_compile():
    """compile_blocks is how a run executed; a probe rebuilding it must not
    inherit minutes of compilation."""
    from helix.model.checkpoint import build_model
    assert build_model({"model": {**SMALL, "type": "Coeff-FM", "compile_blocks": True}}).compile_blocks is False


# ------------------------------------------------------------ execution

def test_activation_checkpointing_gives_the_same_gradients():
    B, m = make_batch(), _model()
    mask = _mask(B["n_cells"])

    def grads():
        m.zero_grad(set_to_none=True)
        m.train()(B, tok_mask=mask)["loss"].backward()
        return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    a = grads()
    m.act_ckpt = True
    b = grads()
    m.act_ckpt = False
    assert set(a) == set(b)
    assert all(torch.allclose(a[k], b[k], atol=1e-6, rtol=1e-5) for k in a)


def test_qk_norm_is_off_by_default_and_bounds_the_logits():
    """Off: no norm parameters. On: each head's query and key have unit RMS
    before RoPE, whatever the projection bias -- which is the point, since the
    late qkv biases grew 7 -> 18 in 47k steps."""
    base = _model()
    assert all(blk.qn is None for blk in list(base.enc) + list(base.dec))
    assert not any("qn." in n or "kn." in n for n, _ in base.named_parameters())
    m = _model(qk_norm=True)
    assert all(blk.qn is not None for blk in list(m.enc) + list(m.dec))
    blk = m.enc[0]
    with torch.no_grad():
        blk.qkv.bias.fill_(50.0)
        q = blk.qkv(torch.randn(7, m.d)).chunk(3, -1)[0].view(7, blk.h, blk.hd)
        rms = blk.qn(q).pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)
    assert "qk_norm" in __import__("helix.model.fm", fromlist=["fm_keys"]).fm_keys()


def test_permute_backward_matches_autograd():
    """The cheap backward (gather by the inverse) is autograd's scatter-add for
    a permutation: the same gradient."""
    from helix.model.serial import _permute
    g = torch.Generator().manual_seed(0)
    T = 37
    idx = torch.randperm(T, generator=g)
    x1 = torch.randn(T, 8, generator=g, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    w = torch.randn(T, 8, generator=g)
    (x1[idx] * w).sum().backward()
    (_permute(x2, idx) * w).sum().backward()
    assert torch.equal(x1[idx], _permute(x2.detach(), idx))
    assert torch.equal(x1.grad, x2.grad)
