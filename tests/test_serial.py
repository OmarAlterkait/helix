"""The serial model's forward must be the per-block formulation, bit for bit.

``helix.model.serial`` carries the residual stream in grouped, padded order and
gathers once per block; the formulation it replaced gathered q, k and v into
groups and scattered the output back inside every block. That formulation is
kept HERE, as the reference, because "the same model, faster" is only a claim
while something can still compute the old answer:

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
from helix.model.fm import rope_angles                             # noqa: E402
from helix.model.layers import apply_rope                          # noqa: E402
from helix.model.loss import losses_cat                            # noqa: E402


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


def _mask(n, every=4):
    m = torch.zeros(n, dtype=torch.bool); m[::every] = True
    return m


# ------------------------------------------------ the reference formulation

def _rope_ref(x, ang_t, ang_w):
    """RoPE as the research tree wrote it: strided halves, per-call cos/sin."""
    h2 = x.shape[-1] // 2

    def rot(v, ang):
        c = torch.cos(ang)[:, None, :].repeat_interleave(2, -1)
        s = torch.sin(ang)[:, None, :].repeat_interleave(2, -1)
        v2 = torch.stack([-v[..., 1::2], v[..., 0::2]], -1).reshape_as(v)
        return v * c + v2 * s
    xt = rot(x[..., :h2], ang_t)
    xw = rot(x[..., h2:], ang_w) if ang_w is not None else x[..., h2:]
    return torch.cat([xt, xw], -1)


def _grp(x, order, npad, nb, g):
    T, h, hd = x.shape
    b = x.new_empty(npad, h, hd); b[:T] = x[order]
    if npad > T:
        b[T:] = x[order[-1]]
    return b.view(nb, g, h, hd).permute(0, 2, 1, 3)


def _uniform_attn(q, k, v, order, g):
    T, h, hd = q.shape; npad = ((T + g - 1) // g) * g; nb = npad // g
    o = F.scaled_dot_product_attention(*(_grp(t, order, npad, nb, g) for t in (q, k, v)))
    o = o.permute(0, 2, 1, 3).reshape(npad, h, hd)[:T]
    out = o.new_empty(T, h, hd); out[order] = o; return out


def _grouped_cross(q, k, v, oq, ok, g):
    Tq, h, hd = q.shape; Tk = k.shape[0]; nb = (max(Tq, Tk) + g - 1) // g
    gq, gk = (Tq + nb - 1) // nb, (Tk + nb - 1) // nb
    o = F.scaled_dot_product_attention(_grp(q, oq, nb * gq, nb, gq),
                                       _grp(k, ok, nb * gk, nb, gk),
                                       _grp(v, ok, nb * gk, nb, gk))
    o = o.permute(0, 2, 1, 3).reshape(nb * gq, h, hd)[:Tq]
    out = o.new_empty(Tq, h, hd); out[oq] = o; return out


def _mix(blk, x, ao, c):
    if blk.adaln:
        sa, ba, ga, sm, bm, gm = blk.ada(c).chunk(6, -1)
        x = x + ga * ao
        return x + gm * blk.mlp(blk.n2(x) * (1 + sm) + bm)
    x = x + ao
    return x + blk.mlp(blk.n2(x))


def _pre(blk, x, c, norm):
    if blk.adaln:
        sa, ba = blk.ada(c).chunk(6, -1)[:2]
        return norm(x) * (1 + sa) + ba
    return norm(x)


def _self_ref(blk, x, at, aw, order, g, c):
    T, d = x.shape
    q, k, v = blk.qkv(_pre(blk, x, c, blk.n1)).chunk(3, -1)
    q = _rope_ref(q.view(T, blk.h, blk.hd), at, aw)
    k = _rope_ref(k.view(T, blk.h, blk.hd), at, aw)
    o = _uniform_attn(q, k, v.view(T, blk.h, blk.hd), order, g)
    return _mix(blk, x, blk.proj(o.reshape(T, d)), c)


def _cross_ref(blk, q, kv, qat, qaw, kat, kaw, oq, okv, g, c):
    Tq, Tk = q.shape[0], kv.shape[0]
    qh = _rope_ref(blk.q(_pre(blk, q, c, blk.nq)).view(Tq, blk.h, blk.hd), qat, qaw)
    k, v = blk.kv(blk.nk(kv)).chunk(2, -1)
    kh = _rope_ref(k.view(Tk, blk.h, blk.hd), kat, kaw)
    o = _grouped_cross(qh, kh, v.view(Tk, blk.h, blk.hd), oq, okv, g)
    return _mix(blk, q, blk.proj(o.reshape(Tq, blk.h * blk.hd)), c)


def _sched(m, plane, t, wire):
    lay = m._layouts(plane, t, wire)
    return [lay[i % len(lay)] for i in range(len(m.enc))]


def _encode_ref(m, B, sel, layers=()):
    hd = m.d // m.heads
    at = rope_angles(B["t_phys"][sel], hd, *m.lam_t)
    aw = rope_angles(B["wire_pos"][sel], hd, *m.lam_w)
    x = m._emb(B, None if isinstance(sel, slice) else sel)
    c = m._cond(B)[sel] if m.cond == "adaln" else None
    out = {}
    for i, (blk, (o, g, uw)) in enumerate(
            zip(m.enc, _sched(m, B["plane_id"][sel], B["t_phys"][sel], B["wire_pos"][sel])), 1):
        x = _self_ref(blk, x, at, aw if uw else None, o, g, c)
        if i in layers:
            out[i] = x
    return x, out


def _forward_feat_ref(m, B, tok_mask):
    vis = ~tok_mask
    vis_idx, mask_idx = vis.nonzero(as_tuple=True)[0], tok_mask.nonzero(as_tuple=True)[0]
    hd = m.d // m.heads
    at = rope_angles(B["t_phys"], hd, *m.lam_t)
    aw = rope_angles(B["wire_pos"], hd, *m.lam_w)
    xv, _ = _encode_ref(m, B, vis_idx)
    c = m._cond(B) if m.cond == "adaln" else None
    qm = m.mask_tok.expand(mask_idx.numel(), m.d)
    if c is None:
        if m.film is not None:
            g_, b_ = m.film(B["band_id"][tok_mask], B["plane_id"][tok_mask], B["wirefeat"][tok_mask])
            qm = g_ * qm + b_
        qm = qm + m.band_emb(B["band_id"][tok_mask]) + m.plane_emb(B["plane_id"][tok_mask])
    qm = qm.to(xv.dtype)
    oq = torch.argsort(B["t_phys"][tok_mask].double())
    okv = torch.argsort(B["t_phys"][vis].double())
    cm = c[tok_mask] if c is not None else None
    for blk in m.dec:
        qm = _cross_ref(blk, qm, xv, at[tok_mask], aw[tok_mask], at[vis], aw[vis],
                        oq, okv, m.gd, cm)
    x = torch.zeros(B["inp"].shape[0], m.d, dtype=xv.dtype)
    return m.dec_norm(x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm))


# ------------------------------------------------------------------ RoPE

@pytest.mark.parametrize("wire", [True, False])
def test_rope_is_the_research_formula_exactly(wire):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(50, 4, 16, generator=g)
    at = rope_angles(torch.randn(50, generator=g) * 500, 16)
    aw = rope_angles(torch.rand(50, generator=g) * 1900, 16) if wire else None
    assert torch.equal(apply_rope(x, at, aw), _rope_ref(x, at, aw))


# ----------------------------------------------------------------- trunk

CASES = {
    "film": {},
    "rope_split": dict(rope_split=True),
    "adaln": dict(cond="adaln"),
    "6_blocks_wrap": dict(blocks=6),       # layouts cycle: block 4 follows block 3
    "2_blocks": dict(blocks=2),            # fewer blocks than layouts
}


@pytest.mark.parametrize("kw", CASES.values(), ids=CASES.keys())
def test_forward_feat_is_bit_exact(kw):
    B, m = make_batch(), _model(**kw)
    mask = _mask(B["n_cells"])
    with torch.no_grad():
        assert torch.equal(m.forward_feat(B, mask), _forward_feat_ref(m, B, mask))


@pytest.mark.parametrize("kw", CASES.values(), ids=CASES.keys())
def test_encode_layers_is_bit_exact(kw):
    """The probe reads these: every layer, every token, no masking."""
    B, m = make_batch(), _model(**kw)
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
    occ, val, _ = m.raw_heads(B, mask)
    bce, v = losses_cat(occ, val, B, mask, m.bin_edges, vis_w=0.0)
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
    assert {"compile_blocks", "gp", "gd", "rope_split", "serial", "d", "mask_ratio"} <= k
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


def test_compile_blocks_on_the_full_attention_model_raises():
    with pytest.raises(ValueError, match="compile_blocks"):
        build_fm({**SMALL, "dec_mode": "self"}, serial=False, compile_blocks=True)


def test_an_eval_rebuild_does_not_compile():
    """compile_blocks is how a run executed; a probe rebuilding it must not
    inherit minutes of compilation."""
    from helix.model.artifact import Artifact, build
    art = Artifact(arch={**SMALL, "compile_blocks": True}, op=None)
    assert build(art, device="cpu").compile_blocks is False
