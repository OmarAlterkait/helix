"""``varlen=True``: the serial model's groups without padding.

With token counts that need no padding, the padded and the unpadded formulations
run the same groups over the same rows, so they must agree -- that pins the
varlen layouts, group boundaries, permutes and decoder grouping to the existing,
reference-tested path. Where padding WOULD occur they differ by design (no
duplicated last token), which is checked only for being finite and different.

GPU tests (skipped on CPU): flash varlen vs the CPU per-group reference, and the
fused QK-norm + RoPE kernel vs the unfused path.
"""

import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm                                   # noqa: E402
from test_serial import SMALL, make_batch                           # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    m = build_fm({**SMALL, **kw})
    m.set_bins(torch.linspace(-6, 6, SMALL["n_bins"] + 1).repeat(SMALL["n_band"], 1))
    return m.eval()


def _pair(**kw):
    a, b = _model(**kw), _model(varlen=True, **kw)
    b.load_state_dict(a.state_dict())
    return a, b


# 288 cells, every third visible: 96 visible (6 x gp=16, 3 x gd=32) and 192
# masked; the decoder makes 6 groups of 32 queries x 16 keys. No padding anywhere.
EXACT = dict(n_cells=288)


def _exact_mask(n):
    return torch.arange(n) % 3 != 0


def test_varlen_equals_padded_when_nothing_is_padded():
    a, b = _pair()
    B = make_batch(**EXACT)
    m = _exact_mask(EXACT["n_cells"])
    with torch.no_grad():
        fa, fb = a.forward_feat(B, m), b.forward_feat(B, m)
        ea, eb = a.encode(B), b.encode(B)
    torch.testing.assert_close(fb, fa, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(eb, ea, rtol=1e-5, atol=1e-5)


def test_varlen_gradients_equal_padded_when_nothing_is_padded():
    a, b = _pair()
    B = make_batch(**EXACT)
    m = _exact_mask(EXACT["n_cells"])
    for mod in (a, b):
        mod.train()
        mod.forward_feat(B, m).square().mean().backward()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is None:
            continue
        torch.testing.assert_close(pb.grad, pa.grad, rtol=1e-4, atol=1e-6, msg=n)


def test_varlen_differs_where_padding_would_occur():
    a, b = _pair()
    B = make_batch(n_cells=100)                     # 75 visible: 75 % 16, 75 % 32 != 0
    m = torch.arange(100) % 4 != 0
    with torch.no_grad():
        fa, fb = a.forward_feat(B, m), b.forward_feat(B, m)
    assert torch.isfinite(fb).all()
    assert not torch.allclose(fa, fb)


def test_varlen_refuses_attention_sinks():
    with pytest.raises(ValueError, match="sinks"):
        build_fm({**SMALL, "varlen": True, "n_sink": 2})


def test_fused_qk_needs_varlen_qk_norm_and_hd64():
    m = _model(varlen=True, qk_norm=True)            # SMALL has head dim 16
    m.fused_qk = True
    B = make_batch(**EXACT)
    with pytest.raises(ValueError, match="fused_qk"):
        m.forward_feat(B, _exact_mask(EXACT["n_cells"]))


def test_bf16_params_converts_parameters_not_buffers():
    m = _model(bf16_params=True)
    assert all(p.dtype == torch.bfloat16 for p in m.parameters())
    assert m.bin_edges.dtype == torch.float32


@cuda
def test_flash_varlen_matches_the_per_group_reference():
    from helix.model.serial import _vl_attn
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(300, 4, 64, device="cuda", generator=g, dtype=torch.bfloat16) for _ in range(3))
    cu = torch.tensor([0, 128, 256, 300], device="cuda", dtype=torch.int32)
    out = _vl_attn(q, k, v, cu, cu, 128, 128)
    ref = _vl_attn(q.float(), k.float(), v.float(), cu, cu, 128, 128)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@cuda
def test_fused_qk_matches_the_unfused_path_on_gpu():
    kw = dict(d=256, heads=4, qk_norm=True, varlen=True)     # head dim 64
    a = _model(**kw).cuda().train()
    b = _model(**kw).cuda().train()
    b.load_state_dict(a.state_dict()); b.fused_qk = True
    B = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in make_batch(n_cells=400).items()}
    m = (torch.arange(400) % 4 != 0).cuda()
    out = []
    for mod in (a, b):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = mod.forward_feat(B, m)
        f.float().square().mean().backward()
        out.append((f.float(), {n: p.grad.float() for n, p in mod.named_parameters() if p.grad is not None}))
    (fa, ga), (fb, gb) = out
    torch.testing.assert_close(fb, fa, rtol=2e-2, atol=2e-2)
    top = max(float(g.norm()) for g in ga.values())
    for n in ga:
        # At init this toy loss leaves some gradients ~1e-3 of the rest (QK-norm
        # weights ~1e-9: their per-row terms cancel), where bf16 noise decides the
        # direction -- even unfused bf16 vs fp32 agree only to cos 0.007 there.
        # Those are covered exactly by test_normrope_kernel_matches_the_reference.
        if ga[n].norm() < 1e-2 * top:
            continue
        cos = torch.nn.functional.cosine_similarity(ga[n].flatten(), gb[n].flatten(), 0)
        assert cos > 0.999, (n, float(cos))


@cuda
@pytest.mark.parametrize("mode,parts", [("qkv", 3), ("kv", 2), ("q", 1)])
def test_normrope_kernel_matches_the_reference(mode, parts):
    """The kernel vs the unfused path on random inputs AND random upstream
    gradients, so the QK-norm weight gradient cannot cancel to nothing."""
    from types import SimpleNamespace
    from helix.model.kernels import normrope
    from helix.model.serial import _qk_rope
    from helix.model.layers import rope_tables
    g = torch.Generator(device="cuda").manual_seed(0)
    T, d = 500, 256
    X = torch.randn(T, parts * d, device="cuda", generator=g, dtype=torch.bfloat16).requires_grad_()
    ang = torch.rand(T, 16, device="cuda", generator=g) * 6.28
    cos, sin = rope_tables(ang, ang * 0.5, torch.bfloat16)
    qn = torch.nn.RMSNorm(64, eps=1e-6).cuda(); kn = torch.nn.RMSNorm(64, eps=1e-6).cuda()
    with torch.no_grad():
        qn.weight.uniform_(0.5, 1.5); kn.weight.uniform_(0.5, 1.5)
    blk = SimpleNamespace(qn=qn, kn=kn, h=d // 64, hd=64)
    res = []
    for fused in (False, True):
        for t in (X, qn.weight, kn.weight):
            t.grad = None
        out = _qk_rope(blk, X, cos, sin, mode, fused)
        if not res:
            ups = [torch.randn(o.shape, device="cuda", generator=g, dtype=o.dtype) for o in out]
        torch.autograd.backward([o.float() for o in out], [u.float() for u in ups])
        res.append([o.float() for o in out] + [X.grad.float()] +
                   [w.grad.float() for w in (qn.weight, kn.weight) if w.grad is not None])
    for a, b in zip(*res):
        assert float((a - b).norm() / a.norm()) < 1e-2
