"""The attention groups' execution paths, and partial decoding.

CPU: the fused-QK switch's checks and its CPU fallback, bf16 parameters,
``dec_frac``. GPU tests (skipped on CPU): flash varlen vs the CPU per-group
reference, and the fused QK-norm + RoPE kernel vs the unfused path.
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


# 288 cells, every third visible: 96 visible (6 x gp=16, 3 x gd=32) and 192
# masked; the decoder makes 6 groups of 32 queries x 16 keys.
EXACT = dict(n_cells=288)


def _exact_mask(n):
    return torch.arange(n) % 3 != 0


def test_fused_qk_needs_qk_norm_and_hd64():
    m = _model(qk_norm=True)                         # SMALL has head dim 16
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
    kw = dict(d=256, heads=4, qk_norm=True)                  # head dim 64
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


def test_dec_frac_decodes_a_subset_in_training_only():
    m = _model().train()
    m.dec_frac = 1 / 3
    B = make_batch(**EXACT)
    mask = _exact_mask(EXACT["n_cells"])                   # 192 masked
    feat, rows = m.forward_feat(B, mask, masked_only=True)
    assert rows.numel() == 64 and feat.shape[0] == 64
    assert bool(mask[rows].all())                          # only masked rows are decoded
    m.eval()
    with torch.no_grad():
        _, rows_eval = m.forward_feat(B, mask, masked_only=True)
    assert rows_eval.numel() == 192                        # evaluation decodes them all


def test_dec_frac_one_is_the_default_path():
    a, b = _model().train(), _model().train()
    b.load_state_dict(a.state_dict()); b.dec_frac = 1.0
    B = make_batch(**EXACT); mask = _exact_mask(EXACT["n_cells"])
    torch.manual_seed(0); fa, ra = a.forward_feat(B, mask, masked_only=True)
    torch.manual_seed(0); fb, rb = b.forward_feat(B, mask, masked_only=True)
    assert torch.equal(ra, rb) and torch.equal(fa, fb)


def test_fused_qk_model_runs_on_cpu_through_the_reference_path():
    a = _model(qk_norm=True, heads=1)          # hd 64: fused_qk is accepted
    b = _model(qk_norm=True, heads=1)
    b.load_state_dict(a.state_dict()); b.fused_qk = True
    B = make_batch(**EXACT); m = _exact_mask(EXACT["n_cells"])
    with torch.no_grad():
        torch.testing.assert_close(b.forward_feat(B, m), a.forward_feat(B, m))
