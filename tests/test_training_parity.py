"""helix must reproduce the research TRAINING step, not just the forward pass.

`tests/test_model_fm.py` pins the forward — head outputs and `encode()` — against
the frozen golden. That is necessary and not sufficient: a model can have an
identical forward and still train differently, because training dynamics also
depend on

  * which tokens get masked            (make_mask)
  * what the loss computes             (losses / losses_fused / losses_cat)
  * how gradients flow from it         (backward)
  * how the optimizer consumes them    (muP param groups)

Any one of those drifting would leave the golden green while the model trained to
somewhere else entirely. These tests close that gap by running a full step —
forward, loss, backward, optimizer update — in helix and in the research
implementation from identical initial weights, and demanding bit-equality at
every stage.

Skips when the research tree is absent (it is scheduled for retirement; see
TODO.md 6). When it goes, these tests go with it — by then they will have done
their job, which is to license the deletion.
"""

import os
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

RESEARCH = "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm"
research_required = pytest.mark.skipif(
    not os.path.isdir(RESEARCH), reason=f"research tree absent: {RESEARCH}")

ARCH = dict(n_slot=16, n_band=4, n_plane=6, d=64, blocks=2, dec_blocks=1,
            heads=4, dec_mode="cross", mup=True, d_base=32)
N_CELLS = 48


def _batch(n_slot, n_band, n_plane, n_cells=N_CELLS, seed=7):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    occ = (r(n_cells, n_slot) < 0.4).float()
    tgt = torch.randn(n_cells, n_slot, generator=g)
    cell, slot = occ.nonzero(as_tuple=True)
    return dict(
        band_id=torch.randint(0, n_band, (n_cells,), generator=g),
        plane_id=torch.randint(0, n_plane, (n_cells,), generator=g),
        t_phys=torch.randn(n_cells, generator=g) * 500,
        wire_pos=r(n_cells) * 1900,
        wirefeat=r(n_cells, 1),
        inp=torch.randn(n_cells, n_slot, generator=g),
        occ=occ, valid=(r(n_cells, n_slot) < 0.9), tgt=tgt,
        target=tgt[cell, slot], cell=cell, slot=slot, n_cells=n_cells,
    )


def _pair(arch):
    """helix model and research model, identically initialised."""
    sys.path.insert(0, RESEARCH)
    from model_serial import SerialFMModel as Research
    from helix.model import build_fm

    torch.manual_seed(0)
    new = build_fm(dict(arch), serial=True)
    ref = Research(**arch)
    ref.load_state_dict(new.state_dict(), strict=True)   # identical init
    new.eval(); ref.eval()
    return new, ref


def _edges(n_band, K):
    return torch.linspace(-4, 4, K + 1).repeat(n_band, 1)


@pytest.fixture(autouse=True)
def _deterministic():
    """CPU float reductions depend on the intra-op thread count, so the same
    computation gives different last bits under different machine load. Pin it —
    the same trap the goldens hit twice."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@research_required
def test_mask_draw_is_identical():
    """Same generator seed must select the same tokens. A different mask means a
    different objective on every step, however identical the forward is."""
    from helix.model.mask import make_mask as helix_mask

    # Compile make_mask straight out of research/train.py rather than importing
    # it: that module drags the research tree's vendored pywt, which is broken in
    # this environment. The function itself only needs torch, so exec'ing its
    # source is both sufficient and more robust than the import.
    import ast
    src = pathlib.Path(RESEARCH, "train.py").read_text()
    tree = ast.parse(src)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "make_mask")
    ns = {"torch": torch}
    exec(compile(ast.get_source_segment(src, node), "research/train.py", "exec"), ns)
    research_mask = ns["make_mask"]

    B = _batch(ARCH["n_slot"], ARCH["n_band"], ARCH["n_plane"])
    for mode, ratio, n_planes in (("random", 0.5, 1), ("plane", 0.5, 2),
                                  ("block", 0.3, 1)):
        a = helix_mask(B, mode, ratio, n_planes, torch.Generator().manual_seed(3))
        b = research_mask(B, mode, ratio, n_planes, torch.Generator().manual_seed(3))
        assert torch.equal(a, b), f"mask differs for mode={mode}"


@research_required
@pytest.mark.parametrize("head", ["l2", "fused", "cat"])
def test_loss_is_identical(head):
    """The scalar the optimizer minimises must match exactly."""
    sys.path.insert(0, RESEARCH)
    import model as research_model
    from helix.model import loss as helix_loss

    arch = dict(ARCH)
    if head == "cat":
        arch["n_bins"] = 8
    new, ref = _pair(arch)
    B = _batch(arch["n_slot"], arch["n_band"], arch["n_plane"])
    m = torch.zeros(B["n_cells"], dtype=torch.bool); m[::2] = True

    with torch.no_grad():
        occ, val, lv = new.raw_heads(B, m)
        occ_r, val_r, lv_r = ref(B, m)
    assert torch.equal(occ, occ_r) and torch.equal(val, val_r)

    if head == "cat":
        e = _edges(arch["n_band"], arch["n_bins"])
        got = helix_loss.losses_cat(occ, val, B, m, e)
        want = research_model.losses_cat(occ_r, val_r, B, m, e)
    elif head == "fused":
        got = helix_loss.losses_fused(occ, val, lv, B, m)
        want = research_model.losses_fused(occ_r, val_r, lv_r, B, m)
    else:
        got = helix_loss.losses(occ, val, lv, B, m)
        want = research_model.losses(occ_r, val_r, lv_r, B, m)
    for g, w, name in zip(got, want, ("bce", "value")):
        assert torch.equal(g, w), f"{head}: {name} differs ({float(g)} vs {float(w)})"


@research_required
def test_muP_param_groups_are_identical():
    """param_groups drives the per-group LR/WD. A silent change here rescales
    learning across width and would only show up as a worse model."""
    new, ref = _pair(ARCH)
    a = new.param_groups(3e-4, weight_decay=0.05)
    b = ref.param_groups(3e-4, weight_decay=0.05)
    assert len(a) == len(b)
    for i, (ga, gb) in enumerate(zip(a, b)):
        assert ga.get("lr") == gb.get("lr"), f"group {i} lr"
        assert ga.get("weight_decay") == gb.get("weight_decay"), f"group {i} wd"
        assert len(ga["params"]) == len(gb["params"]), f"group {i} membership"
        assert sum(p.numel() for p in ga["params"]) == \
               sum(p.numel() for p in gb["params"]), f"group {i} size"


@research_required
@pytest.mark.parametrize("head", ["l2", "cat"])
def test_full_step_gradients_and_update_are_identical(head):
    """The whole dynamic: forward -> loss -> backward -> AdamW step.

    Compares gradients before the update and parameters after it. This is what
    licenses retiring mae_ddp: the new stack takes the same step the old one
    would have taken from the same state."""
    sys.path.insert(0, RESEARCH)
    import model as research_model
    from helix.model import loss as helix_loss

    arch = dict(ARCH)
    if head == "cat":
        arch["n_bins"] = 8
    new, ref = _pair(arch)
    new.train(); ref.train()
    B = _batch(arch["n_slot"], arch["n_band"], arch["n_plane"])
    m = torch.zeros(B["n_cells"], dtype=torch.bool); m[::2] = True
    e = _edges(arch["n_band"], arch.get("n_bins", 1))

    LR, WD = 3e-4, 0.05
    opt_n = torch.optim.AdamW(new.param_groups(LR, weight_decay=WD), lr=LR,
                              betas=(0.9, 0.95))
    opt_r = torch.optim.AdamW(ref.param_groups(LR, weight_decay=WD), lr=LR,
                              betas=(0.9, 0.95))

    def step(model, opt, losses_mod, is_research):
        occ, val, lv = model(B, m) if is_research else model.raw_heads(B, m)
        if head == "cat":
            bce, v = losses_mod.losses_cat(occ, val, B, m, e)
        else:
            bce, v = losses_mod.losses(occ, val, lv, B, m)
        loss = bce + v
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        opt.step()
        return float(loss), float(gn), grads

    l_n, gn_n, g_n = step(new, opt_n, helix_loss, False)
    l_r, gn_r, g_r = step(ref, opt_r, research_model, True)

    assert l_n == l_r, f"loss differs: {l_n} vs {l_r}"
    assert gn_n == gn_r, f"grad norm differs: {gn_n} vs {gn_r}"
    assert set(g_n) == set(g_r), "different parameter names carry gradients"
    for k in g_n:
        assert torch.equal(g_n[k], g_r[k]), f"gradient differs at {k}"

    pn = dict(new.named_parameters())
    pr = dict(ref.named_parameters())
    for k in pn:
        assert torch.equal(pn[k].detach(), pr[k].detach()), \
            f"parameter differs after the optimizer step at {k}"
