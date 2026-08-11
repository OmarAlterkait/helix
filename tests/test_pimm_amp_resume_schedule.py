"""Three things pimm's Trainer does to a run that nothing here had exercised.

`test_pimm_step_contract.py` covers the batch reaching the model. This covers
what happens around the step:

  AMP        run_step wraps the forward in bf16 autocast. Every parity test so
             far is fp32, so the categorical loss in reduced precision was
             unverified — and a 128-way log-softmax is exactly the shape that
             loses precision quietly rather than producing NaN.
  resume     a restored optimizer must keep the muP per-group lr/weight_decay.
             Losing them on resume rescales learning across width for the rest
             of the run, and the loss curve would look plausible throughout.
  scheduler  muP is expressed as a per-group LR ratio, and pimm drives LRs with
             OneCycleLR. If it collapses the groups to one LR, muP is silently
             discarded — the hidden group drifts to base_lr instead of base_lr/m.

These use torch directly rather than pimm's wrappers where the behaviour under
test is torch's (autocast, AdamW state, OneCycleLR's per-group max_lr); pimm's
OneCycleLR subclass only reinterprets `pct_start`, which is noted where it
matters.
"""

import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm  # noqa: E402

ARCH = dict(n_slot=16, n_band=4, n_plane=6, d=64, blocks=2, dec_blocks=1,
            heads=4, dec_mode="cross", mup=True, d_base=32)
LR, WD = 3e-4, 0.05


def _batch(n_cells=48, n_slot=16, seed=7):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    occ = (r(n_cells, n_slot) < 0.4).float()
    tgt = torch.randn(n_cells, n_slot, generator=g)
    cell, slot = occ.nonzero(as_tuple=True)
    return dict(band_id=torch.randint(0, 4, (n_cells,), generator=g),
                plane_id=torch.randint(0, 6, (n_cells,), generator=g),
                t_phys=torch.randn(n_cells, generator=g) * 500,
                wire_pos=r(n_cells) * 1900, wirefeat=r(n_cells, 1),
                inp=torch.randn(n_cells, n_slot, generator=g), occ=occ,
                valid=(r(n_cells, n_slot) < 0.9), tgt=tgt,
                target=tgt[cell, slot], cell=cell, slot=slot, n_cells=n_cells)


def _model(**over):
    torch.manual_seed(0)
    cfg = dict(ARCH, **over)
    m = build_fm(cfg)
    if cfg.get("n_bins", 0):
        m.set_bins(torch.linspace(-4, 4, cfg["n_bins"] + 1).repeat(cfg["n_band"], 1))
    return m


@pytest.fixture(autouse=True)
def _one_thread():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


# ---- 1. AMP ---------------------------------------------------------------

@pytest.mark.parametrize("head", ["l2", "nll", "cat"])
def test_bf16_autocast_loss_is_finite_and_close_to_fp32(head):
    """pimm runs the forward under bf16 autocast. bfloat16 has ~3 decimal digits
    of mantissa, so the question is not whether the loss is NaN but whether it is
    still the same number to a useful tolerance."""
    over = {"nll": dict(nll=True), "cat": dict(n_bins=16)}.get(head, {})
    m = _model(**over)
    B = _batch(n_slot=ARCH["n_slot"])
    mask = torch.zeros(B["n_cells"], dtype=torch.bool); mask[::2] = True

    with torch.no_grad():
        fp32 = float(m(B, tok_mask=mask)["loss"])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            bf16 = float(m(B, tok_mask=mask)["loss"])
    assert torch.isfinite(torch.tensor(bf16)), f"{head}: bf16 loss is not finite"
    # Measured on this fixture: 0.002% (l2), 0.002% (nll), 0.000% (cat). The
    # 1% bound is loose enough not to be brittle and tight enough that a real
    # precision regression — e.g. a reduction moved into bf16 — would trip it.
    assert bf16 == pytest.approx(fp32, rel=0.01), (
        f"{head}: bf16 loss {bf16} diverges from fp32 {fp32} by more than 1%")


@pytest.mark.parametrize("head", ["l2", "cat"])
def test_bf16_backward_produces_finite_gradients(head):
    """A finite loss with NaN/Inf gradients would poison the optimizer state on
    the first step and never recover."""
    over = {"cat": dict(n_bins=16)}.get(head, {})
    m = _model(**over)
    B = _batch(n_slot=ARCH["n_slot"])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = m(B)["loss"]
    loss.backward()
    grads = [(n, p.grad) for n, p in m.named_parameters() if p.grad is not None]
    assert grads, "no gradients at all"
    bad = [n for n, g in grads if not torch.isfinite(g).all()]
    assert not bad, f"{head}: non-finite gradients in {bad[:5]}"
    assert any(g.abs().sum() > 0 for _, g in grads), "all gradients are exactly zero"


# ---- 2. resume ------------------------------------------------------------

def test_optimizer_resume_preserves_muP_groups(tmp_path):
    """Save mid-run, restore, and the per-group lr/weight_decay must come back.

    AdamW's state_dict stores param_groups, so this is really asserting that the
    muP structure round-trips rather than being rebuilt from a flat config on
    resume — which is how it would silently be lost."""
    m = _model(n_bins=16)
    opt = torch.optim.AdamW(m.param_groups(LR, weight_decay=WD), lr=LR,
                            betas=(0.9, 0.95))
    before = [(g["lr"], g["weight_decay"]) for g in opt.param_groups]
    m_width = ARCH["d"] // ARCH["d_base"]
    assert any(abs(lr / LR - 1.0 / m_width) < 1e-12 for lr, _ in before), \
        "muP is not active in this fixture"

    B = _batch(n_slot=ARCH["n_slot"])
    for _ in range(3):
        loss = m(B)["loss"]
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    path = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(), "optimizer": opt.state_dict()}, path)

    m2 = _model(n_bins=16)
    opt2 = torch.optim.AdamW(m2.param_groups(LR, weight_decay=WD), lr=LR,
                             betas=(0.9, 0.95))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m2.load_state_dict(blob["model"], strict=True)
    opt2.load_state_dict(blob["optimizer"])

    after = [(g["lr"], g["weight_decay"]) for g in opt2.param_groups]
    assert after == before, f"muP groups changed across resume: {before} -> {after}"
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p1.detach(), p2.detach())


def test_step_after_resume_matches_uninterrupted_training(tmp_path):
    """The strong form: a run saved and restored must take the SAME next step as
    one that never stopped. Adam moments are what make this non-trivial."""
    def run(save_restore):
        m = _model(n_bins=16)
        opt = torch.optim.AdamW(m.param_groups(LR, weight_decay=WD), lr=LR,
                                betas=(0.9, 0.95))
        for i in range(3):
            B = _batch(n_slot=ARCH["n_slot"], seed=100 + i)
            mask = torch.zeros(B["n_cells"], dtype=torch.bool); mask[::2] = True
            loss = m(B, tok_mask=mask)["loss"]
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if save_restore:
            p = tmp_path / "mid.pt"
            torch.save({"m": m.state_dict(), "o": opt.state_dict()}, p)
            m = _model(n_bins=16)
            opt = torch.optim.AdamW(m.param_groups(LR, weight_decay=WD), lr=LR,
                                    betas=(0.9, 0.95))
            blob = torch.load(p, map_location="cpu", weights_only=False)
            m.load_state_dict(blob["m"], strict=True)
            opt.load_state_dict(blob["o"])
        B = _batch(n_slot=ARCH["n_slot"], seed=999)
        mask = torch.zeros(B["n_cells"], dtype=torch.bool); mask[::2] = True
        loss = m(B, tok_mask=mask)["loss"]
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return float(loss), {n: p.detach().clone() for n, p in m.named_parameters()}

    l_a, p_a = run(False)
    l_b, p_b = run(True)
    assert l_a == l_b, f"loss after resume differs: {l_a} vs {l_b}"
    for k in p_a:
        assert torch.equal(p_a[k], p_b[k]), f"parameter {k} differs after resume"


# ---- 3. scheduler ---------------------------------------------------------

def test_onecycle_per_group_max_lr_preserves_muP_ratio():
    """pimm drives LRs with OneCycleLR, whose `max_lr` accepts a per-group LIST
    (that is how HMAE does layer-wise decay). Feeding it the muP ratios keeps
    them for the whole schedule; feeding it a scalar would flatten them.

    pimm's subclass only reinterprets pct_start > 1 as a warmup step count, so
    the per-group behaviour under test is torch's."""
    m = _model()
    opt = torch.optim.AdamW(m.param_groups(LR, weight_decay=WD), lr=LR,
                            betas=(0.9, 0.95))
    ratios = [g["lr"] / LR for g in opt.param_groups]
    m_width = ARCH["d"] // ARCH["d_base"]
    assert any(abs(r - 1.0 / m_width) < 1e-12 for r in ratios)

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR * r for r in ratios], total_steps=20, pct_start=0.25,
        anneal_strategy="cos", div_factor=10.0, final_div_factor=1000.0)

    seen_peak = 0.0
    for _ in range(20):
        opt.step()
        sched.step()
        lrs = [g["lr"] for g in opt.param_groups]
        base = lrs[0]
        seen_peak = max(seen_peak, base)
        for lr_i, r in zip(lrs, ratios):
            assert lr_i == pytest.approx(base * r / ratios[0], rel=1e-9), (
                "OneCycleLR broke the muP per-group ratio — the hidden group is "
                "no longer scaled by 1/m relative to the rest")
    assert seen_peak == pytest.approx(LR, rel=1e-6), (
        f"schedule never reached the requested peak LR ({seen_peak} vs {LR})")


def test_scalar_max_lr_would_discard_muP():
    """The failure this guards against, made explicit: a scalar max_lr assigns
    every group the same LR, so muP's 1/m on the hidden group disappears."""
    m = _model()
    opt = torch.optim.AdamW(m.param_groups(LR, weight_decay=WD), lr=LR,
                            betas=(0.9, 0.95))
    ratios = [g["lr"] / LR for g in opt.param_groups]
    assert len(set(ratios)) > 1, "fixture has no distinct muP groups"

    torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=10)
    flattened = {g["lr"] for g in opt.param_groups}
    assert len(flattened) == 1, (
        "a scalar max_lr no longer flattens the groups — if torch changed this, "
        "the per-group list may no longer be required")
