"""Training through pimm: the batch through pimm's step, AMP and resume, the WSD schedules, and
the derived cadences."""
import importlib.util

import pytest

torch = pytest.importorskip("torch")

#: The WSD tests import pimm (they had a file of their own guarded on it).
needs_pimm = pytest.mark.skipif(importlib.util.find_spec("pimm") is None, reason="pimm not installed")

import ast
import os
import pathlib
import math

import numpy as np

from _fixtures import PIMM_ROOT as PIMM, CORPUS                   # noqa: E402
from helix.model import build_fm  # noqa: E402
from helix.model import schedule as cadence


# ---- the batch through pimm's step -------------------------------------------------------

# The batch must survive what pimm's Trainer actually does to it.
#
# Every other test here calls the model directly. pimm does not: it runs the
# sample through a transform pipeline, then ``collate_fn``, then
# ``move_batch_to_device``, and only then ``model(input_dict)``. Each of those can
# reshape a batch, and the first attempt at this integration was broken at all
# three points while every model-level test stayed green:
#
#   * the batch stayed nested under its part, so the model saw no ``plane_id``
#   * ``collate_fn`` STACKS non-tensor leaves (numpy -> ``default_collate``) while
#     it CONCATENATES tensors, giving ``inp`` a spurious ``(1, n_cells, n_slot)``
#   * ``n_cells`` arrived as ``tensor([30976])`` where ``make_mask`` wants an int
#
# pimm's helpers are compiled from their source rather than imported, because
# importing ``pimm.datasets`` pulls optional dependencies (pyarrow, addict) that a
# helix environment has no reason to carry. The functions themselves only need
# torch, so this tests the REAL code without the package's import graph.
#
# Skips when pimm's source tree is absent.

pimm_src = pytest.mark.skipif(
    not os.path.isdir(PIMM), reason=f"pimm source tree absent: {PIMM}")
corpus = pytest.mark.skipif(
    not os.path.isdir(CORPUS), reason=f"corpus absent: {CORPUS}")


def _from_source(relpath, *names, preamble=""):
    """Compile named top-level functions out of a pimm module, no import."""
    src = pathlib.Path(PIMM, relpath).read_text()
    ns = {"torch": torch, "np": np}
    if preamble:
        exec(preamble, ns)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.get_source_segment(src, node), relpath, "exec"), ns)
    missing = [n for n in names if n not in ns]
    assert not missing, f"{relpath}: could not extract {missing}"
    return tuple(ns[n] for n in names)


@pytest.fixture(scope="module")
def pimm_collate():
    (fn,) = _from_source(
        "pimm/datasets/utils.py", "collate_fn",
        preamble=("from collections.abc import Mapping, Sequence\n"
                  "from torch.utils.data.dataloader import default_collate\n"
                  "import random"))
    return fn


@pytest.fixture(scope="module")
def pimm_to_device():
    (fn,) = _from_source("pimm/distributed/distributed.py", "move_batch_to_device",
                         preamble="from typing import Any\nimport torch.nn as nn")
    return fn


@pytest.fixture(scope="module")
def sample():
    """One real corpus event through tokenize + the collect adapter."""
    pytest.importorskip("pimm_data")
    from helix.data import CoeffTPCDataset
    from helix.model.tokenize import CoeffTokenize

    ds = CoeffTPCDataset(data_root=CORPUS, dataset_name="sim_wire", transform=None)
    tok = CoeffTokenize(part="coeff", cfg=dict())(ds.get_data(0))
    # CoeffCollect without importing helix.integrations.pimm (which needs pimm):
    # exercise the same logic the registered transform applies.
    out = {k: torch.from_numpy(np.ascontiguousarray(v))
           for k, v in tok["coeff"].items()
           if isinstance(v, np.ndarray) and not k.startswith("_") and k != "n_cells"}
    return out


@pimm_src
@corpus
def test_collate_preserves_shapes_at_batch_size_one(pimm_collate, sample):
    """No spurious batch dimension. Tensors take collate's CONCAT path, so a
    single sample passes through unchanged — numpy would have been STACKED."""
    out = pimm_collate([sample])
    assert out["inp"].shape == sample["inp"].shape, (
        f"inp gained a dimension: {tuple(out['inp'].shape)} vs "
        f"{tuple(sample['inp'].shape)} — a non-tensor leaf reached default_collate")
    assert out["plane_id"].shape == sample["plane_id"].shape
    assert out["inp"].dim() == 2 and out["plane_id"].dim() == 1


@pimm_src
@corpus
def test_batch_carries_no_offset(pimm_collate, sample):
    """Our batch must carry NO `offset`, and this asserts it directly.

    pimm's run_step does
    `if "offset" in input_dict: input_dict["coord"].shape[0]`, so an offset
    without a coord raises KeyError AFTER the forward — a crash on step 1. We
    stay clear of it by never emitting one, which is a property of CoeffCollect
    rather than a happy accident, so it is worth asserting.

    Written as an unconditional assertion on purpose: the obvious form,
    `if "offset" in out: assert "coord" in out`, asserts NOTHING while no offset
    exists, which is exactly the state we are in.

    `offset` only enters a batch when a transform emits one (pimm's `Collect`
    via offset_keys_dict, multiview, hmae) — collate never adds it. So the way
    this breaks is someone reaching for pimm's `Collect`, the natural idiom, in
    place of `CoeffCollect`."""
    out = pimm_collate([sample])
    assert "offset" not in out, (
        "batch gained an 'offset'. pimm's run_step will then read "
        "input_dict['coord'], which the FM does not have -> KeyError after the "
        "forward. Either drop the offset or add a coord.")
    assert "coord" not in out, "unexpected 'coord' — has the batch contract changed?"


@pimm_src
@corpus
def test_full_pimm_step_path(pimm_collate, pimm_to_device, sample):
    """dataset -> tokenize -> collect -> collate -> to_device -> model(batch).

    The end-to-end shape pimm's Trainer executes, asserted to produce the
    contract it consumes: an output dict carrying a finite scalar ``loss`` that
    carries grad."""
    from helix.model import build_fm

    torch.manual_seed(0)
    n_slot = sample["inp"].shape[1]
    model = build_fm(dict(n_slot=n_slot, n_band=4, n_plane=6, d=64, blocks=2,
                          dec_blocks=1, heads=4, n_bins=16))
    model.set_bins(torch.linspace(-4, 4, 17).repeat(4, 1))

    batch = pimm_to_device(pimm_collate([sample]), torch.device("cpu"))
    out = model(batch)                       # exactly how run_step calls it

    assert isinstance(out, dict) and "loss" in out
    loss = out["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss), f"loss not a finite scalar: {loss}"
    assert loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


@pimm_src
@corpus
def test_n_cells_is_derived_not_collated(pimm_collate, sample):
    """`n_cells` must not survive collate as a tensor: make_mask does
    torch.rand(n) with it. forward derives it from plane_id instead."""
    assert "n_cells" not in sample
    out = pimm_collate([sample])
    assert "n_cells" not in out
    from helix.model import build_fm
    m = build_fm(dict(n_slot=sample["inp"].shape[1], n_band=4, n_plane=6, d=32,
                      blocks=1, dec_blocks=1, heads=4, n_bins=16))
    B = dict(out)
    B.setdefault("n_cells", B["plane_id"].shape[0])
    assert isinstance(B["n_cells"], int)
    assert m.make_mask(B).shape == (sample["plane_id"].shape[0],)


# ---- AMP, resume and the schedule --------------------------------------------------------

# Two things pimm's Trainer does around a step.
#
# The section above covers the batch reaching the model. This covers
# what happens around the step:
#
#   AMP        run_step wraps the forward in bf16 autocast. The parity tests are
#              fp32, and a 128-way log-softmax is exactly the shape that loses
#              precision quietly rather than producing NaN.
#   resume     a restored optimizer must keep the muP per-group lr/weight_decay.
#              Losing them on resume rescales learning across width for the rest
#              of the run, and the loss curve would look plausible throughout.
#
# These use torch directly rather than pimm's wrappers where the behaviour under
# test is torch's (autocast, AdamW state). The schedules' muP property is in
# `test_model.py` (muP under a schedule).

ARCH = dict(n_slot=16, n_band=4, n_plane=6, d=64, blocks=2, dec_blocks=1,
            heads=4, mup=True, d_base=32, n_bins=16)
LR, WD = 3e-4, 0.05


def _batch(n_cells=48, n_slot=16, seed=7):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    occ = (r(n_cells, n_slot) < 0.4).float()
    tgt = torch.randn(n_cells, n_slot, generator=g)
    return dict(band_id=torch.randint(0, 4, (n_cells,), generator=g),
                plane_id=torch.randint(0, 6, (n_cells,), generator=g),
                t_phys=torch.randn(n_cells, generator=g) * 500,
                wire_pos=r(n_cells) * 1900, wirefeat=r(n_cells, 1),
                inp=torch.randn(n_cells, n_slot, generator=g), occ=occ,
                valid=(r(n_cells, n_slot) < 0.9), tgt=tgt, n_cells=n_cells)


def _model(**over):
    torch.manual_seed(0)
    cfg = dict(ARCH, **over)
    m = build_fm(cfg)
    m.set_bins(torch.linspace(-4, 4, cfg["n_bins"] + 1).repeat(cfg["n_band"], 1))
    return m


@pytest.fixture
def _one_thread():
    """One intra-op thread for the AMP and resume tests (was autouse when they had a file of
    their own)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


# ---- 1. AMP ---------------------------------------------------------------

@pytest.mark.usefixtures("_one_thread")
def test_bf16_autocast_loss_is_finite_and_close_to_fp32():
    """pimm runs the forward under bf16 autocast. bfloat16 has ~3 decimal digits
    of mantissa, so the question is not whether the loss is NaN but whether it is
    still the same number to a useful tolerance."""
    m = _model()
    B = _batch(n_slot=ARCH["n_slot"])
    mask = torch.zeros(B["n_cells"], dtype=torch.bool); mask[::2] = True

    with torch.no_grad():
        fp32 = float(m(B, tok_mask=mask)["loss"])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            bf16 = float(m(B, tok_mask=mask)["loss"])
    assert torch.isfinite(torch.tensor(bf16)), "bf16 loss is not finite"
    # The 1% bound is loose enough not to be brittle and tight enough that a
    # real precision regression — e.g. a reduction moved into bf16 — would trip it.
    assert bf16 == pytest.approx(fp32, rel=0.01), (
        f"bf16 loss {bf16} diverges from fp32 {fp32} by more than 1%")


@pytest.mark.usefixtures("_one_thread")
def test_bf16_backward_produces_finite_gradients():
    """A finite loss with NaN/Inf gradients would poison the optimizer state on
    the first step and never recover."""
    m = _model()
    B = _batch(n_slot=ARCH["n_slot"])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = m(B)["loss"]
    loss.backward()
    grads = [(n, p.grad) for n, p in m.named_parameters() if p.grad is not None]
    assert grads, "no gradients at all"
    bad = [n for n, g in grads if not torch.isfinite(g).all()]
    assert not bad, f"non-finite gradients in {bad[:5]}"
    assert any(g.abs().sum() > 0 for _, g in grads), "all gradients are exactly zero"


# ---- 2. resume ------------------------------------------------------------

@pytest.mark.usefixtures("_one_thread")
def test_optimizer_resume_preserves_muP_groups(tmp_path):
    """Save mid-run, restore, and the per-group lr/weight_decay must come back.

    AdamW's state_dict stores param_groups, so this is really asserting that the
    muP structure round-trips rather than being rebuilt from a flat config on
    resume — which is how it would silently be lost."""
    m = _model()
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

    m2 = _model()
    opt2 = torch.optim.AdamW(m2.param_groups(LR, weight_decay=WD), lr=LR,
                             betas=(0.9, 0.95))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m2.load_state_dict(blob["model"], strict=True)
    opt2.load_state_dict(blob["optimizer"])

    after = [(g["lr"], g["weight_decay"]) for g in opt2.param_groups]
    assert after == before, f"muP groups changed across resume: {before} -> {after}"
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p1.detach(), p2.detach())


@pytest.mark.usefixtures("_one_thread")
def test_step_after_resume_matches_uninterrupted_training(tmp_path):
    """The strong form: a run saved and restored must take the SAME next step as
    one that never stopped. Adam moments are what make this non-trivial."""
    def run(save_restore):
        m = _model()
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
            m = _model()
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


# ---- the WSD schedules -------------------------------------------------------------------

# WSDCooldownLR: pure cooldown at stable_frac=0, full warmup-stable-decay above.

def _lrs(steps, **kw):
    from helix.integrations.pimm.trainer import WSDCooldownLR
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sch = WSDCooldownLR(opt, **kw)
    out = []
    for _ in range(steps + 1):
        out.append(opt.param_groups[0]["lr"])
        opt.step(); sch.step()
    return out


@needs_pimm
def test_stable_frac_zero_is_the_plain_cooldown():
    lr = _lrs(100, total_steps=100, warmup=0)
    assert lr[0] == 1.0 and lr[25] == pytest.approx(0.5)
    assert lr[100] == pytest.approx(1e-3)


@needs_pimm
def test_warmup_then_flat_then_one_minus_sqrt():
    lr = _lrs(110, total_steps=110, warmup=10, stable_frac=0.5)
    assert lr[5] == pytest.approx(0.5)                       # warmup
    assert all(v == 1.0 for v in lr[10:61])                  # flat to 10 + 50
    assert lr[85] == pytest.approx(1 - math.sqrt(0.5))       # halfway down
    assert lr[110] == pytest.approx(1e-3)                    # floor


# ---- the derived cadences (helix.model.schedule) -----------------------------------------

# Warmup and cadence follow the run's step count (helix.model.schedule).

def test_warmup_scales_with_width_and_is_at_most_a_fifth_of_the_run():
    assert cadence.warmup_steps(1_000_000, 128, 128) == 500
    assert cadence.warmup_steps(1_000_000, 512, 128) == 1000
    assert cadence.warmup_steps(25, 512, 128) == 5
    assert cadence.warmup_steps(3, 128, 128) == 1


def test_a_short_run_still_evaluates_and_saves():
    assert 25 // cadence.eval_every(25) >= cadence.MIN_EVALS
    assert 25 // cadence.save_every(25) >= cadence.MIN_SAVES
