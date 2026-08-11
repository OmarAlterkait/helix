"""The batch must survive what pimm's Trainer actually does to it.

Every other test here calls the model directly. pimm does not: it runs the
sample through a transform pipeline, then ``collate_fn``, then
``move_batch_to_device``, and only then ``model(input_dict)``. Each of those can
reshape a batch, and the first attempt at this integration was broken at all
three points while every model-level test stayed green:

  * the batch stayed nested under its part, so the model saw no ``plane_id``
  * ``collate_fn`` STACKS non-tensor leaves (numpy -> ``default_collate``) while
    it CONCATENATES tensors, giving ``inp`` a spurious ``(1, n_cells, n_slot)``
  * ``n_cells`` arrived as ``tensor([30976])`` where ``make_mask`` wants an int

pimm's helpers are compiled from their source rather than imported, because
importing ``pimm.datasets`` pulls optional dependencies (pyarrow, addict) that a
helix environment has no reason to carry. The functions themselves only need
torch, so this tests the REAL code without the package's import graph.

Skips when pimm's source tree is absent.
"""

import ast
import os
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

PIMM = "/sdf/group/neutrino/omara/pimm-fm"
CORPUS = "/sdf/data/neutrino/omara/coeff_tpc/run_0027575715"

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
    from pimm_data import CoeffTPCDataset
    from helix.model.tokenize import CoeffTokenize

    ds = CoeffTPCDataset(data_root=CORPUS, dataset_name="sim_wire",
                         modalities=("coeff", "coeff_clean"), transform=None)
    tok = CoeffTokenize(part="coeff", clean_part="coeff_clean")(ds.get_data(0))
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
def test_no_offset_without_coord(pimm_collate, sample):
    """pimm's run_step does `if "offset" in input_dict: input_dict["coord"]...`,
    so emitting an offset without a coord KeyErrors AFTER the forward — a crash
    on step 1 that no model-level test would show."""
    out = pimm_collate([sample])
    if "offset" in out:
        assert "coord" in out, (
            "batch carries 'offset' but no 'coord'; pimm's run_step will raise "
            "KeyError after the forward pass")


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
                          dec_blocks=1, heads=4, dec_mode="cross", n_bins=16))
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
                      blocks=1, dec_blocks=1, heads=4, dec_mode="cross"))
    B = dict(out)
    B.setdefault("n_cells", B["plane_id"].shape[0])
    assert isinstance(B["n_cells"], int)
    assert m.make_mask(B).shape == (sample["plane_id"].shape[0],)


# ---- the eval hook we intend to reuse -------------------------------------

MAE_HOOK = os.path.join(PIMM, "pimm/engines/hooks/eval/pretrain/mae.py")


@pytest.mark.skipif(not os.path.exists(MAE_HOOK), reason="pimm MAEEvaluator absent")
def test_mae_evaluator_call_contract_is_documented():
    """pimm's MAEEvaluator is the obvious hook to reuse — the coeff FM IS a
    masked autoencoder — but it does not fit as-is, in two different ways.

    This pins both so the mismatch is a known quantity rather than a surprise on
    the first eval step. It reads the hook's SOURCE (no import: the module pulls
    wandb and pimm's package graph)."""
    src = pathlib.Path(MAE_HOOK).read_text()

    # 1. HARD failure: it passes a kwarg our forward does not accept.
    assert "return_pred=return_viz" in src
    import inspect
    from helix.model import FMModel
    params = inspect.signature(FMModel.forward).parameters
    accepts = "return_pred" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    assert not accepts, (
        "FMModel.forward now accepts return_pred — MAEEvaluator is compatible, "
        "so delete this test and the note in TODO.md")

    # 2. SILENT degradation: it reads metric names we do not emit, via .get(...,
    #    0.0), so the numbers would be zero rather than wrong-looking.
    for k in ("coord_loss", "feat_loss", "mask_ratio_actual"):
        assert f'output_dict.get("{k}"' in src
    torch.manual_seed(0)
    m = build_small()
    out = m(_tiny_batch())
    assert set(out) == {"loss", "bce", "val", "masked_frac"}
    assert not ({"coord_loss", "feat_loss", "mask_ratio_actual"} & set(out)), (
        "the model now emits MAEEvaluator's metric names — update this test")


def build_small():
    from helix.model import build_fm
    return build_fm(dict(n_slot=8, n_band=4, n_plane=6, d=32, blocks=1,
                         dec_blocks=1, heads=4, dec_mode="cross"))


def _tiny_batch(n_cells=16, n_slot=8):
    g = torch.Generator().manual_seed(0)
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
