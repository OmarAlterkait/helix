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

from _paths import PIMM_ROOT as PIMM, CORPUS                   # noqa: E402

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

