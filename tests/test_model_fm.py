"""The packaged FM must be the research model, not a lookalike.

``helix/model/`` was extracted from ``coeff_foundation_model/fm/model.py`` by
line slice precisely so this is checkable: the parameter tree is untouched, so a
research checkpoint loads ``strict=True`` and produces bit-identical outputs. An
earlier hand-transcribed version of this extraction silently dropped ``n_bins``
and ``losses_cat`` — which is exactly the failure mode these tests exist to
catch, since the model still built and still trained.

The parity test needs the research checkpoint and skips without it. The rest run
anywhere.
"""

import os

import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm, FMModel, losses, losses_fused, losses_cat  # noqa: E402

from _paths import RESEARCH_FM as RESEARCH                     # noqa: E402
CKPT = os.path.join(RESEARCH, "ckpt_clean160cat_m113_snap1000000.pt")

SMALL = dict(n_slot=8, n_band=4, n_plane=6, d=64, blocks=2, dec_blocks=1, heads=4,
             dec_mode="cross")      # SerialFMModel (the default) requires cross


def make_batch(n_cells=32, n_slot=8, n_band=4, n_plane=6, seed=0):
    """A batch with the same key contract helix.model.tokenize.to_fm() emits: the
    dense (n_cells, n_slot) grid AND the sparse active-row view of it, since
    losses() gathers rows while losses_fused()/losses_cat() read the grid."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    occ = (r(n_cells, n_slot) < 0.3).float()
    tgt = torch.randn(n_cells, n_slot, generator=g)
    cell, slot = occ.nonzero(as_tuple=True)          # sparse view of the same tokens
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
        cell=cell,
        slot=slot,
        n_cells=n_cells,
    )


# --------------------------------------------------------------------------
# the pimm Trainer contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("head", ["l2", "nll", "cat"])
def test_forward_returns_loss_dict(head, fused):
    """pimm's Trainer calls ``model(batch)["loss"]`` — every head mode must
    honour that, and the loss must carry grad."""
    cfg = dict(SMALL, loss_fused=fused)
    if head == "nll":
        cfg["nll"] = True
    elif head == "cat":
        cfg["n_bins"] = 16
    model = build_fm(cfg)
    if head == "cat":
        edges = torch.linspace(-4, 4, 17).repeat(cfg["n_band"], 1)
        model.set_bins(edges)

    out = model(make_batch(n_slot=cfg["n_slot"]))
    assert set(out) >= {"loss", "bce", "val", "masked_frac"}
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert out["loss"].requires_grad
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_cat_head_without_bins_is_refused():
    """A categorical model whose edges were never registered must fail loudly,
    not silently score against a garbage buffer."""
    model = build_fm(dict(SMALL, n_bins=16))
    with pytest.raises(AssertionError, match="set_bins"):
        model(make_batch(n_slot=SMALL["n_slot"]))


def test_set_bins_rejects_wrong_k():
    model = build_fm(dict(SMALL, n_bins=16))
    with pytest.raises(AssertionError, match="n_bins"):
        model.set_bins(torch.linspace(-4, 4, 9).repeat(4, 1))    # K+1 = 9, not 17


def test_val_head_width_tracks_head_mode():
    """The three head modes are distinguished only by val_head width; the
    converter infers the mode from it, so the mapping must hold."""
    assert build_fm(SMALL).val_head.weight.shape[0] == SMALL["n_slot"]
    assert build_fm(dict(SMALL, nll=True)).val_head.weight.shape[0] == 2 * SMALL["n_slot"]
    assert build_fm(dict(SMALL, n_bins=16)).val_head.weight.shape[0] == 16 * SMALL["n_slot"]


def test_explicit_mask_is_honoured():
    model = build_fm(SMALL)
    B = make_batch(n_slot=SMALL["n_slot"])
    m = torch.zeros(B["n_cells"], dtype=torch.bool)
    m[:4] = True
    assert model(B, tok_mask=m)["masked_frac"] == pytest.approx(4 / B["n_cells"])


def test_train_opts_are_not_constructor_params():
    """Training policy must stay off the arch signature: the converter maps a
    checkpoint's metadata onto constructor kwargs, so an arch kwarg that is
    really a training knob would corrupt that mapping."""
    import inspect
    params = set(inspect.signature(FMModel.__init__).parameters)
    assert not params & {"mask_mode", "mask_ratio", "loss_fused", "vis_w", "noisy"}
    m = build_fm(dict(SMALL, mask_ratio=0.75, mask_mode="plane"))
    assert (m.mask_ratio, m.mask_mode) == (0.75, "plane")


def test_build_fm_ignores_unknown_keys():
    """build_fm() is handed whole checkpoint/config dicts; stray keys must not
    raise."""
    m = build_fm(dict(SMALL, some_trainer_key=1, another="x"))
    assert m.d == SMALL["d"]


# --------------------------------------------------------------------------
# extraction fidelity — the whole point of the line-slice extraction
# --------------------------------------------------------------------------

GOLDEN = os.path.join(os.path.dirname(__file__), "goldens_fm.json")


@pytest.mark.skipif(not os.path.exists(GOLDEN), reason="no golden captured")
def test_matches_frozen_golden():
    """helix.model must still reproduce the m113 outputs frozen in
    tests/goldens_fm.json.

    That file was captured by tools/capture_fm_golden.py while the research tree
    still existed, and only after the two implementations were verified equal
    one final time — so this check inherits the bit-exact parity WITHOUT
    importing research/, which is what lets research/ be deleted.

    Deliberately NOT skipped when the anchor checkpoint is missing. An earlier
    version skipped, which meant deleting research/ would have made the whole
    guarantee vanish silently — the exact failure this golden exists to
    prevent. A missing anchor is a hard failure telling you to restore it."""
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "tools/capture_fm_golden.py", "--check"],
                       cwd=root, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": root})
    assert r.returncode == 0, r.stdout + r.stderr


def test_golden_was_witnessed():
    """A golden captured with --no-verify would enshrine whatever helix happened
    to produce. Only a research-verified capture is a real guarantee."""
    import json
    with open(GOLDEN) as f:
        g = json.load(f)
    assert g.get("verified_against_research") is True, \
        "golden was not cross-checked against the research implementation"
    assert "tokenizer" in g, "golden does not pin the token layout"


@pytest.mark.skipif(not os.path.exists(CKPT), reason=f"research checkpoint absent: {CKPT}")
def test_converter_infers_arch_from_tensors_alone():
    """The converter must not need the checkpoint's metadata to be right — it
    derives arch from shapes and only cross-checks against metadata."""
    from tools.convert_fm_ckpt import infer_arch, strip_ddp
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg, bad = infer_arch(strip_ddp(ck["model"]), meta={"heads": ck["heads"]})
    assert not bad, bad
    assert (cfg["d"], cfg["n_slot"], cfg["n_bins"]) == (512, 128, 128)
    assert (cfg["blocks"], cfg["dec_blocks"], cfg["dec_mode"]) == (12, 4, "cross")


@pytest.mark.skipif(not os.path.exists(CKPT), reason=f"research checkpoint absent: {CKPT}")
def test_converter_flags_metadata_disagreeing_with_weights():
    """A metadata field that contradicts the tensors is reported, not trusted."""
    from tools.convert_fm_ckpt import infer_arch, strip_ddp
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    _, bad = infer_arch(strip_ddp(ck["model"]), meta={"heads": 8, "blocks": 99})
    assert any("blocks" in b for b in bad), bad


def test_serial_rejects_self_decoder():
    """SerialFMModel's decoder is grouped-cross attention; with dec_mode='self'
    the research code dies with an AttributeError deep in attention. build_fm
    must refuse it up front."""
    with pytest.raises(ValueError, match="dec_mode='cross'"):
        build_fm(dict(SMALL, dec_mode="self"), serial=True)
    build_fm(dict(SMALL, dec_mode="self"), serial=False)      # fine unserialised


def test_missing_batch_keys_name_themselves():
    """A batch lacking what the configured objective reads must say which keys
    and which objective — not raise a bare KeyError from inside the loss."""
    model = build_fm(SMALL)                       # sparse losses(): needs cell/slot/target
    B = make_batch(n_slot=SMALL["n_slot"])
    for k in ("cell", "slot", "target"):
        B.pop(k)
    with pytest.raises(KeyError, match="losses"):
        model(B)
