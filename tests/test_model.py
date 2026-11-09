"""helix.model.fm: the model, its training contracts, and the muP ratios a scheduler keeps."""
import pytest

torch = pytest.importorskip("torch")

import numpy as np

from helix.model import build_fm, FMModel                          # noqa: E402
from helix.model.denoise import pad_input_columns                  # noqa: E402
from helix.model.fm import make_mask, VIEWS_PER_VOLUME             # noqa: E402
from helix.model.schedule import param_group_ratios  # noqa: E402


# ---- the model (helix.model.fm) ----------------------------------------------------------

# The FM's construction and its pimm Trainer contract.
#
# The arithmetic of the forward is pinned in tests/test_attention.py; this file pins
# what a caller relies on: ``model(batch)`` returns a loss dict, the head is
# categorical and refuses to run without its bin edges, the config is strict, and
# the masking policy mixes modes per step.

SMALL = dict(n_slot=8, n_band=4, n_plane=6, d=64, blocks=2, dec_blocks=1, heads=4,
             n_bins=16)


def make_batch(n_cells=32, n_slot=8, n_band=4, n_plane=6, seed=0):
    """A batch with the key contract helix.model.tokenize.to_fm() emits."""
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


def binned(model):
    """``model`` with bin edges registered, as every forward requires."""
    return model.set_bins(torch.linspace(-4, 4, model.n_bins + 1).repeat(SMALL["n_band"], 1))


# --------------------------------------------------------------------------
# the pimm Trainer contract
# --------------------------------------------------------------------------

def test_forward_returns_loss_dict():
    """pimm's Trainer calls ``model(batch)["loss"]``; the loss must carry grad."""
    model = binned(build_fm(SMALL))
    out = model(make_batch(n_slot=SMALL["n_slot"]))
    assert set(out) >= {"loss", "bce", "val", "masked_frac"}
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert out["loss"].requires_grad
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_forward_without_bins_is_refused():
    """A model whose edges were never registered must fail loudly, not silently
    score against a garbage buffer."""
    model = build_fm(SMALL)
    with pytest.raises(AssertionError, match="set_bins"):
        model(make_batch(n_slot=SMALL["n_slot"]))


def test_set_bins_rejects_wrong_k():
    model = build_fm(SMALL)
    with pytest.raises(AssertionError, match="n_bins"):
        model.set_bins(torch.linspace(-4, 4, 9).repeat(4, 1))    # K+1 = 9, not 17


def test_the_head_is_categorical_and_needs_bins():
    assert build_fm(SMALL).val_head.weight.shape[0] == SMALL["n_bins"] * SMALL["n_slot"]
    with pytest.raises(ValueError, match="n_bins"):
        build_fm(dict(SMALL, n_bins=0))
    with pytest.raises(TypeError, match="n_bins"):
        build_fm({k: v for k, v in SMALL.items() if k != "n_bins"})


def test_explicit_mask_is_honoured():
    model = binned(build_fm(SMALL))
    B = make_batch(n_slot=SMALL["n_slot"])
    m = torch.zeros(B["n_cells"], dtype=torch.bool)
    m[:4] = True
    assert model(B, tok_mask=m)["masked_frac"] == pytest.approx(4 / B["n_cells"])


def test_train_opts_are_not_constructor_params():
    """Training policy must stay off the arch signature: the constructor is the
    architecture a checkpoint records, so a training knob there would corrupt
    that record."""
    import inspect
    params = set(inspect.signature(FMModel.__init__).parameters)
    assert not params & {"mask_mode", "mask_ratio", "plane_frac", "compile_blocks"}
    m = build_fm(dict(SMALL, mask_ratio=0.6, mask_mode="plane"))
    assert (m.mask_ratio, m.mask_mode) == (0.6, "plane")


def test_build_fm_refuses_unknown_keys():
    """A key build_fm does not consume would otherwise do nothing, silently."""
    with pytest.raises(TypeError, match="some_trainer_key"):
        build_fm(dict(SMALL, some_trainer_key=1))


def test_missing_batch_keys_name_themselves():
    """A batch lacking what the objective reads must say which keys -- not raise
    a bare KeyError from inside the loss."""
    model = binned(build_fm(SMALL))
    B = make_batch(n_slot=SMALL["n_slot"])
    B.pop("tgt")
    with pytest.raises(KeyError, match="tgt"):
        model(B)


def test_plane_frac_mixes_modes_per_step():
    """`plane_frac` is a per-STEP mix, not a mode.

    A run that only ever masks randomly never has to reconstruct a plane it
    cannot see, so nothing forces cross-plane triangulation -- which is the
    whole reason the option exists.
    """
    B = make_batch(n_slot=SMALL["n_slot"])
    gid = B["plane_id"]

    def whole_plane(msk):
        return all(bool(msk[gid == g].all()) or bool((~msk[gid == g]).all())
                   for g in gid.unique())

    counts = {}
    for pf in (0.0, 0.5, 1.0):
        model = build_fm(dict(SMALL), plane_frac=pf)
        torch.manual_seed(0)
        counts[pf] = sum(whole_plane(model.make_mask(B)) for _ in range(200))

    assert counts[0.0] == 0, "plane_frac=0 must never mask a whole plane"
    assert counts[1.0] == 200, "plane_frac=1 must always mask whole planes"
    assert 60 < counts[0.5] < 140, f"plane_frac=0.5 should mix, got {counts[0.5]}/200"

    # An explicit mode= overrides the policy and skips the draw entirely.
    model = build_fm(dict(SMALL), plane_frac=1.0)
    assert not whole_plane(model.make_mask(B, mode="random"))


# ---- training contracts ------------------------------------------------------------------

# The training step's CONTRACTS: muP scaling, mask reproducibility, and that a
# step moves what it should.
#
# These assert invariants rather than recorded outputs: they are what has to stay
# true across a width change, a new masking policy, or a different bin grid.

def _batch(n=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(n_cells=n, plane_id=torch.arange(n) % 6,
                wire_pos=torch.rand(n, generator=g) * 1900,
                band_id=torch.randint(0, 4, (n,), generator=g))


# ---------------------------------------------------------------- muP contract

@pytest.mark.parametrize("d,d_base", [(64, 64), (128, 64), (256, 64)])
def test_hidden_lr_scales_as_one_over_width_multiplier(d, d_base):
    """HIDDEN gets base_lr/m; everything else base_lr. That is the whole of muP's
    LR rule, and it must hold AT EVERY WIDTH -- which is exactly what a frozen
    group list could not express, since the groups change with d."""
    m = build_fm(dict(SMALL, d=d, heads=max(1, d // 16), mup=True, d_base=d_base))
    groups = m.param_groups(1e-3, weight_decay=0.05)
    mult = d / d_base
    lrs = sorted({g["lr"] for g in groups})
    assert lrs == sorted({1e-3, 1e-3 / mult}) if mult != 1 else lrs == [1e-3]


def test_weight_decay_is_decoupled_from_the_mup_lr_scaling():
    """AdamW applies decay as lr*wd*p, so a hidden lr of base/m would shrink
    effective decay by 1/m across width -- a width-scaling confound. The hidden
    group's wd is multiplied by m so lr*wd is width-invariant."""
    m = build_fm(dict(SMALL, d=128, heads=8, mup=True, d_base=64))
    for g in m.param_groups(1e-3, weight_decay=0.05):
        if g["weight_decay"] != 0.0:                    # skip the nodecay group
            assert g["lr"] * g["weight_decay"] == pytest.approx(1e-3 * 0.05), \
                "effective decay (lr*wd) is not width-invariant"


def test_one_dimensional_parameters_never_get_weight_decay():
    """Biases, LayerNorm gains and 1-D tokens must not be decayed (nanoGPT/timm
    standard). A new 1-D parameter landing in the decay bucket is silent."""
    m = build_fm(dict(SMALL))
    for g in m.param_groups(1e-3, weight_decay=0.05):
        if g["weight_decay"] == 0.0:
            assert all(p.dim() < 2 for p in g["params"])
        else:
            assert all(p.dim() >= 2 for p in g["params"])


def test_every_trainable_parameter_lands_in_exactly_one_group():
    """A parameter in no group is never optimised; in two, it is updated twice."""
    m = build_fm(dict(SMALL, mup=True, d_base=32))
    seen = [p for g in m.param_groups(1e-3, weight_decay=0.05) for p in g["params"]]
    ids = [id(p) for p in seen]
    trainable = [p for p in m.parameters() if p.requires_grad]
    assert len(ids) == len(set(ids)), "a parameter appears in two groups"
    assert set(ids) == {id(p) for p in trainable}, "a trainable parameter is unoptimised"


# ------------------------------------------------------- mask reproducibility

@pytest.mark.parametrize("mode", ["random", "plane"])
def test_the_same_generator_seed_gives_the_same_mask(mode):
    """Reproducibility is the contract; the exact token selection is NOT.

    Pinning which tokens get drawn would break the moment the masking policy
    changes -- which is planned work. That the same seed reproduces the same
    draw is what the evaluator relies on and what must survive.
    """
    B = _batch()
    draws = {make_mask(B, mode, 0.5, 1, torch.Generator().manual_seed(11))
             .numpy().tobytes() for _ in range(6)}
    assert len(draws) == 1, f"{mode} is not reproducible under a fixed generator"


def test_different_seeds_give_different_masks():
    """The counterpart: a seeded draw must still be a DRAW, not a constant."""
    B = _batch()
    draws = {make_mask(B, "random", 0.5, 1, torch.Generator().manual_seed(s))
             .numpy().tobytes() for s in range(6)}
    assert len(draws) > 1, "the mask ignores its generator"


def test_random_mode_honours_the_ratio():
    B = _batch(n=4000)
    for ratio in (0.25, 0.75):
        m = make_mask(B, "random", ratio, 1, torch.Generator().manual_seed(3))
        assert abs(float(m.float().mean()) - ratio) < 0.03


# --------------------------------------------------------- full-step wiring

def test_a_step_updates_exactly_the_parameters_that_got_gradient():
    """forward -> loss -> backward -> AdamW.step over the model's OWN param_groups.

    Every parameter carrying gradient moves, and nothing else does.
    """

    model = binned(build_fm(dict(SMALL, mup=True, d_base=32)))
    B = make_batch(n_slot=SMALL["n_slot"], n_band=SMALL["n_band"],
                   n_plane=SMALL["n_plane"])
    opt = torch.optim.AdamW(model.param_groups(1e-2, weight_decay=0.0))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    out = model(B)
    out["loss"].backward()
    got_grad = {n for n, p in model.named_parameters()
                if p.grad is not None and p.grad.abs().sum() > 0}
    assert got_grad, "no parameter received gradient"
    opt.step()

    moved = {n for n, p in model.named_parameters()
             if not torch.equal(p.detach(), before[n])}
    assert moved == got_grad, (
        f"moved-but-no-grad: {sorted(moved - got_grad)}; "
        f"grad-but-unmoved: {sorted(got_grad - moved)}")


# ------------------------------------------------ masking policy

def test_plane_mask_punctures_every_volume():
    """'plane' hides n_planes views PER VOLUME, leaving no volume intact.

    The point of the mode is to force cross-plane triangulation: a volume left
    fully sighted would let the model interpolate from it instead.
    """
    B = _batch()
    gid = B["plane_id"]
    gids = gid.unique()
    vols = torch.div(gids, VIEWS_PER_VOLUME, rounding_mode="floor").unique()
    assert len(vols) > 1, "fixture must span >1 volume or this proves nothing"

    for n_planes in (1, 2):
        for seed in range(25):
            m = make_mask(B, "plane", 0.5, n_planes,
                          torch.Generator().manual_seed(seed))
            hit = gid[m].unique()
            # whole planes, never partial ones
            for g in hit:
                assert bool(m[gid == g].all()), f"gid {int(g)} only partly masked"
            # every volume punctured, exactly n_planes of its views
            per_vol = torch.div(hit, VIEWS_PER_VOLUME, rounding_mode="floor")
            for v in vols:
                k = int((per_vol == v).sum())
                assert k == n_planes, (
                    f"volume {int(v)} lost {k} planes, expected {n_planes} — a "
                    f"fully sighted volume lets the model interpolate instead of "
                    f"triangulating")


def test_unknown_mask_mode_is_refused():
    with pytest.raises(ValueError, match="mask mode"):
        make_mask(_batch(), "plane_any", 0.5, 1)


def test_plane_frac_steps_mask_planes_in_every_volume():
    """The mixer's plane steps use the 'plane' selection."""
    B = _batch()
    gid = B["plane_id"]
    vol_of = lambda g: torch.div(g, VIEWS_PER_VOLUME, rounding_mode="floor")
    vols = vol_of(gid.unique()).unique()
    model = build_fm(dict(SMALL), plane_frac=1.0, n_planes=1)
    torch.manual_seed(0)
    intact = sum(len(vol_of(gid[model.make_mask(B)].unique()).unique()) < len(vols)
                 for _ in range(50))
    assert intact == 0, "a plane_frac step left a volume fully sighted"


# ---- muP under a schedule (helix.model.schedule) -----------------------------------------

# muP's per-group LR must survive the scheduler.
#
# `FMModel.param_groups` encodes muP as a per-group `lr`: hidden weights get
# `base_lr / m`. That encoding has exactly one fragile point — a scheduler that
# assigns a single LR to every group discards it, and the failure is invisible.
# The loss still falls, just to a worse place, and only a run at another width
# would reveal it.
#
# The WSD schedules are `LambdaLR`s, which scale each group's own `lr`; driving
# a real LambdaLR over muP's groups covers the behaviour that matters without
# pimm installed. `helix.model.schedule.param_group_ratios` reads the ratios back.

ARCH = dict(n_slot=16, n_band=4, n_plane=6, d=64, blocks=2, dec_blocks=1,
            heads=4, mup=True, d_base=32, n_bins=16)
LR, WD = 3e-4, 0.05


def _opt():
    torch.manual_seed(0)
    m = build_fm(dict(ARCH))
    return m, torch.optim.AdamW(m.param_groups(LR, weight_decay=WD), lr=LR,
                                betas=(0.9, 0.95))


def test_ratios_capture_the_width_multiplier():
    _, opt = _opt()
    ratios = param_group_ratios(opt.param_groups, LR)
    m_width = ARCH["d"] // ARCH["d_base"]
    assert any(abs(r - 1.0 / m_width) < 1e-12 for r in ratios), (
        f"no group scaled by 1/m = {1/m_width}; muP is not active")
    assert any(abs(r - 1.0) < 1e-12 for r in ratios), "no unscaled group"


def test_lambda_schedule_preserves_ratios():
    """The end-to-end property: drive a LambdaLR (what WSDStableLR and
    WSDCooldownLR are) through warmup and decay and the muP ratios must hold
    at every step."""
    import math
    _, opt = _opt()
    ratios = param_group_ratios(opt.param_groups, LR)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 5) * max(1e-3, 1.0 - math.sqrt(s / 25)))
    for _ in range(25):
        opt.step(); sched.step()
        lrs = [g["lr"] for g in opt.param_groups]
        for lr_i, r in zip(lrs, ratios):
            assert lr_i == pytest.approx(lrs[0] * r / ratios[0], rel=1e-9)


def test_ratios_reject_a_group_without_an_lr():
    """Guard the assumption: a group with no 'lr' has no defined ratio, and
    silently treating it as 1.0 would mis-scale it forever."""
    with pytest.raises(ValueError, match="no 'lr'"):
        param_group_ratios([{"params": []}], LR)
    with pytest.raises(ValueError, match="non-zero"):
        param_group_ratios([{"params": [], "lr": 1e-4}], 0)


def test_trainer_wires_the_helpers():
    """FMTrainer is unimportable without pimm, so check by source that its
    optimizer override is the one described — and that it does not silently
    no-op."""
    import pathlib
    # `trainer` specifically, not the whole integration: since the split, the
    # file that defines FMTrainer is the file this reads.
    src = (pathlib.Path(__file__).parent.parent
           / "helix" / "integrations" / "pimm" / "trainer.py")
    text = src.read_text()
    assert "class FMTrainer(Trainer)" in text
    assert "def build_optimizer" in text and "model.param_groups(" in text
    assert "unwrap_model(self.model)" in text, (
        "build_model wraps for DDP, so param_groups must come from the unwrapped "
        "model")
    assert "param_dicts" in text, (
        "FMTrainer must reject cfg.param_dicts rather than silently ignoring it")


# ---- the linear-amplitude input channel (lin_amp_scale) -----------------------------------

# 0 is the model without the channel (same parameters, same output); S > 0 widens the embedding
# by n_slot columns that read sinh(inp) / S, start at zero, and leave every other parameter drawn
# as at 0; pad_input_columns loads a checkpoint without the columns (old copied, new zeroed).

N = SMALL["n_slot"]


def _pair(scale, seed=0):
    torch.manual_seed(seed)
    a = build_fm(dict(SMALL)).eval()
    torch.manual_seed(seed)
    return a, build_fm(dict(SMALL, lin_amp_scale=scale)).eval()


def test_zero_scale_is_the_model_without_the_channel():
    a, b = _pair(0.0)
    assert b.embed.in_features == 2 * N
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys() and all(torch.equal(pa[k], pb[k]) for k in pa)
    B = make_batch(n_slot=N)
    with torch.no_grad():
        assert torch.equal(a.encode(B), b.encode(B))                # bit-identical


def test_a_positive_scale_adds_zero_columns_that_read_the_linear_amplitude():
    a, b = _pair(30.0)
    assert b.embed.in_features == 3 * N
    assert torch.equal(b.embed.weight[:, :2 * N], a.embed.weight) and torch.equal(b.embed.bias, a.embed.bias)
    assert (b.embed.weight[:, 2 * N:] == 0).all()
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert all(torch.equal(pa[k], pb[k]) for k in pa if k != "embed.weight")   # drawn as at 0
    B = make_batch(n_slot=N)
    seen = []
    h = b.embed.register_forward_hook(lambda m, i, o: seen.append(i[0]))
    with torch.no_grad():
        torch.testing.assert_close(b.encode(B), a.encode(B), atol=1e-5, rtol=0)   # starts as the old model
    h.remove()
    torch.testing.assert_close(seen[0][:, 2 * N:], torch.sinh(B["inp"]) / 30.0)
    with torch.no_grad():
        b.embed.weight[:, 2 * N:] = 0.1
        assert (b.encode(B) - a.encode(B)).abs().max() > 1e-3     # the channel reaches the encoder


def test_pad_input_columns_copies_the_old_columns_and_zeros_the_new():
    torch.manual_seed(1)
    old = build_fm(dict(SMALL)).eval()
    torch.manual_seed(2)
    new = build_fm(dict(SMALL, lin_amp_scale=30.0)).eval()
    sd = old.state_dict()
    padded = pad_input_columns(new, sd)
    w = padded["embed.weight"]
    assert w.shape == new.embed.weight.shape
    assert torch.equal(w[:, :2 * N], sd["embed.weight"]) and (w[:, 2 * N:] == 0).all()
    assert all(padded[k] is sd[k] for k in sd if k != "embed.weight")             # nothing else touched
    new.load_state_dict(padded, strict=True)
    B = make_batch(n_slot=N)
    with torch.no_grad():
        torch.testing.assert_close(new.encode(B), old.encode(B), atol=1e-5, rtol=0)
    assert pad_input_columns(old, new.state_dict())["embed.weight"].shape[1] == 3 * N   # never shrinks


def test_a_frozen_encoder_refuses_the_channel():
    """The channel's columns start at zero; frozen they would stay zero, a silent no-op."""
    from helix.model.denoise import build_denoise
    with pytest.raises(ValueError, match="freeze_encoder"):
        build_denoise("/nonexistent-run", freeze_encoder=True, lin_amp_scale=30.0)


def test_pad_input_columns_pads_only_the_embedding():
    """Any other widened matrix is a real architecture mismatch: left unpadded, so a strict load
    refuses it instead of loading zeros."""
    torch.manual_seed(1)
    old = build_fm(dict(SMALL)).eval()
    new = build_fm(dict(SMALL, lin_amp_scale=30.0)).eval()
    sd = dict(old.state_dict())
    k = next(k for k, v in sd.items() if v.dim() == 2 and k != "embed.weight")
    sd[k] = sd[k][:, :-1].clone()                       # as if a checkpoint had a narrower layer
    assert pad_input_columns(new, sd)[k].shape == sd[k].shape
