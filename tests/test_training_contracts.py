"""The training step's CONTRACTS: muP scaling, mask reproducibility, and that a
step moves what it should.

`test_training_parity` pinned these bit-for-bit against the research
implementation, and retires with `research/`. What it proved -- "helix does
exactly what research did" -- is the right guarantee for a finished extraction
and the WRONG one going forward: masking, the loss/head and the model width are
all expected to change, and a test that pins the old output fires on every
intentional change until someone deletes it.

So these assert the invariants instead. They are what has to stay true across a
width change, a new masking policy, or a different head -- and they are checkable
without any external tree.

Where a frozen artefact IS the only available statement (the legacy DSP chain,
which is finished and has no analytic form) helix keeps one:
tests/goldens_legacy_dsp.npz. This file is the other case.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm                                   # noqa: E402
from helix.model.mask import make_mask, VIEWS_PER_VOLUME           # noqa: E402

SMALL = dict(n_slot=8, n_band=4, n_plane=6, d=64, blocks=2, dec_blocks=1,
             heads=4, dec_mode="cross")


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

@pytest.mark.parametrize("mode", ["random", "block", "plane", "plane_any"])
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

    The parity tests checked this by diffing every tensor against research. The
    invariant that outlives them: every parameter carrying gradient moves, and
    nothing else does.
    """
    from test_model_fm import make_batch

    model = build_fm(dict(SMALL, mup=True, d_base=32))
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
#
# These three came from test_training_parity.py, which is retired. They never
# had a research dependency: they assert what `plane` vs `plane_any` DO --
# every volume punctured vs one left intact -- which is the property the
# cross-plane result rests on, and it survives a change of masking policy
# because it is stated as structure rather than as a recorded draw.

def test_plane_mask_punctures_every_volume():
    """'plane' hides n_planes views PER VOLUME, leaving no volume intact.

    The point of the mode is to force cross-plane triangulation, and the global
    selection it replaces did not. Measured on a real R1 event (gids 0..5 = two
    volumes x three views), `n_planes=1` masked 16.5% of cells and left the
    punctured volume two of three views in EVERY draw while the other volume
    stayed fully sighted; two views already determine a 3D point, so the model
    could interpolate. Even `n_planes=3` left a volume untouched in ~90% of
    draws.

    This is a deliberate divergence from research/train.py: research picks
    n_planes gids from the whole event, helix picks n_planes per VOLUME. The
    byte-identity parity that recorded that divergence retired with research/;
    the property it protected is asserted here directly.
    """
    from helix.model.mask import make_mask, VIEWS_PER_VOLUME

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


def test_plane_any_leaves_a_volume_intact_and_plane_does_not():
    """The two selections are two different TASKS; pin the difference.

    `plane_any` is the historical/research selection: n_planes from the whole
    event, so with two volumes one of them routinely survives untouched. Two
    views already determine a 3D point, so an intact volume lets the model
    interpolate instead of triangulating -- which is why `plane` exists and why
    `plane_any` is kept only for reproducing the runs that used it.
    """
    from helix.model.mask import make_mask, VIEWS_PER_VOLUME

    B = _batch()
    gid = B["plane_id"]
    vol_of = lambda g: torch.div(g, VIEWS_PER_VOLUME, rounding_mode="floor")
    vols = vol_of(gid.unique()).unique()
    assert len(vols) > 1, "fixture must span >1 volume or this proves nothing"

    intact = {"plane": 0, "plane_any": 0}
    for mode in intact:
        for seed in range(50):
            hit = gid[make_mask(B, mode, 0.5, 1,
                                torch.Generator().manual_seed(seed))].unique()
            if len(vol_of(hit).unique()) < len(vols):
                intact[mode] += 1
    assert intact["plane"] == 0, (
        f"'plane' left a volume fully sighted in {intact['plane']}/50 draws")
    assert intact["plane_any"] > 0, (
        "'plane_any' never left a volume intact -- it is supposed to be the "
        "selection that can, so either the fixture or the mode is wrong")


def test_plane_frac_mixer_honours_plane_mode():
    """`plane_frac` steps must use the selection the run asked for.

    The mixer hardcoded "plane". A run set up to reproduce m113 needs those steps
    to be `plane_any`, and nothing in the loss would show the difference.
    """
    from helix.model import build_fm

    B = _batch()
    gid = B["plane_id"]
    vol_of = lambda g: torch.div(g, 3, rounding_mode="floor")
    vols = vol_of(gid.unique()).unique()

    seen = {}
    for pm in ("plane", "plane_any"):
        model = build_fm(dict(SMALL), plane_frac=1.0, plane_mode=pm, n_planes=1)
        torch.manual_seed(0)
        seen[pm] = sum(len(vol_of(gid[model.make_mask(B)].unique()).unique())
                       < len(vols) for _ in range(50))
    assert seen["plane"] == 0, "plane_mode='plane' still left a volume intact"
    assert seen["plane_any"] > 0, (
        "plane_mode='plane_any' never left one intact — the mixer is ignoring "
        "plane_mode and hardcoding a selection again")
