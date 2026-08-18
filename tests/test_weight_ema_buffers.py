"""WeightEMA must average PARAMETERS and copy BUFFERS.

``bin_edges`` is a persistent buffer: a constant table of training-set statistics
that rides in the state_dict so a checkpoint carries its own bin scheme. The EMA
hook used to walk ``state_dict()`` and average anything floating-point, which
included it.

``sh.mul_(d).add_(v, alpha=1-d)`` with ``sh == v`` is the identity in exact
arithmetic but not in float32: ``v*0.9999`` and ``v*1e-4`` each round, and for a
generic edge value they do not sum back to ``v``. Measured on a real table the
error reaches 1.9e-4 within 400 steps and settles at 9.1e-4 — 0.83% of a
0.11-wide bin — by ~4,000. That matches what the shipped EMA exports carry
(0.91% and 1.08% for the two runs) while the raw exports are exact.

Two things make this easy to under-test, and both are pinned below:

  * the +-1e18 open edges do NOT drift (their ulp swamps the increment), and
    neither do edges at exactly-representable values — a table built from
    ``linspace(-3, 3, ...)`` sits on fixed points and stays bit-exact under the
    OLD rule, so it proves nothing. ``_edges`` therefore uses the real band-0
    span and width.
  * ``test_parameters_are_still_averaged`` passes under the old rule too, by
    design: it is a guard that the fix did not turn the EMA into a plain copy,
    not a regression test for the bug.

Verified to fail against the pre-fix rule via
``_diag/drive_ema_regress.py``, which restores it by monkeypatching
``_averaged_names`` back to the full state_dict.
"""

import importlib.util

import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm  # noqa: E402


def _pimm_importable():
    """Same guard as ``test_integration_pimm``: the hook lives in
    ``helix.integrations.pimm``, whose module body imports pimm's registries, so
    these tests cannot run in an image that has torch but not pimm."""
    try:
        if importlib.util.find_spec("pimm") is None:
            return False
        import pimm.datasets.builder  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pimm_importable(),
                                reason="pimm not importable")

ARCH = dict(n_slot=16, n_band=4, n_plane=6, d=64, blocks=2, dec_blocks=1,
            heads=4, dec_mode="cross", mup=True, d_base=32, n_bins=8)
STEPS = 400


def _edges(n_band=4, n_bins=8):
    """A bin table shaped AND VALUED like the real one.

    The span is band 0's from the shipped tables (-6.84699 .. +7.10671) rather
    than a tidy symmetric range: values that are exactly representable in float32
    are fixed points of the old update and stay bit-exact, so a tidy table hides
    the bug entirely. Outer edges are the +-1e18 open sentinels, as
    ``derive_coeff_bins`` writes them.
    """
    inner = torch.linspace(-6.84699, 7.10671, n_bins - 1)
    row = torch.cat([torch.tensor([-1e18]), inner, torch.tensor([1e18])])
    return row.repeat(n_band, 1)


class _Trainer:
    """The two attributes WeightEMA reads off the trainer."""

    def __init__(self, model, save_path):
        self.model = model
        self.global_step = 0
        self.cfg = type("cfg", (), {"save_path": str(save_path), "resume": False})()
        self.logger = type("log", (), {"info": lambda *a: None,
                                       "warning": lambda *a: None})()


def _run(tmp_path, steps=STEPS):
    from helix.integrations.pimm import WeightEMA

    torch.manual_seed(0)
    model = build_fm(**ARCH)
    edges = _edges(ARCH["n_band"], ARCH["n_bins"])
    model.set_bins(edges)
    before = model.bin_edges.detach().clone()

    hook = WeightEMA(decay=0.9999, save_freq=None)
    hook.trainer = _Trainer(model, tmp_path)
    hook.before_train()
    for i in range(steps):
        # Perturb the PARAMETERS every step, as an optimizer would. The buffer is
        # left alone — it is a constant, which is the whole point.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 1e-3)
        hook.trainer.global_step = i + 1
        hook.after_step()
    return hook, model, before


def test_bin_edges_buffer_is_not_averaged(tmp_path):
    """The shadow's bin_edges must be EXACTLY the model's, sentinels included."""
    hook, model, before = _run(tmp_path)
    shadow = hook._shadow["bin_edges"]

    # The live buffer never moved: nothing in the hook may write to the model.
    assert torch.equal(model.bin_edges, before), \
        "WeightEMA mutated the model's own bin_edges buffer"
    # And the shadow copied it verbatim rather than averaging toward it.
    assert torch.equal(shadow, before.float()), (
        "bin_edges was averaged, not copied: max |delta| = "
        f"{(shadow - before.float()).abs().max().item():.6g}")


def test_parameters_are_still_averaged(tmp_path):
    """The fix must not turn the EMA into a plain copy of the weights."""
    hook, model, _ = _run(tmp_path)
    name, p = next(iter(model.named_parameters()))
    sh = hook._shadow[name]
    assert not torch.equal(sh, p.detach().float()), \
        f"{name} was copied verbatim — the EMA is not averaging parameters"
    # At decay 0.9999 over 400 steps the shadow still lags the live weights.
    assert (sh - p.detach().float()).abs().max().item() > 0, "shadow did not lag"


def test_shadow_covers_the_whole_state_dict(tmp_path):
    """Every state_dict key must be present, or model_ema.pth stops loading strict."""
    hook, model, _ = _run(tmp_path, steps=3)
    assert set(hook._shadow) == set(model.state_dict()), \
        "shadow key set diverged from the model's state_dict"


def test_averaged_names_are_parameters_only(tmp_path):
    """The selector itself: parameters in, buffers out."""
    from helix.integrations.pimm import WeightEMA

    torch.manual_seed(0)
    model = build_fm(**ARCH)
    model.set_bins(_edges(ARCH["n_band"], ARCH["n_bins"]))
    hook = WeightEMA()
    hook.trainer = _Trainer(model, tmp_path)

    avg = hook._averaged_names()
    params = {n for n, _ in model.named_parameters()}
    buffers = {n for n, _ in model.named_buffers()}
    assert avg == params
    assert "bin_edges" in buffers and "bin_edges" not in avg
