"""The training hooks (helix.integrations.pimm.hooks): WeightEMA's buffers and its resume, and
the finite-loss guard."""
import importlib.util

import pytest

torch = pytest.importorskip("torch")


def _pimm_importable():
    """The hooks live in ``helix.integrations.pimm``, whose module body imports pimm's
    registries, so these tests cannot run in an image that has torch but not pimm."""
    try:
        if importlib.util.find_spec("pimm") is None:
            return False
        import pimm.datasets.builder  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pimm_importable(), reason="pimm not importable")

import logging

from helix.model import build_fm  # noqa: E402
from helix.integrations.pimm.hooks import FiniteGuard  # noqa: E402


# ---- WeightEMA's buffers -----------------------------------------------------------------

# WeightEMA must average PARAMETERS and copy BUFFERS.
#
# ``bin_edges`` is a persistent buffer: a constant table of training-set statistics
# that rides in the state_dict so a checkpoint carries its own bin scheme. The EMA
# hook used to walk ``state_dict()`` and average anything floating-point, which
# included it.
#
# ``sh.mul_(d).add_(v, alpha=1-d)`` with ``sh == v`` is the identity in exact
# arithmetic but not in float32: ``v*0.9999`` and ``v*1e-4`` each round, and for a
# generic edge value they do not sum back to ``v``. Measured on a real table the
# error reaches 1.9e-4 within 400 steps and settles at 9.1e-4 — 0.83% of a
# 0.11-wide bin — by ~4,000. That matches what the shipped EMA exports carry
# (0.91% and 1.08% for the two runs) while the raw exports are exact.
#
# Two things make this easy to under-test, and both are pinned below:
#
#   * the +-1e18 open edges do NOT drift (their ulp swamps the increment), and
#     neither do edges at exactly-representable values — a table built from
#     ``linspace(-3, 3, ...)`` sits on fixed points and stays bit-exact under the
#     OLD rule, so it proves nothing. ``_edges`` therefore uses the real band-0
#     span and width.
#   * ``test_parameters_are_still_averaged`` passes under the old rule too, by
#     design: it is a guard that the fix did not turn the EMA into a plain copy,
#     not a regression test for the bug.
#
# Verified to fail against the pre-fix rule via
# ``_diag/drive_ema_regress.py``, which restores it by monkeypatching
# ``_averaged_names`` back to the full state_dict.


ARCH = dict(n_slot=16, n_band=4, n_plane=6, d=64, blocks=2, dec_blocks=1,
            heads=4, mup=True, d_base=32, n_bins=8)
STEPS = 400


def _edges(n_band=4, n_bins=8):
    """A bin table shaped AND VALUED like the real one.

    The span is band 0's from the shipped tables (-6.84699 .. +7.10671) rather
    than a tidy symmetric range: values that are exactly representable in float32
    are fixed points of the old update and stay bit-exact, so a tidy table hides
    the bug entirely. Outer edges are the +-1e18 open sentinels, as
    ``helix.data.bins`` writes them.
    """
    inner = torch.linspace(-6.84699, 7.10671, n_bins - 1)
    row = torch.cat([torch.tensor([-1e18]), inner, torch.tensor([1e18])])
    return row.repeat(n_band, 1)


class _EmaTrainer:
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
    hook.trainer = _EmaTrainer(model, tmp_path)
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
    hook.trainer = _EmaTrainer(model, tmp_path)

    avg = hook._averaged_names()
    params = {n for n, _ in model.named_parameters()}
    buffers = {n for n, _ in model.named_buffers()}
    assert avg == params
    assert "bin_edges" in buffers and "bin_edges" not in avg


# ---- WeightEMA on resume -----------------------------------------------------------------

# The resume path: which saved shadow WeightEMA is allowed to adopt.
#
# Every branch here was previously untested, and the one guard that existed
# (``abs(now - saved) > 1``) rejected nothing reachable. An intact pair drifts by
# EXACTLY 0 -- WeightEMA precedes CheckpointSaver in ``cfg.hooks`` and both key off
# ``trainer.global_step``, already advanced by pimm's ``_record_step_state`` -- so
# ``> 1`` accepted every torn pair it was written to catch.
#
# The replacement threshold is derived, not chosen: after K contaminated steps
# ``1 - decay**K`` of the average sits on wrong updates, while DISCARDING costs
# 100% and needs a half-life to recover. Using therefore beats discarding right up
# to ``ln(0.5)/ln(decay)`` -- 6,931 steps at 0.9999. These tests pin both sides of
# that line, plus the decay-mismatch rejection that ``_save`` recorded the field
# for and nothing read back.


DECAY = 0.9999
HALF_LIFE = 6931          # ln(0.5)/ln(0.9999), floor


class _Logger:
    """Records instead of printing, so a branch can be asserted on its message."""

    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, msg, *a):
        self.infos.append(str(msg))

    def warning(self, msg, *a):
        self.warnings.append(str(msg))


class _ResumeTrainer:
    def __init__(self, model, save_path, step, resume=True):
        self.model = model
        self.global_step = step
        self.cfg = type("cfg", (), {"save_path": str(save_path),
                                    "resume": resume})()
        self.logger = _Logger()


def _tiny():
    """A two-parameter model. Nothing here exercises the ARCHITECTURE -- the
    resume path only ever walks a state_dict -- so a real build_fm would only
    make the test slower and its failures harder to read."""
    m = torch.nn.Linear(3, 2, bias=True)
    with torch.no_grad():
        m.weight.fill_(1.0)
        m.bias.fill_(1.0)
    return m


def _hook(tmp_path, model, *, step, decay=DECAY, **kw):
    from helix.integrations.pimm import WeightEMA
    h = WeightEMA(decay=decay, **kw)
    h.trainer = _ResumeTrainer(model, tmp_path, step)
    return h


def _write_sidecar(tmp_path, model, *, step, decay=DECAY, fill=7.0):
    """Write the sidecar the way ``_save`` does, with a shadow that is visibly
    NOT the live model so adoption can be told from a fresh clone."""
    import os
    path = os.path.join(str(tmp_path), "model", "model_ema.pth")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = {k: torch.full_like(v.float(), fill) for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "decay": decay, "step": step}, path)
    return path


def test_an_intact_pair_drifts_by_zero_and_is_adopted(tmp_path):
    """The common case, and the one the old `> 1` tolerance was aimed at."""
    m = _tiny()
    _write_sidecar(tmp_path, m, step=5000)
    h = _hook(tmp_path, m, step=5000)
    h.before_train()
    assert h._shadow is not None, "an intact sidecar was not adopted"
    assert h._step == 5000
    assert torch.equal(h._shadow["weight"], torch.full((2, 3), 7.0)), \
        "adopted a fresh clone of the live model instead of the saved shadow"
    assert not h.trainer.logger.warnings, \
        f"zero drift must not warn, got {h.trainer.logger.warnings}"


def test_drift_inside_the_half_life_is_used_with_a_warning(tmp_path):
    """Using a contaminated shadow beats discarding it below the half-life."""
    m = _tiny()
    _write_sidecar(tmp_path, m, step=5000)
    h = _hook(tmp_path, m, step=5000 + HALF_LIFE - 1)
    h.before_train()
    assert h._shadow is not None, \
        "drift below the half-life must be USED -- discarding costs more"
    assert h._step == 5000
    assert len(h.trainer.logger.warnings) == 1
    assert "%" in h.trainer.logger.warnings[0], \
        "the warning must quantify the contamination, not just report drift"


def test_drift_past_the_half_life_is_discarded(tmp_path):
    """Past the half-life a fresh average recovers faster than the old one."""
    m = _tiny()
    _write_sidecar(tmp_path, m, step=5000)
    h = _hook(tmp_path, m, step=5000 + HALF_LIFE + 100)
    h.before_train()
    assert h._shadow is None, "drift past the half-life must be discarded"
    assert "DISCARDING" in " ".join(h.trainer.logger.warnings)


def test_on_drift_error_raises_instead_of_discarding(tmp_path):
    """The strict reading, for a caller that wants a torn pair to be fatal."""
    m = _tiny()
    _write_sidecar(tmp_path, m, step=0)
    h = _hook(tmp_path, m, step=HALF_LIFE + 100, on_drift="error")
    with pytest.raises(RuntimeError, match="drift"):
        h.before_train()


def test_max_drift_overrides_the_derived_budget(tmp_path):
    """An explicit budget must win over the half-life in BOTH directions."""
    m = _tiny()
    _write_sidecar(tmp_path, m, step=5000)

    tight = _hook(tmp_path, m, step=5010, max_drift=5)
    tight.before_train()
    assert tight._shadow is None, "max_drift=5 must reject a drift of 10"

    loose = _hook(tmp_path, m, step=5000 + HALF_LIFE + 100, max_drift=10 ** 9)
    loose.before_train()
    assert loose._shadow is not None, \
        "an explicit max_drift must be able to accept past the half-life"


def test_a_sidecar_written_at_another_decay_is_rejected(tmp_path):
    """A shadow cannot change its time constant retroactively: mixing 0.999 and
    0.9999 mass produces an average over no defined window. `_save` has always
    recorded `decay`; nothing read it back."""
    m = _tiny()
    _write_sidecar(tmp_path, m, step=5000, decay=0.999)
    h = _hook(tmp_path, m, step=5000, decay=DECAY)
    h.before_train()
    assert h._shadow is None, "a sidecar at a different decay must not be adopted"
    joined = " ".join(h.trainer.logger.warnings)
    assert "0.999" in joined and "decay" in joined


def test_no_sidecar_restarts_without_raising(tmp_path):
    m = _tiny()
    h = _hook(tmp_path, m, step=5000)
    h.before_train()
    assert h._shadow is None
    assert any("restarts" in s for s in h.trainer.logger.infos)


def test_a_fresh_run_never_reads_the_sidecar(tmp_path):
    """`resume=False` must ignore a stale model_ema.pth left in save_path."""
    m = _tiny()
    _write_sidecar(tmp_path, m, step=5000)
    h = _hook(tmp_path, m, step=0)
    h.trainer.cfg.resume = False
    h.before_train()
    assert h._shadow is None, "a fresh run adopted a leftover EMA sidecar"


# ---- the finite-loss guard ---------------------------------------------------------------

# FiniteGuard: a non-finite gradient skips the step; non-finite weights stop the run.
#
# The first 47k-step d768 run went NaN at ~19.4k steps, finished "COMPLETED", and
# its part 2 and a branch cooldown then trained from NaN weights.

class _GuardTrainer:
    def __init__(self):
        self.model = torch.nn.Linear(4, 3)
        self.comm_info = {"iter": 5, "iter_per_epoch": 10, "epoch": 2}
        self.logger = logging.getLogger("finite-guard-test")


def _guard(**kw):
    g = FiniteGuard(**kw)
    g.trainer = _GuardTrainer()
    return g


def _backward(g):
    g.trainer.model(torch.randn(2, 4)).sum().backward()


def test_a_finite_step_is_left_alone():
    g = _guard()
    _backward(g)
    before = [p.grad.clone() for p in g.trainer.model.parameters()]
    g.after_backward()
    assert all(torch.equal(a, p.grad) for a, p in zip(before, g.trainer.model.parameters()))
    assert g.skipped == 0


def test_a_non_finite_gradient_skips_the_whole_step():
    g = _guard()
    _backward(g)
    g.trainer.model.weight.grad[0, 0] = float("nan")
    g.after_backward()
    assert all(p.grad is None for p in g.trainer.model.parameters())
    assert g.skipped == 1
    # the optimizer then skips every parameter: weights and moments are untouched
    opt = torch.optim.AdamW(g.trainer.model.parameters(), lr=1.0)
    w = g.trainer.model.weight.detach().clone()
    opt.step()
    assert torch.equal(w, g.trainer.model.weight) and not opt.state


def test_consecutive_non_finite_steps_are_a_divergence():
    g = _guard(max_skips=2)
    for i in range(2):
        _backward(g)
        g.trainer.model.bias.grad[0] = float("inf")
        if i == 0:
            g.after_backward()
        else:
            with pytest.raises(RuntimeError, match="consecutive"):
                g.after_backward()


def test_non_finite_weights_stop_the_run_on_the_check_step():
    g = _guard(check_every=26)            # global step 2*10 + 5 = 25 -> check at 26
    g.after_step()                         # finite: nothing happens
    with torch.no_grad():
        g.trainer.model.weight[1, 1] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite weights"):
        g.after_step()
    g.trainer.comm_info["iter"] = 6       # not a check step
    g.after_step()
