"""The training hooks (helix.integrations.pimm.hooks): the finite-loss guard and the step watchdog."""
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
# the finite guard is imported at module scope below, so without pimm skip before it
pytest.importorskip("pimm")

import logging

from helix.integrations.pimm.hooks import FiniteGuard, StepWatchdog  # noqa: E402


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


# ---- the step watchdog -------------------------------------------------------------------

def test_the_watchdog_rearms_after_every_step_and_disarms_at_the_end(monkeypatch):
    """Armed only from the end of a step (so step 1's compile is not timed), re-armed on every
    step with the configured timeout and exit=True, cancelled when training ends."""
    import faulthandler
    calls = []
    monkeypatch.setattr(faulthandler, "dump_traceback_later",
                        lambda t, exit=False, file=None: calls.append((t, exit)))
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", lambda: calls.append("cancel"))
    w = StepWatchdog(timeout=42)
    assert calls == []                                              # nothing armed before step 1
    w.after_step()
    w.after_step()
    w.after_train()
    assert calls == [(42.0, True), (42.0, True), "cancel"]
    assert StepWatchdog().timeout == 300.0


def test_a_stuck_step_dumps_the_stacks_and_exits():
    """The real hook, in a subprocess: a step that never ends exits with the stuck stack."""
    import subprocess
    import sys
    prog = ("import time\nfrom helix.integrations.pimm.hooks import StepWatchdog\n"
            "StepWatchdog(timeout=0.5).after_step()\n"
            "def stuck_in_a_collective():\n    time.sleep(120)\n"
            "stuck_in_a_collective()\n")
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "stuck_in_a_collective" in r.stderr
