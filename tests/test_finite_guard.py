"""FiniteGuard: a non-finite gradient skips the step; non-finite weights stop the run.

The first 47k-step d768 run went NaN at ~19.4k steps, finished "COMPLETED", and
its part 2 and a branch cooldown then trained from NaN weights.
"""

import logging

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pimm")

from helix.integrations.pimm.hooks import FiniteGuard  # noqa: E402


class _Trainer:
    def __init__(self):
        self.model = torch.nn.Linear(4, 3)
        self.comm_info = {"iter": 5, "iter_per_epoch": 10, "epoch": 2}
        self.logger = logging.getLogger("finite-guard-test")


def _guard(**kw):
    g = FiniteGuard(**kw)
    g.trainer = _Trainer()
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
