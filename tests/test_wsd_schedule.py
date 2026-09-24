"""WSDCooldownLR: pure cooldown at stable_frac=0, full warmup-stable-decay above."""
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pimm")
from helix.integrations.pimm.trainer import WSDCooldownLR  # noqa: E402


def _lrs(steps, **kw):
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sch = WSDCooldownLR(opt, **kw)
    out = []
    for _ in range(steps + 1):
        out.append(opt.param_groups[0]["lr"])
        opt.step(); sch.step()
    return out


def test_stable_frac_zero_is_the_plain_cooldown():
    lr = _lrs(100, total_steps=100, warmup=0)
    assert lr[0] == 1.0 and lr[25] == pytest.approx(0.5)
    assert lr[100] == pytest.approx(1e-3)


def test_warmup_then_flat_then_one_minus_sqrt():
    lr = _lrs(110, total_steps=110, warmup=10, stable_frac=0.5)
    assert lr[5] == pytest.approx(0.5)                       # warmup
    assert all(v == 1.0 for v in lr[10:61])                  # flat to 10 + 50
    assert lr[85] == pytest.approx(1 - math.sqrt(0.5))       # halfway down
    assert lr[110] == pytest.approx(1e-3)                    # floor
