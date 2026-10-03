"""``FlatAdamW`` (helix.integrations.pimm.optim) against torch's AdamW + clip.

Fed IDENTICAL gradients, the fp32 master it keeps must follow what torch.optim.AdamW
does to fp32 weights, per param group (muP: groups differ in lr and weight
decay). Its checkpoint state must round-trip through the API pimm saves with
(torch.distributed.checkpoint.state_dict) without being rounded to bf16.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pimm")

from helix.integrations.pimm.optim import FlatAdamW                 # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _net(dev):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)).to(dev)
    return net.to(torch.bfloat16)


def _groups(net):
    w = [p for p in net.parameters() if p.dim() == 2]
    b = [p for p in net.parameters() if p.dim() == 1]
    return [dict(params=b, lr=2e-3, weight_decay=0.0), dict(params=w, lr=1e-3, weight_decay=0.1)]


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("clip", [None, 0.05])
def test_matches_torch_adamw_with_identical_gradients(dev, clip):
    net = _net(dev)
    ref = [p.detach().float().clone().requires_grad_() for p in net.parameters()]
    rmap = dict(zip(net.parameters(), ref))
    opt = FlatAdamW(_groups(net), betas=(0.9, 0.95), eps=1e-8); opt.clip = clip
    topt = torch.optim.AdamW([dict(params=[rmap[p] for p in g["params"]], lr=g["lr"],
                                   weight_decay=g["weight_decay"]) for g in _groups(net)],
                             betas=(0.9, 0.95), eps=1e-8)
    x = torch.randn(16, 32, device=dev, dtype=torch.bfloat16)
    for _ in range(4):
        net(x).float().square().mean().mul(10).backward()
        for p in net.parameters():
            rmap[p].grad = p.grad.float().clone()
        if clip:
            torch.nn.utils.clip_grad_norm_(ref, clip)
        topt.step(); topt.zero_grad(); opt.step(); opt.zero_grad()
    for p in net.parameters():
        torch.testing.assert_close(opt.state[p]["master"], rmap[p].detach(), rtol=1e-5, atol=1e-6)
        assert torch.equal(p.detach(), opt.state[p]["master"].to(torch.bfloat16))


def test_refuses_fp32_parameters():
    with pytest.raises(TypeError, match="bf16"):
        FlatAdamW(torch.nn.Linear(4, 4).parameters())


def test_skips_a_step_whose_gradients_were_dropped():
    net = _net("cpu"); opt = FlatAdamW(_groups(net))
    before = [p.detach().clone() for p in net.parameters()]
    opt.step()                                     # no grads at all: FiniteGuard dropped them
    assert opt._t == 0
    assert all(torch.equal(a, p) for a, p in zip(before, net.parameters()))


def test_checkpoint_round_trip_keeps_fp32_state():
    from torch.distributed.checkpoint.state_dict import (get_optimizer_state_dict,
                                                         set_optimizer_state_dict)
    net = _net("cpu"); opt = FlatAdamW(_groups(net), betas=(0.9, 0.95))
    x = torch.randn(16, 32, dtype=torch.bfloat16)
    for _ in range(3):
        net(x).float().square().mean().backward(); opt.step(); opt.zero_grad()
    sd = get_optimizer_state_dict(net, opt)
    snap = {k: v.clone() for k, v in opt.state[next(net.parameters())].items()}

    net2 = _net("cpu"); opt2 = FlatAdamW(_groups(net2), betas=(0.9, 0.95))
    set_optimizer_state_dict(net2, opt2, sd)
    st = opt2.state[next(net2.parameters())]
    assert opt2._t == 3
    for k in ("master", "exp_avg", "exp_avg_sq"):
        assert st[k].dtype == torch.float32
        assert torch.equal(st[k], snap[k]), k
    for p, q in zip(net.parameters(), net2.parameters()):
        assert torch.equal(p, q)                    # weights rewritten from the restored master
