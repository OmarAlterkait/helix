"""muP's per-group LR must survive the scheduler.

`FMModel.param_groups` encodes muP as a per-group `lr`: hidden weights get
`base_lr / m`. That encoding has exactly one fragile point — a scheduler that
assigns a single LR to every group discards it, and the failure is invisible.
The loss still falls, just to a worse place, and only a run at another width
would reveal it.

The WSD schedules are `LambdaLR`s, which scale each group's own `lr`; driving
a real LambdaLR over muP's groups covers the behaviour that matters without
pimm installed. `helix.model.mup.param_group_ratios` reads the ratios back.
"""

import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm  # noqa: E402
from helix.model.mup import param_group_ratios  # noqa: E402

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
