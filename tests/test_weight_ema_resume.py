"""The resume path: which saved shadow WeightEMA is allowed to adopt.

Every branch here was previously untested, and the one guard that existed
(``abs(now - saved) > 1``) rejected nothing reachable. An intact pair drifts by
EXACTLY 0 -- WeightEMA precedes CheckpointSaver in ``cfg.hooks`` and both key off
``trainer.global_step``, already advanced by pimm's ``_record_step_state`` -- so
``> 1`` accepted every torn pair it was written to catch.

The replacement threshold is derived, not chosen: after K contaminated steps
``1 - decay**K`` of the average sits on wrong updates, while DISCARDING costs
100% and needs a half-life to recover. Using therefore beats discarding right up
to ``ln(0.5)/ln(decay)`` -- 6,931 steps at 0.9999. These tests pin both sides of
that line, plus the decay-mismatch rejection that ``_save`` recorded the field
for and nothing read back.
"""

import importlib.util

import pytest

torch = pytest.importorskip("torch")


def _pimm_importable():
    try:
        if importlib.util.find_spec("pimm") is None:
            return False
        import pimm.datasets.builder  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pimm_importable(),
                                reason="pimm not importable")

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


class _Trainer:
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
    h.trainer = _Trainer(model, tmp_path, step)
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
