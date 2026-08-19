"""Parsing `train.log` — specifically, surviving a change in the log's columns.

The evaluator prints `sorted(avg.items())`, so adding a metric inserts a column
in the MIDDLE of the line. The parser used one positional regex over the whole
line, so when the grid-free metrics switched on, `charge_bias`/`charge_closure`
landed between `bce` and `loss`, the pattern stopped matching, and every eval
point silently vanished from every plot. Nothing errored — the curves were just
empty, which reads as "the evaluator never ran".
"""
import importlib.util
import os

import numpy as np
import pytest

pytest.importorskip("matplotlib")

_SPEC = importlib.util.spec_from_file_location(
    "_ptp", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scripts", "plot_train_progress.py"))
ptp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ptp)

TRAIN = ("Train: [1/3][{it}/100] Scan 1 Data 0.01 Batch 0.2 Remain 00:01 "
         "loss: 1.5 bce: 0.6 val: 0.9 masked_frac: 0.75 Lr: 3.0e-04")
OLD_EVAL = "   [coeff-eval] batches=145 bce=0.5100 loss=1.4000 masked_frac=0.7500 val=0.8900"
NEW_EVAL = ("   [coeff-eval] batches=145 bce=0.5100 charge_bias=0.9900 "
            "charge_closure=0.9700 loss=1.4000 masked_frac=0.7500 val=0.8900 "
            "var_expl=0.6500")


def _write(tmp_path, lines):
    d = tmp_path / "run"
    d.mkdir()
    (d / "train.log").write_text("\n".join(lines) + "\n")
    return str(d)


def test_parses_the_pre_grid_free_format(tmp_path):
    tr, ev = ptp.parse(_write(tmp_path, [TRAIN.format(it=100), OLD_EVAL]))
    assert len(ev["step"]) == 1
    assert ev["bce"][0] == pytest.approx(0.51)
    assert ev["val"][0] == pytest.approx(0.89)
    assert ev["mask"][0] == pytest.approx(0.75), "masked_frac must map to `mask`"
    assert "batches" not in ev, "a count is not a metric"


def test_extra_columns_do_not_drop_the_eval_point(tmp_path):
    """The regression. `charge_*` sorts between `bce` and `loss`."""
    tr, ev = ptp.parse(_write(tmp_path, [TRAIN.format(it=100), NEW_EVAL]))
    assert len(ev["step"]) == 1, "a new metric must not silence the parser"
    assert ev["bce"][0] == pytest.approx(0.51)
    assert ev["loss"][0] == pytest.approx(1.40)
    assert ev["var_expl"][0] == pytest.approx(0.65)
    assert ev["charge_closure"][0] == pytest.approx(0.97)


def test_a_metric_appearing_mid_run_is_backfilled_with_nan(tmp_path):
    """Columns must stay aligned with `step`, or the curve shifts under itself."""
    tr, ev = ptp.parse(_write(tmp_path, [
        TRAIN.format(it=50), OLD_EVAL, TRAIN.format(it=100), NEW_EVAL]))
    assert len(ev["step"]) == 2
    for k, v in ev.items():
        assert len(v) == 2, f"column {k} has {len(v)} entries for 2 evals"
    assert np.isnan(ev["var_expl"][0]) and ev["var_expl"][1] == pytest.approx(0.65)


def test_eval_before_any_train_line_is_skipped_not_mis_stamped(tmp_path):
    tr, ev = ptp.parse(_write(tmp_path, [NEW_EVAL, TRAIN.format(it=100), NEW_EVAL]))
    assert len(ev["step"]) == 1, "an eval with no preceding step has no step"


def test_a_run_with_no_eval_lines_yields_empty_columns(tmp_path):
    """`ev['bce'][-1]` used to IndexError here; no evaluator is an ordinary state."""
    tr, ev = ptp.parse(_write(tmp_path, [TRAIN.format(it=100)]))
    for k in ("loss", "bce", "val", "mask"):
        assert k in ev and len(ev[k]) == 0


def test_the_step_is_global_and_does_not_collide_across_epochs(tmp_path):
    """pimm's counter is 1-based and `it` reaches `iters` inclusive."""
    tr, _ = ptp.parse(_write(tmp_path, [
        TRAIN.format(it=100),
        TRAIN.format(it=100).replace("[1/3][100/100]", "[2/3][1/100]")]))
    assert tr["step"].tolist() == [100, 101]
