# Running the tests

There is no single environment that runs everything. Two containers each hold
half of what the suite needs, and the halves do not overlap:

| container | has | lacks |
|---|---|---|
| `/sdf/group/neutrino/images/develop.sif` | `pywt`, torch, the DSP stack | pimm and its deps |
| `/sdf/data/neutrino/youngsam/images/pimm-latest.sif` | pimm's full dependency set | `pywt` |

A plain `pytest` in either one is **green and misleading**. In `develop.sif` the
seven pimm-facing modules skip themselves silently — that hid forty tests,
including every test of the pimm-facing evaluator, launcher and WeightEMA code,
which had never run once. In `pimm-latest.sif` collection dies on `pywt`.

## The two runs, together, are the check

**1. DSP half** — the default suite, no pimm:

```bash
H=/sdf/group/neutrino/omara/helix-extraction
apptainer exec -B /sdf,/lscratch /sdf/group/neutrino/images/develop.sif \
  env PYTHONPATH=$H python3 -m pytest -q
# expect: 287 passed, ~49 skipped (40 of those are the pimm seam — see below)
```

**2. pimm seam** — needs a pimm CHECKOUT on `PYTHONPATH`; pimm is not installed
in either container, so pointing at the venv interpreter alone is not enough:

```bash
H=/sdf/group/neutrino/omara/helix-extraction
P=<a pimm checkout>          # e.g. /sdf/group/neutrino/omara/pimm-evalcontract
PYX=/sdf/data/neutrino/omara/exp/_diag/pyx     # staged pytest
apptainer exec -B /sdf,/lscratch /sdf/data/neutrino/youngsam/images/pimm-latest.sif \
  env PYTHONPATH=$PYX:$H:$P HELIX_REQUIRE_PIMM=1 \
  /opt/pimm/.venv/bin/python -m pytest -q --ignore=tests/test_optical.py
```

`HELIX_REQUIRE_PIMM=1` makes the run **abort** if pimm is not importable,
instead of skipping the seam and reporting success. Always set it for a run
whose result is meant to mean "the integration is good".

`--ignore=tests/test_optical.py` and the ~50 `No module named 'pywt'` failures
in that container are the missing DSP half, covered by run 1.

## Known failures, not yours

- `test_integration_pimm.py::test_dataset_wrapper_forwards_every_inner_parameter`
  fails against both the current pimm branch and pimm's `origin/main`.
- `test_model_fm.py::test_matches_frozen_golden` fails in `pimm-latest.sif`
  (a `pywt`-dependent golden).

## GPU

A handful of tests need a CUDA device and skip without one. For those:

```bash
sbatch --partition=ampere --account=neutrino:ml-dev --gpus=1 \
       --cpus-per-task=8 --mem=48G --time=01:00:00 <script>
```
