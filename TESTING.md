# Testing

One image, one command:

    IMG=${HELIX_IMAGE:-/sdf/data/neutrino/omara/images/helix-train.sif}

    cd $HELIX_ROOT
    apptainer exec -B /sdf,/lscratch $IMG /opt/pimm/.venv/bin/python -m pytest -q

**Test counts depend on whether pimm is importable**, and the difference is
large enough to mislead: ~40 tests are `@pimm_importable` and SKIP wherever pimm
is absent.

| environment | helix | pimm-data |
|---|---|---|
| pimm not on the path | 496 passed, 53 skipped | 363 passed, 7 skipped |
| pimm on `PYTHONPATH` | **536 passed, 12 skipped, 1 xfailed** | 363 passed, 7 skipped |

Run with pimm on the path when you want the real number. Seven failures hid in
that gap once -- see the clean-room note in `TESTING.md`.

## Testing a change to pimm-data

**The container INSTALLS pimm-data.** A plain `pytest` in the pimm-data checkout
imports the installed copy from `/opt/pimm/.venv/...`, not your working tree —
so your edits are not what gets tested, and the suite passes for the wrong
reason. This is not hypothetical; it silently green-lit several pimm-data edits
during the consolidation.

Put the working tree ahead of site-packages:

    cd <pimm-data checkout>
    apptainer exec -B /sdf,/lscratch $IMG env PYTHONNOUSERSITE=1 \
      PYTHONPATH=$PWD/src /opt/pimm/.venv/bin/python -m pytest -q

Check which copy you got before trusting a result:

    ... python -c "import pimm_data; print(pimm_data.__file__)"

helix is NOT installed in the image, so helix's own suite always tests the
working tree and needs no such care. The asymmetry is the trap.

The same `PYTHONPATH=<pimm-data>/src` is needed for helix's
`tests/test_cross_repo_duplication.py` to compare against your pimm-data edits
rather than the installed copy.

## Run it in a clean environment

    apptainer exec -B /sdf,/lscratch $IMG \
      env PYTHONNOUSERSITE=1 /opt/pimm/.venv/bin/python -m pytest -q

`PYTHONNOUSERSITE=1` is worth using deliberately. `-B /sdf` remounts home, so
anything pip-installed under `~/.local` is visible inside the container. That is
how the old DSP image appeared to have pimm-data: an editable `.pth` in one
developer's home pointed at their checkout, and test results measured that way
were partly an artifact of whose shell ran them.

If a run passes for you and fails for a colleague, this is the first thing to
check.

## Reading the output

* **A collection ERROR means the environment is wrong**, not the code. pytest
  counts it as a failure and the whole suite stops. The usual cause is a missing
  optional dependency reaching a module-scope import before its `importorskip`.
* **Skips are informative.** `pytest -q -rs` prints the reason for each. Most are
  "needs real production data" or "needs the `pimm` framework", both expected.
* Tests that need real doraemon shards skip when the data is unreachable rather
  than failing, so a green run on a machine without `/sdf/data` mounted is not
  the same claim as a green run with it.

## What the suite guards that is easy to break

`tests/test_boundary.py` enforces that `helix.core` and `helix.tpc` never import
`pimm_data` — the property that lets helix's DSP half run where pimm-data does
not exist. It has three parts because no single check is sufficient (plus a fourth that
keeps the allowed-side list honest); see `docs/ARCHITECTURE.md` §6. If you move code between subpackages and this fails,
the test is right and the move is wrong.

`tests/test_pimm_config_contract.py` checks the training configs' `sys.path` /
`custom_imports` pairing. It catches breakage that would otherwise appear only on
a requeue, hours into a run.

## The dense-chain tests

`tests/test_dense_chain.py` covers the full Densify -> AddNoise -> Digitize
chain, which spans the helix/pimm-data boundary. Three of its tests need a CUDA
device and skip without one; the JAXTPC reconciliation tests skip unless
`JAXTPC_ROOT` (or the S3DF default) is importable.

Those reconciliation tests are the only thing pinning helix's `DEFAULT_ENC` and
its coherent implementation against JAXTPC's own `noise_spectrum.npz` and
`tools/coherent_noise.py`. If they start skipping everywhere, that agreement is
unverified — the forward model would be free to drift from the simulator it is
supposed to reproduce.

## The clean-room run

The suite passing in a development tree is weaker evidence than it looks. Run
it from FRESH CLONES with nothing else reachable:

```bash
CR=$SCRATCH/cleanroom && mkdir -p $CR && cd $CR
git clone <helix> helix && git clone <pimm-data> pimm-data && git clone <pimm> pimm
cd $CR/helix
apptainer exec -B /sdf,/lscratch <image> env PYTHONNOUSERSITE=1 \
  PYTHONPATH=$CR/helix:$CR/pimm-data/src:$CR/pimm \
  /opt/pimm/.venv/bin/python -m pytest tests/ -q
```

Done once (2026-09-17) it found seven failures invisible in the development
tree, every one of them because ~40 tests are `@pimm_importable` and skip
wherever pimm is absent:

* `tests/test_export_artifact.py` imported `from tests.test_artifact_formats`.
  There is no `tests/__init__.py`, so that resolves only when the REPO ROOT
  lands on `sys.path` — true for one invocation style and not another. The rest
  of the suite imports top-level (`from _paths import`); this now does too.
* Five tests in `tests/test_evaluator_ddp_reduce.py` built their evaluator with
  `CoeffFMEvaluator.__new__` and hand-listed its attributes, so the stub drifted
  when the class gained `mask_mode`/`n_planes`. Its fake `make_mask` had drifted
  the same way. Construct with `__init__` and fake only what you must.
* `tests/test_boundary.py` claimed `helix.integrations.pimm.hooks` imports
  `pimm_data`. It reaches it lazily, inside functions, so a module-scope probe
  cannot see it.

It also found `scripts/smoke_train_fm.py` broken in every invocation, and a
`pimm_data.testing` fixture that could not reach the probe stages. Neither was
visible to any test.

**Re-run it before a handover**, and from a clone you have not copied files
into — `coeff_verify` correctly refuses a corpus built from a dirty tree, which
is what a polluted clone gives you.
