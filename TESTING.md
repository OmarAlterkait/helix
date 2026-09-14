# Testing

One image, one command:

    IMG=${HELIX_IMAGE:-/sdf/data/neutrino/omara/images/helix-train.sif}

    cd $HELIX_ROOT
    apptainer exec -B /sdf,/lscratch $IMG /opt/pimm/.venv/bin/python -m pytest -q

Expect **395 passed, 50 skipped**. In pimm-data, **360 passed, 8 skipped**.

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
