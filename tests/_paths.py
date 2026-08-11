"""External paths the suite validates against, in one place and overridable.

Eight absolute ``/sdf`` paths were hardcoded across seven test files with no
override. Two consequences, both quiet:

* Three of them point at ``.../omara/helix/research/...`` — a DIFFERENT checkout
  from this repo's own ``research/``. So deleting ``research/`` here would not
  retire those tests; they would keep validating against an unversioned external
  tree, and keep passing.
* On any machine without these paths the tests skip, and a skip for "the data is
  not here" looks exactly like a skip for "this machine has no GPU". Nothing
  distinguishes a suite that checked nothing from one that checked everything.

pimm-data solved this with ``*_DATA_ROOT`` env vars plus a collection-time skip;
this mirrors it. Defaults preserve today's behaviour exactly, so nothing changes
unless an env var is set.

    HELIX_RESEARCH_ROOT   the research tree the parity tests diff against
    HELIX_PIMM_ROOT       the pimm checkout the config/step contracts import
    HELIX_CORPUS          a built coeff corpus
    HELIX_SENSOR_ROOT     root of the real sensor shards
"""

import os

#: Research tree (``coeff_foundation_model``). The FM tests use the ``fm``
#: subdirectory; ``test_legacy_parity`` uses the parent.
RESEARCH_ROOT = os.environ.get(
    "HELIX_RESEARCH_ROOT",
    "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model")
RESEARCH_FM = os.path.join(RESEARCH_ROOT, "fm")

#: pimm checkout providing the trainer/registry the contract tests import.
PIMM_ROOT = os.environ.get("HELIX_PIMM_ROOT", "/sdf/group/neutrino/omara/pimm-fm")

#: A built coefficient corpus (noisy + clean shard families).
CORPUS = os.environ.get("HELIX_CORPUS",
                        "/sdf/data/neutrino/omara/coeff_tpc/run_0027575715")

#: Root of the real sensor shards. Each test picks its own run/shard beneath it
#: (they deliberately use different runs), so this is a ROOT, not one file.
SENSOR_ROOT = os.environ.get(
    "HELIX_SENSOR_ROOT",
    "/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor")


def sensor_shard(run, name):
    """One shard beneath :data:`SENSOR_ROOT`."""
    return os.path.join(SENSOR_ROOT, run, name)


def describe():
    """``{name: (path, exists)}`` — so a skip can say WHICH input was missing."""
    return {n: (p, os.path.exists(p)) for n, p in (
        ("RESEARCH_ROOT", RESEARCH_ROOT), ("RESEARCH_FM", RESEARCH_FM),
        ("PIMM_ROOT", PIMM_ROOT), ("CORPUS", CORPUS),
        ("SENSOR_ROOT", SENSOR_ROOT))}
