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

    HELIX_PIMM_ROOT       the pimm checkout the config/step contracts import
    HELIX_CORPUS          a built coeff corpus
    HELIX_SENSOR_ROOT     root of the real sensor shards
"""

import os

from helix import paths as _hpaths

#: pimm checkout providing the trainer/registry the contract tests import.
PIMM_ROOT = os.environ.get("HELIX_PIMM_ROOT", "/sdf/group/neutrino/omara/pimm-fm")

#: A built coefficient corpus (noisy + clean shard families).
#
#: Resolved by :mod:`helix.paths`, not redeclared here. This used to carry its
#: OWN default -- and a different one: the pre-tau `coeff_tpc` vintage, while
#: helix.paths said `coeff_tpc_r1`. One env var, two values, and two KINDS of
#: value (a run dir here, a corpus root there). Both corpora are real, both read
#: cleanly, and they differ in the coherent-removal gate (r1 records tau=0.05),
#: so the wrong one is a plausible wrong answer rather than a crash.
CORPUS = str(_hpaths.root("HELIX_CORPUS"))

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
        ("PIMM_ROOT", PIMM_ROOT), ("CORPUS", CORPUS),
        ("SENSOR_ROOT", SENSOR_ROOT))}
