"""External paths the suite validates against, in one place and overridable.

Hardcoded absolute paths spread over test files are quiet in two ways: a test
can keep validating against an unversioned external tree without anyone
noticing, and on a machine without the path it skips, and a skip for "the data is
not here" looks exactly like a skip for "this machine has no GPU". Nothing
distinguishes a suite that checked nothing from one that checked everything.

pimm-data solved this with ``*_DATA_ROOT`` env vars plus a collection-time skip;
this mirrors it. Defaults preserve today's behaviour exactly, so nothing changes
unless an env var is set.

    HELIX_PIMM_ROOT       the pimm checkout the config/step contracts import
    HELIX_CORPUS          a built coeff corpus
    HELIX_SIM_ROOT        the JAXTPC wire campaign (<campaign>/data/jaxtpc_wire)
"""

import os

from helix import paths as _hpaths

def _opt(name):
    """A root's value, or "" when this site does not configure it.

    ALWAYS via helix.paths, never a second default. Every entry here used to
    carry its own copy of an S3DF literal, and one of them had already drifted
    (see CORPUS below). "" rather than None so the `os.path.isdir(...)` guards
    that gate every real-data test keep working unchanged -- isdir("") is False.

    `resolve`, not `root`: root() RAISES on an unconfigured root, and this module
    is imported at COLLECTION time. A raise here would turn "this machine has no
    corpus" into an error that hides the entire suite, which is the opposite of
    what these guards are for.
    """
    value, _src = _hpaths.resolve(name)
    return str(value) if value is not None else ""


#: pimm checkout providing the trainer/registry the contract tests import.
PIMM_ROOT = _opt("HELIX_PIMM_ROOT")

#: A built coefficient corpus (noisy + clean shard families).
#
#: Resolved by :mod:`helix.paths`, not redeclared here. This used to carry its
#: OWN default -- and a different one: the pre-tau `coeff_tpc` vintage, while
#: helix.paths said `coeff_tpc_r1`. One env var, two values, and two KINDS of
#: value (a run dir here, a corpus root there). Both corpora are real, both read
#: cleanly, and they differ in the coherent-removal gate (r1 records tau=0.05),
#: so the wrong one is a plausible wrong answer rather than a crash.
CORPUS = _opt("HELIX_CORPUS")

#: The JAXTPC wire campaign the corpus is built from. Each test picks its own task beneath
#: it, so this is a ROOT, not one file.
SIM_ROOT = _opt("HELIX_SIM_ROOT")

#: The task the real-data tests read: it holds an event with an empty volume (entry 202).
SIM_GROUP, SIM_TASK = "j000000", "jaxtpc_wire_j000085-000089"


def sim_file(kind="sensor", group=SIM_GROUP, task=SIM_TASK):
    """One task's ``kind`` (sensor / step / hits) file beneath :data:`SIM_ROOT`."""
    return os.path.join(SIM_ROOT, group, task, kind, f"sim_wire_{kind}_{task}.h5")


def describe():
    """``{name: (path, exists)}`` — so a skip can say WHICH input was missing."""
    return {n: (p, os.path.exists(p)) for n, p in (
        ("PIMM_ROOT", PIMM_ROOT), ("CORPUS", CORPUS),
        ("SIM_ROOT", SIM_ROOT))}
