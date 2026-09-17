"""Every path that points OUTSIDE the repo, resolved in one place.

helix's configs and scripts referenced ~80 absolute ``/sdf/...`` paths across
nine configs and a dozen scripts. On the machine they were written on, they
work; anywhere else they are silent breakage -- a config resolves, a job
launches, and it reads a corpus that is not there.

Every external root is an env var with an S3DF default, so:

* on S3DF nothing changes -- the defaults ARE the current paths;
* elsewhere, exporting the vars is the entire port;
* and ``python -m helix.paths`` prints what resolved and what is missing, which
  is the first thing to run on a new machine.

The names follow the ones ``tests/_paths.py`` already established
(``HELIX_CORPUS``, ``HELIX_SENSOR_ROOT``, ``HELIX_PIMM_ROOT``), so there is one
vocabulary rather than two.

    ============================  ==========================================
    variable                      what it points at
    ============================  ==========================================
    HELIX_CORPUS_ROOT             parent of the built coeff corpora
    HELIX_CORPUS                  ONE corpus: a run dir under the root.
                                  r1 (tau=0.05), not the pre-tau vintage --
                                  the gate rule the builder still produces.
    HELIX_SENSOR_ROOT             simulator output the corpus is built FROM
    HELIX_ARCHIVE                 bin tables, converted checkpoints
    HELIX_EXP                     where runs write (save_path, logs, exports)
    HELIX_SCRATCH                 node-local/temporary space
    HELIX_PIMM_ROOT               a pimm checkout (the trainer)
    HELIX_PIMM_DATA_SRC           a pimm-data checkout's src/
    HELIX_IMAGE                   the container: DSP and training both
    HELIX_JAXTPC_ROOT             JAXTPC checkout (noise spectrum, geometry)
    HELIX_LEGACY_CORPUS           the pre-tau corpus, for m113 only
    HELIX_OPTICAL_DATA            the goop PMT light file
    HELIX_ROOT                    this checkout (DERIVED; override only to
                                  pin a resumed job to link 1's tree)
    ============================  ==========================================
"""
from __future__ import annotations

import os
from pathlib import Path

#: (env var, S3DF default, one-line description). The defaults are where these
#: live on the machine helix was developed on; they are defaults, not truths.
_ROOTS = (
    ("HELIX_CORPUS_ROOT",   "/sdf/data/neutrino/omara",                 "parent of built coeff corpora"),
    ("HELIX_CORPUS",        "/sdf/data/neutrino/omara/coeff_tpc_r1/run_0027575715",
                                                                        "ONE corpus (a run dir)"),
    ("HELIX_SENSOR_ROOT",   "/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor",
                                                                        "simulator sensor shards"),
    ("HELIX_ARCHIVE",       "/sdf/data/neutrino/omara/archive",         "bin tables, converted ckpts"),
    ("HELIX_EXP",           "/sdf/data/neutrino/omara/exp/helix",       "run outputs"),
    ("HELIX_SCRATCH",       os.environ.get("SCRATCH", "/tmp"),          "temporary space"),
    ("HELIX_PIMM_ROOT",     "/sdf/group/neutrino/omara/pimm-fm",        "pimm checkout"),
    ("HELIX_PIMM_DATA_SRC", "/sdf/group/neutrino/omara/pimm-data/src",  "pimm-data src/"),
    # ONE image now. It was two -- a DSP container with pywt but no pimm-data,
    # and a pimm container with pimm-data but no pywt -- and both were broken:
    # the pimm one baked pimm_data 0.3.0, which still registers the forward model
    # helix registers now, and the DSP one had no pimm-data at all (the corpus
    # builder needs it) and only ever worked through an editable .pth in one
    # developer's home. Built by container/helix-train.def; see
    # docs/ARCHITECTURE.md section 4.
    ("HELIX_IMAGE",         "/sdf/data/neutrino/omara/images/helix-train.sif",
                                                                        "the container"),
    ("HELIX_JAXTPC_ROOT",   "/sdf/group/neutrino/omara/JAXTPC",         "JAXTPC checkout"),
    ("HELIX_LEGACY_CORPUS",  "/sdf/data/neutrino/omara/coeff_tpc/run_0027575715",
                                                                        "pre-tau corpus (m113 only)"),
    ("HELIX_OPTICAL_DATA",  "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5",
                                                                        "goop PMT light file"),
)


def repo() -> Path:
    """This checkout, located from THIS file -- never from a name.

    Three helix checkouts have existed side by side (`helix`, `helix-consolidate`,
    a worktree), and hardcoding one is not a portability problem but a
    CORRECTNESS one: `scripts/build_coeff_corpus.py` records that an old default
    of `helix-consolidate` was inserted at ``sys.path[0]``, so running the builder
    from any other checkout silently used that tree's code -- "32 commits behind,
    missing the MAD median fix, the packaged noise spectrum, the m113 anchoring
    and the whole pimm integration" -- and produced corpora with stale DSP and no
    error.

    ``HELIX_ROOT`` still overrides, for the resume case where a job must re-enter
    the checkout link 1 used rather than wherever the code happens to be read
    from. But the DEFAULT is derived, so it cannot name a directory that has been
    renamed or that never existed on this machine.
    """
    return Path(os.environ.get("HELIX_ROOT", Path(__file__).resolve().parent.parent))


def root(name: str) -> Path:
    """Resolve one external root. Raises on an unknown NAME rather than
    inventing a path, because a typo that silently returns something is how a
    job reads the wrong corpus."""
    for var, default, _ in _ROOTS:
        if var == name:
            return Path(os.environ.get(var, default))
    known = ", ".join(v for v, _, _ in _ROOTS)
    raise KeyError(f"unknown helix root {name!r}; known roots: {known}")


def corpus(run: str | None = None) -> Path:
    """The corpus, or one run directory inside it."""
    base = root("HELIX_CORPUS")
    return base / run if run else base


def archive(*parts: str) -> Path:
    """A file under the archive (bin tables, converted checkpoints)."""
    return root("HELIX_ARCHIVE").joinpath(*parts)


def exp(*parts: str) -> Path:
    """A run output directory."""
    return root("HELIX_EXP").joinpath(*parts)


def packaged(name: str) -> Path:
    """A data file helix SHIPS, under ``helix/tpc/data/``.

    The DSP's external inputs used to be resolved only from a JAXTPC checkout,
    by an absolute path baked into the corpus builder. helix packages its own
    copy of the noise spectrum (see pyproject package-data) precisely so that a
    clone can build a corpus with no second repository present -- this is how a
    caller reaches it without knowing where the package landed.
    """
    return Path(__file__).resolve().parent / "tpc" / "data" / name


def pythonpath() -> str:
    """The PYTHONPATH a pimm run needs: pimm, helix, pimm-data src."""
    return os.pathsep.join((
        str(root("HELIX_PIMM_ROOT")),
        str(Path(__file__).resolve().parent.parent),
        str(root("HELIX_PIMM_DATA_SRC")),
    ))


def report() -> int:
    """Print every root, its source, and whether it exists. Exit non-zero if
    anything required is missing -- the first command to run on a new machine."""
    missing = 0
    width = max(len(v) for v, _, _ in _ROOTS)
    for var, default, desc in _ROOTS:
        val = os.environ.get(var)
        src = "env" if val is not None else "default"
        p = Path(val if val is not None else default)
        ok = p.exists()
        missing += not ok
        print(f"{var:<{width}}  {'ok ' if ok else 'MISSING'}  ({src:7s}) {p}"
              f"\n{'':<{width}}  {desc}")
    if missing:
        print(f"\n{missing} root(s) missing. Set the env vars above, or see "
              f"docs/RUNBOOK.md for how to obtain each.")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(report())
