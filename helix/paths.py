"""Every path that points OUTSIDE the repo, resolved in one place.

helix's configs and scripts referenced ~80 absolute ``/sdf/...`` paths across
nine configs and a dozen scripts. On the machine they were written on, they
work; anywhere else they are silent breakage -- a config resolves, a job
launches, and it reads a corpus that is not there.

The first fix was to make every external root an env var *with an S3DF default*.
That was better, but the defaults were still S3DF literals living in this file,
so the module docstring could honestly say "on S3DF nothing changes -- the
defaults ARE the current paths". At a second site eleven of twelve roots report
MISSING and there is nowhere to put the real values except a shell profile that
no test can see.

So the values moved OUT of the code and into **site profiles**:
``helix/sites/<name>.yaml``. This module is now a resolver, and contains no
site-specific path at all. Resolution order, highest first:

1. an explicit ``HELIX_<NAME>`` environment variable -- always wins, so a
   one-off override needs no file edit;
2. the selected site profile;
3. nothing. There is deliberately no fallback default: a path that resolves to
   another site's layout is worse than one that does not resolve, because the
   first fails late and the second fails at ``python -m helix.paths``.

The site is chosen by ``HELIX_SITE``, or auto-detected from each profile's
``detect:`` block (a ``path`` that exists, or an ``env`` var that is set). A
profile may set a root to ``null`` to say "this site does not have that thing",
which reports as ``not set`` rather than ``MISSING`` -- absence and breakage
look different because they are different.

``python -m helix.paths`` prints the site, every root, where its value came
from, and whether it exists. It is the first thing to run on a new machine.

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
    HELIX_JAXTPC_ROOT             JAXTPC checkout (noise spectrum, geometry)
    HELIX_LEGACY_CORPUS           the pre-tau corpus, for m113 only
    HELIX_OPTICAL_DATA            the goop PMT light file
    HELIX_ROOT                    this checkout (DERIVED; override only to
                                  pin a resumed job to link 1's tree)
    ============================  ==========================================

Two things that are NOT roots, because they are not filesystem paths, and whose
absence from the old table is why they ended up hardcoded in shell scripts:

* the **container** -- a runtime, an image, an in-image interpreter and a bind
  list. ``HELIX_IMAGE`` used to be a single root typed as a path to a ``.sif``,
  which cannot describe a site whose image is a registry reference. See
  :func:`container`.
* the **scheduler** -- account, partition, qos, constraint, per job KIND,
  because at S3DF the corpus build and the training run need different
  accounts. See :func:`scheduler`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as _exc:  # pragma: no cover - dependency is declared
    raise ModuleNotFoundError(
        "helix.paths needs PyYAML to read helix/sites/*.yaml. It is a declared "
        "base dependency (pyproject.toml); reinstall with `pip install -e .`."
    ) from _exc

#: Where the site profiles live. A plain data directory, NOT a subpackage: this
#: module is imported by everything, including the DSP-only install, so it must
#: not pull in any helix subpackage to find its own configuration.
SITES_DIR = Path(__file__).resolve().parent / "sites"

#: The vocabulary. Name -> one-line meaning. This is deliberately NOT in the
#: site files: what a root MEANS is a property of helix, not of a machine, and
#: duplicating it per site is how two profiles come to disagree about what
#: HELIX_CORPUS is. Only the VALUES are site-specific.
ROOTS: dict[str, str] = {
    "HELIX_CORPUS_ROOT":   "parent of built coeff corpora",
    "HELIX_CORPUS":        "ONE corpus (a run dir)",
    "HELIX_SENSOR_ROOT":   "simulator sensor shards",
    "HELIX_ARCHIVE":       "bin tables, converted ckpts",
    "HELIX_EXP":           "run outputs",
    "HELIX_SCRATCH":       "temporary space",
    "HELIX_PIMM_ROOT":     "pimm checkout",
    "HELIX_PIMM_DATA_SRC": "pimm-data src/",
    "HELIX_JAXTPC_ROOT":   "JAXTPC checkout",
    "HELIX_LEGACY_CORPUS": "pre-tau corpus (m113 only)",
    "HELIX_OPTICAL_DATA":  "goop PMT light file",
}

#: Job kinds :func:`scheduler` knows. They differ at S3DF (the corpus build runs
#: on turing under a different account than training on ampere) so the split is
#: not cosmetic.
JOB_KINDS = ("train", "corpus")

_SITE_CACHE: dict[str, Any] | None = None


class SiteError(RuntimeError):
    """A site profile is missing, unreadable, or does not declare what was asked.

    Raised rather than returning a default, because every default this module
    used to carry was an S3DF path, and inheriting one at another site is
    exactly the silent breakage site profiles exist to end.
    """


def available_sites() -> list[str]:
    """Profile names that ship with this checkout, sorted."""
    return sorted(p.stem for p in SITES_DIR.glob("*.yaml"))


def _read_profile(name: str) -> dict[str, Any]:
    path = SITES_DIR / f"{name}.yaml"
    if not path.exists():
        raise SiteError(
            f"unknown site {name!r}: no {path}. Known sites: "
            f"{', '.join(available_sites()) or '(none installed)'}. "
            f"A wheel built without package-data has no sites/ at all."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SiteError(f"site profile is not a mapping: {path}")
    return data


def _detect() -> str | None:
    """Pick a site from its ``detect:`` block, or None.

    Detection is a convenience so that an unconfigured shell on a known machine
    still works; ``HELIX_SITE`` is the explicit form and always wins. Ambiguity
    is an error rather than a coin flip -- two sites matching means the rules
    are wrong, and picking one silently would send a job to the wrong storage.
    """
    hits = []
    for name in available_sites():
        rule = _read_profile(name).get("detect") or {}
        if "env" in rule and os.environ.get(str(rule["env"])):
            hits.append(name)
        elif "path" in rule and Path(str(rule["path"])).exists():
            hits.append(name)
    if len(hits) > 1:
        raise SiteError(
            f"site detection is ambiguous: {', '.join(hits)} all match. "
            f"Set HELIX_SITE to one of them."
        )
    return hits[0] if hits else None


def site_name() -> str | None:
    """The selected site, or None if nothing is configured or detected."""
    explicit = os.environ.get("HELIX_SITE")
    if explicit:
        return explicit
    return _detect()


def site() -> dict[str, Any]:
    """The selected site profile, loaded once.

    Returns an empty mapping when no site is selected, so that a machine with
    no profile still resolves whatever the environment provides. That is the
    case a fresh clone is in before ``HELIX_SITE`` is set, and it must produce a
    useful report rather than an exception.
    """
    global _SITE_CACHE
    if _SITE_CACHE is None:
        name = site_name()
        _SITE_CACHE = _read_profile(name) if name else {}
    return _SITE_CACHE


def _reset_cache() -> None:
    """Forget the loaded profile. For tests that change HELIX_SITE in-process."""
    global _SITE_CACHE
    _SITE_CACHE = None


def _expand(value: str) -> str:
    """Expand ``$VAR`` against the environment, strictly.

    ``os.path.expandvars`` leaves an unset variable in place, so ``$CFS/x``
    silently becomes the literal path ``$CFS/x`` -- which then reports MISSING
    for a reason that has nothing to do with the profile being wrong. Site
    values legitimately reference ``$CFS``, ``$SCRATCH`` and ``$HOME``, so the
    expansion has to happen; catching the unset case is what makes it safe.
    """
    out = os.path.expandvars(value)
    if "$" in out:
        missing = [w for w in out.replace("/", " ").split() if w.startswith("$")]
        raise SiteError(
            f"site profile value {value!r} references {', '.join(missing)}, "
            f"which is not set in this environment."
        )
    return out


def resolve(name: str) -> tuple[Path | None, str]:
    """Resolve one root to ``(path_or_None, source)`` without raising on absence.

    ``source`` is one of ``env``, ``site:<name>``, ``site:<name> (null)`` or
    ``unset``. :func:`report` needs the provenance, and a caller that wants to
    degrade gracefully needs the None; :func:`root` is the strict wrapper.
    """
    if name not in ROOTS:
        raise KeyError(
            f"unknown helix root {name!r}; known roots: {', '.join(ROOTS)}"
        )
    from_env = os.environ.get(name)
    if from_env:
        return Path(_expand(from_env)), "env"

    profile = site()
    roots = profile.get("roots") or {}
    if name in roots:
        raw = roots[name]
        if raw is None:
            return None, f"site:{profile.get('name', '?')} (null)"
        return Path(_expand(str(raw))), f"site:{profile.get('name', '?')}"
    return None, "unset"


def root(name: str) -> Path:
    """Resolve one external root, or fail with a message that says what to do.

    Raises on an unknown NAME rather than inventing a path, because a typo that
    silently returns something is how a job reads the wrong corpus. Raises on an
    UNCONFIGURED name for the same reason at one remove: the old behaviour was
    to fall back to an S3DF literal, which at any other site is a path that
    cannot exist and whose failure surfaces much later, as a FileNotFoundError
    from deep inside a reader.
    """
    value, source = resolve(name)
    if value is None:
        where = site_name() or "no site selected"
        raise SiteError(
            f"{name} is not configured ({source}; site: {where}).\n"
            f"  {ROOTS[name]}\n"
            f"  Set {name} in the environment, or add it to "
            f"helix/sites/<site>.yaml, or set HELIX_SITE.\n"
            f"  Run `python -m helix.paths` to see every root at once."
        )
    return value


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

    This is the one root that is NOT in a site profile, for that reason: it is a
    property of where this file is, not of the machine.
    """
    return Path(os.environ.get("HELIX_ROOT", Path(__file__).resolve().parent.parent))


def container(key: str | None = None) -> Any:
    """The container facts for this site: runtime, image, python, binds.

    Why this is not a root. ``HELIX_IMAGE`` used to be a single entry in the
    roots table typed as a filesystem path to a ``.sif``, and ``report()``
    stat()ed it. At NERSC the image is a REGISTRY REFERENCE
    (``ghcr.io/...@sha256:...``) consumed by shifter or podman-hpc, which is not
    a path and never exists on disk -- so the one root reported MISSING while the
    image was perfectly fine, and the two submit scripts that fed it to
    ``apptainer exec`` had nowhere to learn that apptainer is not installed.

    Keys: ``runtime``, ``image``, ``python``, ``binds``, and optionally
    ``interactive_runtime`` / ``interactive_flags`` where a site's batch and
    shell runtimes differ. Each is overridable by ``HELIX_CONTAINER_<KEY>``.
    """
    spec = dict(site().get("container") or {})
    for k in list(spec) + ["runtime", "image", "python"]:
        override = os.environ.get(f"HELIX_CONTAINER_{k.upper()}")
        if override:
            spec[k] = override
    if key is None:
        return spec
    if key not in spec:
        raise SiteError(
            f"container.{key} is not declared for site {site_name() or '(none)'}. "
            f"Declared: {', '.join(sorted(spec)) or '(nothing)'}."
        )
    return spec[key]


def world_size(default: int = 4) -> int:
    """The number of ranks this process is part of, from ``WORLD_SIZE``.

    An environment fact, which is why it lives here rather than in a config.

    It is load-bearing for the coefficient FM specifically: the model takes
    exactly ONE event per rank (it has no event separation, so two events in a
    forward would attend across each other -- MULTI_EVENT_BATCHING.md), so the
    global batch is *defined* as the rank count and
    ``helix.integrations.pimm.trainer`` raises when they disagree.

    Hardcoding it made every config correct at exactly one GPU count: ``4`` was
    right on four ranks and fatal on one and on sixteen, and changing the count
    meant editing three files. A batch-size/learning-rate scaling study is a
    sweep over rank counts, so that hardcoding is the thing that makes the study
    an editing exercise.

    torchrun -- which pimm's ``scripts/train.sh`` uses -- exports ``WORLD_SIZE``
    before the config is read, so inside a job this is the true count. The
    ``default`` applies only where it is absent: ``pimm submit`` preflight on a
    login node, and a bare single-process config load. Getting it wrong there is
    safe, because the trainer's guard fires with a named cause rather than
    training something subtly different.
    """
    raw = os.environ.get("WORLD_SIZE")
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        raise SiteError(f"WORLD_SIZE={raw!r} is not an integer") from None
    if n < 1:
        raise SiteError(f"WORLD_SIZE={n} is not a positive rank count")
    return n


def site_env() -> dict[str, str]:
    """Environment variables this site needs set, from its ``env:`` block.

    Not paths and not scheduler flags: facts about the machine that some library
    reads directly. The motivating one is ``HDF5_USE_FILE_LOCKING=FALSE`` --
    GPFS does not support the POSIX locks HDF5 takes by default, so at NERSC
    every ``h5py.File()`` on the corpus raises ``OSError: [Errno 524] ... unable
    to lock file`` until it is set.

    That is worth a first-class slot rather than a note in a runbook, because it
    is invisible until something opens a real shard, and the tests that open real
    shards had been SKIPPING (their paths did not resolve before site profiles
    existed). Fixing the paths is what surfaced it.

    An existing value in the environment WINS: this is a default for the site,
    not an override of a deliberate choice.
    """
    out: dict[str, str] = {}
    for key, value in (site().get("env") or {}).items():
        if value is None:
            continue
        out[str(key)] = os.environ.get(str(key), str(value))
    return out


def apply_site_env() -> dict[str, str]:
    """Set :func:`site_env` into ``os.environ`` and return what was applied.

    For entry points that run helix code in-process rather than through
    ``scripts/helix_env.sh`` -- a test session, a notebook, a bare
    ``python -m helix.data.coeff_verify``. Idempotent, and never overwrites a
    variable the caller set.
    """
    applied = site_env()
    for key, value in applied.items():
        os.environ.setdefault(key, value)
    return applied


def scheduler(kind: str = "train", key: str | None = None) -> Any:
    """Scheduler facts for one job KIND: account, partition, qos, constraint.

    Split by kind because the split is real: at S3DF the corpus build runs on
    ``turing`` under ``mli:default`` while training runs on ``ampere`` under
    ``mli:cider-ml``, because authorisation is per partition. Collapsing them
    would put one of the two literals back into a shell script.

    These are NOT paths, which is why they were never in the roots table and
    ended up frozen into ``#SBATCH`` directives instead. ``#SBATCH`` genuinely
    cannot read shell variables, so the fix is to pass them as ``sbatch`` CLI
    flags -- which is what ``scripts/chain_coeff_fm_train.sh`` already does for
    ``--account``.

    Each key is overridable by ``HELIX_SLURM_<KEY>``, applied to whichever kind
    is asked for: a one-off ``HELIX_SLURM_QOS=premium`` should not need to know
    which job kind it is about to affect.
    """
    if kind not in JOB_KINDS:
        raise KeyError(f"unknown job kind {kind!r}; known: {', '.join(JOB_KINDS)}")
    spec = dict((site().get("scheduler") or {}).get(kind) or {})
    for k in list(spec) + ["account", "partition", "qos", "constraint"]:
        override = os.environ.get(f"HELIX_SLURM_{k.upper()}")
        if override:
            spec[k] = override
    if key is None:
        return spec
    return spec.get(key)


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
    """The PYTHONPATH a pimm run needs: pimm, then helix.

    NOT pimm-data. The image installs it at the pinned revision and the build
    refuses a stale one, so putting a checkout ahead of it would make the run
    disagree with the pin -- the image governed by one and the run by the other.
    HELIX_PIMM_DATA_SRC remains declared for the case where you deliberately
    want an unreleased pimm-data, and you then set PYTHONPATH yourself.
    """
    return os.pathsep.join((str(root("HELIX_PIMM_ROOT")), str(repo())))


def report() -> int:
    """Print the site, every root, its source, and whether it exists.

    Exit non-zero if anything is missing -- the first command to run on a new
    machine. A root the site declares as ``null`` is reported as ``not set`` and
    does NOT count as missing: the site genuinely does not have it, and treating
    absence as breakage is what made a clean NERSC checkout report eleven
    failures.
    """
    name = site_name()
    known = ", ".join(available_sites())
    if name:
        src = "HELIX_SITE" if os.environ.get("HELIX_SITE") else "auto-detected"
        print(f"site: {name}  ({src};  available: {known})\n")
    else:
        print(f"site: NONE selected and none detected  (available: {known})")
        print("  Set HELIX_SITE, or add a profile under helix/sites/.")
        print("  Every root below must then come from the environment.\n")

    missing = 0
    width = max(len(v) for v in ROOTS)
    for var, desc in ROOTS.items():
        try:
            value, source = resolve(var)
        except SiteError as exc:
            print(f"{var:<{width}}  ERROR    {exc}")
            missing += 1
            continue
        if value is None:
            state = "not set" if source.endswith("(null)") else "UNSET  "
            missing += source == "unset"
            print(f"{var:<{width}}  {state}  ({source})\n{'':<{width}}  {desc}")
            continue
        ok = value.exists()
        missing += not ok
        print(f"{var:<{width}}  {'ok ' if ok else 'MISSING'}  ({source:16s}) {value}"
              f"\n{'':<{width}}  {desc}")

    print(f"\n{'HELIX_ROOT':<{width}}  {'ok ' if repo().exists() else 'MISSING'}  "
          f"(derived)         {repo()}\n{'':<{width}}  this checkout")

    c = container()
    if c:
        print(f"\ncontainer: runtime={c.get('runtime')}  image={c.get('image')}")
        print(f"           python={c.get('python')}  binds={c.get('binds')}")
        if c.get("interactive_runtime"):
            print(f"           interactive: {c['interactive_runtime']} "
                  f"{' '.join(c.get('interactive_flags') or [])}")
    for kind in JOB_KINDS:
        s = scheduler(kind)
        if s:
            print(f"slurm[{kind}]: " + "  ".join(f"{k}={v}" for k, v in s.items()))

    if missing:
        print(f"\n{missing} root(s) missing or unset. Set the env vars above, "
              f"fix helix/sites/{name or '<site>'}.yaml, or see docs/RUNBOOK.md.")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(report())
