"""Every path that points OUTSIDE the repo, resolved in one place.

The values live in **site profiles**, ``helix/sites/<name>.yaml``; this module is a resolver and
names no site-specific path. Resolution order, highest first:

1. an explicit ``HELIX_<NAME>`` environment variable (a one-off override needs no file edit);
2. the selected site profile;
3. nothing. There is deliberately no fallback: a path that resolves to another site's layout
   fails late, one that does not resolve fails at ``python -m helix.paths``.

The site is ``HELIX_SITE``, or auto-detected from each profile's ``detect:`` block (a ``path``
that exists, or an ``env`` var that is set). A profile sets a root to ``null`` to say "this site
does not have that thing", reported as ``not set`` rather than ``MISSING``.

``python -m helix.paths`` prints the site, every root, where its value came from, and whether it
exists: the first thing to run on a new machine. Two site facts are not roots, because they are
not filesystem paths: the **container** (:func:`container`: a runtime, an image reference, an
in-image interpreter, binds) and the **scheduler** (:func:`scheduler`: account, partition, qos,
per job KIND -- at S3DF the corpus build and training need different accounts).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

#: Where the site profiles live: a plain data directory, NOT a subpackage, so this module
#: (imported by everything, the DSP-only install included) pulls in no helix subpackage.
SITES_DIR = Path(__file__).resolve().parent / "sites"

#: The vocabulary, name -> meaning. What a root MEANS is a property of helix, not of a machine,
#: so it is here and only the VALUES are in the site files.
ROOTS: dict[str, str] = {
    "HELIX_CORPUS_ROOT":   "parent of built coeff corpora",
    "HELIX_CORPUS":        "ONE corpus run dir, on the fastest filesystem (the corpus is its parent; "
                           "_calib/RUNS.txt lists the runs)",
    "HELIX_SIM_ROOT":      "JAXTPC wire campaign the corpus is built from (<campaign>/data/jaxtpc_wire)",
    "HELIX_ARCHIVE":       "bin tables, converted checkpoints",
    "HELIX_EXP":           "run outputs (save_path, logs, exports)",
    "HELIX_PIMM_ROOT":     "the pimm checkout training runs (the site's pinned worktree)",
    "HELIX_JAXTPC_ROOT":   "JAXTPC checkout (noise spectrum, geometry)",
}

#: Job kinds :func:`scheduler` knows.
JOB_KINDS = ("train", "corpus", "eval")

_SITE_CACHE: dict[str, Any] | None = None


class SiteError(RuntimeError):
    """A site profile is missing, unreadable, or does not declare what was asked -- raised
    rather than returning a default, since every default this module once carried was an S3DF
    path."""


def available_sites() -> list[str]:
    """Profile names that ship with this checkout, sorted."""
    return sorted(p.stem for p in SITES_DIR.glob("*.yaml"))


def _read_profile(name: str) -> dict[str, Any]:
    path = SITES_DIR / f"{name}.yaml"
    if not path.exists():
        raise SiteError(f"unknown site {name!r}: no {path}. Known sites: "
                        f"{', '.join(available_sites()) or '(none installed)'}.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SiteError(f"site profile is not a mapping: {path}")
    return data


def _detect() -> str | None:
    """The site whose ``detect:`` block matches, or None; two matching is an error, not a coin
    flip (picking one silently would send a job to the wrong storage)."""
    hits = []
    for name in available_sites():
        rule = _read_profile(name).get("detect") or {}
        if ("env" in rule and os.environ.get(str(rule["env"]))) or \
                ("path" in rule and Path(str(rule["path"])).exists()):
            hits.append(name)
    if len(hits) > 1:
        raise SiteError(f"site detection is ambiguous: {', '.join(hits)} all match. "
                        f"Set HELIX_SITE to one of them.")
    return hits[0] if hits else None


def site_name() -> str | None:
    """The selected site, or None if nothing is configured or detected."""
    return os.environ.get("HELIX_SITE") or _detect()


def site() -> dict[str, Any]:
    """The selected site profile, loaded once; empty when no site is selected (a fresh clone then
    resolves whatever the environment provides)."""
    global _SITE_CACHE
    if _SITE_CACHE is None:
        name = site_name()
        _SITE_CACHE = _read_profile(name) if name else {}
    return _SITE_CACHE


def _expand(value: str) -> str:
    """``$VAR`` expanded against the environment, strictly: ``os.path.expandvars`` leaves an
    unset variable in place, and ``$CFS/x`` would then report MISSING for the wrong reason."""
    out = os.path.expandvars(value)
    if "$" in out:
        missing = [w for w in out.replace("/", " ").split() if w.startswith("$")]
        raise SiteError(f"site profile value {value!r} references {', '.join(missing)}, "
                        f"which is not set in this environment.")
    return out


def resolve(name: str) -> tuple[Path | None, str]:
    """One root as ``(path_or_None, source)``, not raising on absence. ``source`` is ``env``,
    ``site:<name>``, ``site:<name> (null)`` or ``unset``; :func:`root` is the strict form."""
    if name not in ROOTS:
        raise KeyError(f"unknown helix root {name!r}; known roots: {', '.join(ROOTS)}")
    from_env = os.environ.get(name)
    if from_env:
        return Path(_expand(from_env)), "env"
    profile = site()
    roots = profile.get("roots") or {}
    if name in roots:
        if roots[name] is None:
            return None, f"site:{profile.get('name', '?')} (null)"
        return Path(_expand(str(roots[name]))), f"site:{profile.get('name', '?')}"
    return None, "unset"


def root(name: str) -> Path:
    """One external root, or a :class:`SiteError` that says what to do -- never an invented
    path, whose failure would surface much later from deep inside a reader."""
    value, source = resolve(name)
    if value is None:
        raise SiteError(
            f"{name} is not configured ({source}; site: {site_name() or 'no site selected'}).\n"
            f"  {ROOTS[name]}\n"
            f"  Set {name} in the environment, or add it to helix/sites/<site>.yaml, or set HELIX_SITE.\n"
            f"  Run `python -m helix.paths` to see every root at once.")
    return value


def repo() -> Path:
    """This checkout, located from THIS file (several checkouts and worktrees sit side by side,
    and importing another's code builds corpora with that tree's DSP). ``HELIX_ROOT`` overrides,
    to pin a resumed job to the tree its first link used."""
    return Path(os.environ.get("HELIX_ROOT", Path(__file__).resolve().parent.parent))


def container(key: str | None = None) -> Any:
    """The site's container: ``runtime``, ``image`` (a registry reference at NERSC, not a path),
    ``python`` (in the image), ``binds``, optionally ``interactive_runtime`` /
    ``interactive_flags``. Each is overridable by ``HELIX_CONTAINER_<KEY>``."""
    spec = dict(site().get("container") or {})
    for k in list(spec) + ["runtime", "image", "python"]:
        override = os.environ.get(f"HELIX_CONTAINER_{k.upper()}")
        if override:
            spec[k] = override
    if key is None:
        return spec
    if key not in spec:
        raise SiteError(f"container.{key} is not declared for site {site_name() or '(none)'}. "
                        f"Declared: {', '.join(sorted(spec)) or '(nothing)'}.")
    return spec[key]


def site_env() -> dict[str, str]:
    """The site's ``env:`` block: variables some library reads directly, above all
    ``HDF5_USE_FILE_LOCKING=FALSE`` (GPFS has no POSIX locks, so every ``h5py.File()`` on the
    corpus fails without it). A value already in the environment wins."""
    return {str(k): os.environ.get(str(k), str(v)) for k, v in (site().get("env") or {}).items()
            if v is not None}


def apply_site_env() -> dict[str, str]:
    """Set :func:`site_env` into ``os.environ`` (never overwriting) and return it: for entry
    points that run in-process rather than through ``scripts/helix_env.sh``."""
    applied = site_env()
    for key, value in applied.items():
        os.environ.setdefault(key, value)
    return applied


def scheduler(kind: str = "train", key: str | None = None) -> Any:
    """Scheduler facts for one job KIND (account, partition, qos, constraint, ...). Not paths:
    ``#SBATCH`` cannot read shell variables, so they reach ``sbatch`` as CLI flags
    (``scripts/submit.sh``, ``scripts/corpus.sh``). Each is overridable by ``HELIX_SLURM_<KEY>``."""
    if kind not in JOB_KINDS:
        raise KeyError(f"unknown job kind {kind!r}; known: {', '.join(JOB_KINDS)}")
    spec = dict((site().get("scheduler") or {}).get(kind) or {})
    for k in list(spec) + ["account", "partition", "qos", "constraint"]:
        override = os.environ.get(f"HELIX_SLURM_{k.upper()}")
        if override:
            spec[k] = override
    return spec if key is None else spec.get(key)


def packaged(name: str) -> Path:
    """A data file helix SHIPS, under ``helix/tpc/data/`` (the noise spectrum: a clone builds a
    corpus with no second repository present)."""
    return Path(__file__).resolve().parent / "tpc" / "data" / name


def report() -> int:
    """Print the site, every root, its source and whether it exists; non-zero if a root is
    missing or unset (a site's ``null`` is absence, not breakage, and does not count)."""
    name = site_name()
    known = ", ".join(available_sites())
    if name:
        src = "HELIX_SITE" if os.environ.get("HELIX_SITE") else "auto-detected"
        print(f"site: {name}  ({src};  available: {known})\n")
    else:
        print(f"site: NONE selected and none detected  (available: {known})")
        print("  Set HELIX_SITE, or add a profile under helix/sites/; every root below must then "
              "come from the environment.\n")
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
            missing += source == "unset"
            state = "not set" if source.endswith("(null)") else "UNSET  "
            print(f"{var:<{width}}  {state}  ({source})\n{'':<{width}}  {desc}")
            continue
        missing += not value.exists()
        print(f"{var:<{width}}  {'ok ' if value.exists() else 'MISSING'}  ({source:16s}) {value}"
              f"\n{'':<{width}}  {desc}")
    print(f"\n{'HELIX_ROOT':<{width}}  {'ok ' if repo().exists() else 'MISSING'}  "
          f"(derived)         {repo()}\n{'':<{width}}  this checkout")
    c = container()
    if c:
        print(f"\ncontainer: runtime={c.get('runtime')}  image={c.get('image')}")
        print(f"           python={c.get('python')}  binds={c.get('binds')}")
        if c.get("interactive_runtime"):
            print(f"           interactive: {c['interactive_runtime']} {' '.join(c.get('interactive_flags') or [])}")
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
