#!/bin/bash
# Build the environment a SUBMITTING HOST needs, and nothing more.
#
#   scripts/make_launcher_env.sh            # create or update it
#   scripts/make_launcher_env.sh --print    # just say where it is
#
# WHY THIS EXISTS
#
# `pimm submit` runs on a login node. sbatch is not inside the container, so it
# cannot run there; the container's interpreter is therefore unavailable to it.
# pimm's own answer is documented -- "Login nodes and remote submission hosts may
# need only YAML parsing, Tyro, and Submitit... This environment can render and
# submit jobs. It cannot import the full model stack" (install.sh
# --launcher-only) -- but that recipe wants `uv`, which Perlmutter's login nodes
# do not carry, and it installs pimm-data, which drags torch back in.
#
# So: a venv with pimm's launcher dependencies MINUS pimm-data, plus helix's own
# base install. helix is needed because `pimm submit` loads the training config
# during preflight to check batch-size divisibility, and helix's config reads
# the bin table and the corpus run list at module scope. Those are numpy and
# h5py respectively -- see helix/data/__init__.py for why importing them no
# longer drags the training stack along, and tests/test_boundary.py for the
# check that keeps it that way.
#
# The dependency list is READ FROM pimm's pyproject.toml rather than retyped
# here. A second copy of someone else's dependency list is a copy that goes
# stale silently, and the failure it produces -- ModuleNotFoundError at submit
# time -- looks like a broken install rather than a stale list.
set -euo pipefail

H=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=/dev/null
source "$H/scripts/helix_env.sh"

: "${HELIX_PIMM_ROOT:?helix.paths did not resolve HELIX_PIMM_ROOT; run python -m helix.paths}"
: "${HELIX_PYTHON:?helix_env.sh did not export HELIX_PYTHON}"

# On SCRATCH by default, not in the checkout and not in $HOME: a venv in the
# checkout would be picked up by the container bind and shadow /opt/pimm/.venv,
# and one in $HOME is visible inside the container for the same reason
# PYTHONNOUSERSITE exists. Overridable, because "where scratch is" is a site
# fact and this is not the place to argue with it.
VENV=${HELIX_LAUNCHER_VENV:-${HELIX_SCRATCH:-/tmp}/helix-launcher-venv}

if [ "${1:-}" = "--print" ]; then
  echo "$VENV"
  exit 0
fi

PYPROJECT="$HELIX_PIMM_ROOT/pyproject.toml"
[ -f "$PYPROJECT" ] || { echo "no pyproject.toml at $PYPROJECT" >&2; exit 1; }

# tomllib is 3.11+; the regex branch covers 3.8-3.10. Both read the same flat
# [project].dependencies array, and both drop pimm-data -- the one entry that
# would pull torch>=2.5 and defeat the whole point.
DEPS=$("$HELIX_PYTHON" - "$PYPROJECT" <<'PY'
import re
import sys

path = sys.argv[1]
try:
    import tomllib
    with open(path, "rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
except ModuleNotFoundError:
    src = open(path, encoding="utf-8").read()
    m = re.search(r"^\s*dependencies\s*=\s*\[(.*?)\]", src, re.S | re.M)
    if not m:
        raise SystemExit("could not find [project].dependencies in " + path)
    deps = re.findall(r'"([^"]+)"', m.group(1))

# pimm-data is the whole reason this is a subset and not `pip install pimm`.
print(" ".join(d for d in deps
                if re.split(r"[<>=!~\[ ]", d, 1)[0].strip().lower()
                not in {"pimm-data", "pimm_data"}))
PY
)
[ -n "$DEPS" ] || { echo "resolved an empty dependency list from $PYPROJECT" >&2; exit 1; }

echo "launcher env : $VENV"
echo "interpreter  : $HELIX_PYTHON ($("$HELIX_PYTHON" -V 2>&1))"
echo "pimm deps    : $DEPS"

[ -d "$VENV" ] || "$HELIX_PYTHON" -m venv "$VENV"

# --no-cache-dir: pip's default cache lives under $HOME, which here is GPFS, and
# a GPFS lock failure surfaces as OSError 524 rather than anything readable.
PIP=("$VENV/bin/python" -m pip install --quiet --no-cache-dir)
"${PIP[@]}" --upgrade pip
# shellcheck disable=SC2086  # DEPS is a deliberate word list
"${PIP[@]}" $DEPS
# helix itself, base extras only -- this IS the DSP-only install, so if it needs
# anything heavier the boundary has been broken and we want to find out here.
"${PIP[@]}" -e "$H"

echo
"$VENV/bin/python" - <<'PY'
import importlib.util as u
import sys

heavy = [m for m in ("torch", "pimm_data", "hdf5plugin") if u.find_spec(m)]
import tyro, submitit, yaml, helix, helix.paths   # noqa: F401
print(f"ok: python {sys.version.split()[0]}, tyro + submitit + helix present")
print("   heavy packages present:", ", ".join(heavy) if heavy else "none "
      "(this is the point: the launcher cannot import the model stack)")
PY

cat <<EOF

Use it:
  HELIX_PIMM_PYTHON=$VENV/bin/python \\
    scripts/submit_helix.sh --recipe configs/launch/nersc-preempt.yaml

or leave HELIX_PIMM_PYTHON unset -- submit_helix.sh looks here by default.
EOF
