#!/bin/bash
# Export every HELIX_* root for the current site. SOURCE this, do not run it.
#
#   source scripts/helix_env.sh
#
# There is exactly one source of these values -- helix/sites/<site>.yaml, read
# through helix.paths -- and this script asks Python for them rather than
# repeating them. That is the whole point: a shell script that hardcodes
# /sdf/... or /global/cfs/... is a second table, and two tables drift. Every
# variable printed here is one a caller could also set by hand; setting one wins,
# because helix.paths checks the environment first.
#
# Why a shell script at all, when helix.paths already exists: `#SBATCH`
# directives cannot read shell variables, pimm's launch YAML cannot read
# environment variables (pimm/launch/config.py resolves {placeholders} against
# the config, never os.environ), and pimm's scripts/train.sh reads EXP_ROOT from
# the environment. All three need the values as env vars, in a shell, before
# Python runs. This is the one place that conversion happens.
#
# HELIX_SITE selects the profile; without it helix.paths auto-detects. Set it
# explicitly in a batch job, where auto-detection is a guess made on a node.
set -u

_helix_env_root() {
  # Locate the checkout from THIS file, never by name: three helix checkouts have
  # existed side by side and picking the wrong one is a correctness bug, not a
  # portability one (see helix/paths.py:repo).
  local src="${BASH_SOURCE[0]}"
  cd "$(dirname "$src")/.." && pwd
}

HELIX_ROOT="${HELIX_ROOT:-$(_helix_env_root)}"
export HELIX_ROOT

# Ask helix.paths for everything else. This is the HOST interpreter -- helix.paths
# imports only os/pathlib/yaml, so it needs none of the scientific stack.
#
# But it does need Python >= 3.7 (`from __future__ import annotations`), and the
# bare `python3` on a Perlmutter COMPUTE node is the 3.6 OS interpreter, which
# dies with "future feature annotations is not defined". It works on a login node
# only because a `module load python` earlier in the session leaked onto PATH --
# i.e. this script appeared to work for a reason that does not survive srun.
# So: search, and say so when nothing is suitable.
_helix_env_pick_py() {
  local c
  for c in "${HELIX_PYTHON:-}" python3 python3.13 python3.12 python3.11 \
           python3.10 python3.9 python3.8 python; do
    [ -n "$c" ] || continue
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' \
      >/dev/null 2>&1 && { echo "$c"; return 0; }
  done
  return 1
}
_helix_env_py=$(_helix_env_pick_py) || {
  echo "helix_env.sh: no Python >= 3.8 found. Tried HELIX_PYTHON, python3," >&2
  echo "              python3.13..3.8, python. On NERSC compute nodes the bare" >&2
  echo "              python3 is 3.6; set HELIX_PYTHON to a real interpreter," >&2
  echo "              or run inside the container where /opt/pimm/.venv has one." >&2
  return 1 2>/dev/null || exit 1
}
_helix_env_out=$(PYTHONPATH="$HELIX_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$_helix_env_py" - <<'PY'
import shlex
from helix import paths

# Roots that resolve. A root the site declares as null is SKIPPED rather than
# exported empty: an empty HELIX_LEGACY_CORPUS would satisfy `${VAR:-default}`
# in a downstream script and then fail as a path, which is worse than unset.
for name in paths.ROOTS:
    value, _src = paths.resolve(name)
    if value is not None:
        print(f"export {name}={shlex.quote(str(value))}")

site = paths.site_name()
if site:
    print(f"export HELIX_SITE={shlex.quote(site)}")

# The container and scheduler facts, flattened into the same vocabulary the
# submit scripts use. These are NOT paths, which is why they were never in the
# roots table and ended up frozen into #SBATCH blocks and `apptainer exec` lines.
for key, value in (paths.container() or {}).items():
    if isinstance(value, (list, tuple)):
        value = ",".join(str(v) for v in value)
    if value not in (None, ""):
        print(f"export HELIX_CONTAINER_{key.upper()}={shlex.quote(str(value))}")

# Site environment: things a library reads directly, like HDF5_USE_FILE_LOCKING.
# An existing value wins (site_env honours os.environ), so this is a default.
# The NAMES are exported too, because helix_run.sh forwards HELIX_* generically
# and these are deliberately not called HELIX_anything -- h5py reads the real name.
_site_env = paths.site_env()
for key, value in _site_env.items():
    print(f"export {key}={shlex.quote(value)}")
if _site_env:
    print(f"export HELIX_SITE_ENV_NAMES={shlex.quote(' '.join(sorted(_site_env)))}")

# HELIX_SLURM_<KIND>_<KEY>, not HELIX_<KIND>_<KEY>: the latter puts
# HELIX_CORPUS_QOS (a scheduler fact) next to HELIX_CORPUS_ROOT (a path root)
# and they read as the same family when they are not.
for kind in paths.JOB_KINDS:
    for key, value in (paths.scheduler(kind) or {}).items():
        if value not in (None, ""):
            print(f"export HELIX_SLURM_{kind.upper()}_{key.upper()}="
                  f"{shlex.quote(str(value))}")
PY
) || {
  echo "helix_env.sh: helix.paths failed. Run '$_helix_env_py -m helix.paths'" >&2
  echo "              from $HELIX_ROOT to see why." >&2
  return 1 2>/dev/null || exit 1
}

eval "$_helix_env_out"
unset _helix_env_out _helix_env_py
unset -f _helix_env_root _helix_env_pick_py
