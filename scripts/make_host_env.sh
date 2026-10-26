#!/bin/bash
# The Python a SUBMITTING host needs for helix's scripts, and nothing more: helix's base install
# (PyYAML, numpy, h5py), so scripts/helix_env.sh can read helix/sites/<site>.yaml. Jobs run in the
# container; the host only reads the site profile and calls sbatch.
#
#   scripts/make_host_env.sh            create or update it ($SCRATCH/helix-host-venv)
#
# Needed where no system python has PyYAML (S3DF interactive nodes: python3 is 3.6, python3.12 has no
# yaml). helix_env.sh finds it there without being told. On SCRATCH, not in the checkout or $HOME:
# both are visible inside the container and would shadow its packages.
set -euo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${HELIX_HOST_VENV:-${SCRATCH:?SCRATCH is not set}/helix-host-venv}
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; raise SystemExit(sys.version_info < (3, 8))' 2>/dev/null \
      && { PY=$c; break; }
  done
fi
[ -n "$PY" ] || { echo "make_host_env.sh: no Python >= 3.8; set PYTHON=" >&2; exit 1; }
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
# --no-cache-dir: pip's cache is under $HOME
"$VENV/bin/python" -m pip install --quiet --no-cache-dir --upgrade pip
"$VENV/bin/python" -m pip install --quiet --no-cache-dir -e "$H"
"$VENV/bin/python" -c 'import yaml, helix.paths; print("ok:", helix.paths.site_name())'
echo "host env: $VENV"
