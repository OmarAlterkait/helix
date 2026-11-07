#!/bin/bash
# The Python a SUBMITTING host needs, and nothing more: helix's base install (PyYAML, numpy, h5py),
# so scripts/helix_env.sh can read helix/sites/<site>.yaml, plus pimm's dependencies minus pimm-data
# (tyro, submitit, ...), so scripts/submit.sh can run `pimm submit`, whose preflight reads the
# training config. No torch: jobs run in the container; the host only renders and calls sbatch.
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
# pimm's launch dependencies, read from its own pyproject so they cannot drift from it.
PIMM=$("$VENV/bin/python" -c 'from helix.paths import root; print(root("HELIX_PIMM_ROOT"))')
DEPS=$("$VENV/bin/python" - "$PIMM/pyproject.toml" <<'PY'
import re, sys
try:
    import tomllib
    deps = tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]
except ModuleNotFoundError:                       # python < 3.11
    m = re.search(r"^\s*dependencies\s*=\s*\[(.*?)\]", open(sys.argv[1]).read(), re.S | re.M)
    deps = re.findall(r'"([^"]+)"', m.group(1))
print(" ".join(d for d in deps if re.split(r"[<>=!~\[ ;]", d, 1)[0].strip().lower()
               not in {"pimm-data", "pimm_data", "torch"}))
PY
)
# shellcheck disable=SC2086
"$VENV/bin/python" -m pip install --quiet --no-cache-dir $DEPS
"$VENV/bin/python" -c 'import yaml, tyro, submitit, helix.paths; print("ok:", helix.paths.site_name())'
echo "host env: $VENV"
