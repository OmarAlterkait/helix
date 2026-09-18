#!/bin/bash
# What goes INTO the helix image, for both packaging front-ends.
#
# Apptainer (container/helix-train.def) and Docker (container/Dockerfile) build
# the same environment; only the packaging differs. Keeping the install steps in
# each would be the duplication hazard this repository keeps paying for -- two
# copies of a lockstep-critical recipe, free to drift, with nothing comparing
# them. There is one copy, and it is this file.
#
# Inputs, all optional:
#
#   JAX_EXTRA       cuda12 (default) or cpu. See the hazard note below.
#   PIMM_DATA_REV   override the revision. Defaults to the one pyproject.toml
#                   pins, which is the point: the image and a dev install cannot
#                   name different revisions, because they read the same line.
#   VENV            the environment to install into (default /opt/pimm/.venv).
#   HELIX_PYPROJECT where to read the pin from (default /opt/helix-build/pyproject.toml).
#
# Requires the pimm base image: it supplies the venv, uv, and torch.

set -e

VENV=${VENV:-/opt/pimm/.venv}
UV=${UV:-/usr/local/uv/bin/uv}
JAX_EXTRA=${JAX_EXTRA:-cuda12}
HELIX_PYPROJECT=${HELIX_PYPROJECT:-/opt/helix-build/pyproject.toml}

# uv installs by hardlinking out of its cache, and in a build sandbox those links
# do not materialise: a cached package "installs" in 99 ms and then imports as
# ModuleNotFoundError: No module named 'pywt.version'. link-mode=copy fixes that.
# The cache goes INSIDE the sandbox so every build starts empty -- which is what
# an earlier UV_NO_CACHE=1 was reaching for, and the better way to get it:
# disabling the cache made uv re-extract into TMPDIR on every invocation and the
# build then died partway through the largest package.
export UV_LINK_MODE=copy
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uvcache}

# TMPDIR is INHERITED from whoever runs the build, and inside an apptainer %post
# sandbox that path usually does not exist: on S3DF it is $LSCRATCH/..., bound
# for `apptainer exec` but NOT for %post. uv writes its interpreter-probe module
# there, so the probe vanishes and uv dies with
#
#   Failed to inspect Python interpreter ...
#   ModuleNotFoundError: No module named 'python.get_interpreter_info'
#
# which reads like a broken interpreter and is nothing of the kind. Pinning it
# makes the build independent of the builder's environment, which is the point.
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
df -h "$TMPDIR" | tail -1

# The revision comes from pyproject.toml unless overridden. helix used to name it
# in the def AND in pyproject.toml, and a bump applied to one of them produced an
# image on the wrong side of the forward-model boundary -- which does not degrade
# quietly, it fails to import. One source, read by both front-ends.
if [ -z "${PIMM_DATA_REV:-}" ]; then
    [ -f "$HELIX_PYPROJECT" ] || {
        echo "FATAL: no pyproject.toml at $HELIX_PYPROJECT and no PIMM_DATA_REV set."
        echo "       The image pins pimm-data by reading helix's own pin; one of the"
        echo "       two must be present."; exit 1; }
    PIMM_DATA_REV=$(grep -oE 'pimm-data[^#]*rev[[:space:]]*=[[:space:]]*"[0-9a-f]{40}"' \
                        "$HELIX_PYPROJECT" | grep -oE '[0-9a-f]{40}' | head -1)
    [ -n "$PIMM_DATA_REV" ] || {
        echo "FATAL: $HELIX_PYPROJECT does not pin pimm-data by an explicit 40-char rev."
        echo "       An unpinned git dependency follows whatever the default branch"
        echo "       holds, which is how an environment ends up straddling the"
        echo "       forward-model move."; exit 1; }
fi
echo "pimm-data revision: $PIMM_DATA_REV (from ${PIMM_DATA_REV_SOURCE:-$HELIX_PYPROJECT})"

# Replace the stale 0.3.0 rather than shadow it. An editable .pth or a PYTHONPATH
# entry would leave two copies resolvable and make which one wins depend on
# sys.path order -- exactly the fragility this image exists to end.
#
# setuptools copies sources into build/lib and never prunes, so a build/ left from
# before the forward model moved re-ships every module deleted since. It did: an
# earlier image had pimm_data.{noise,noise_jax,coeff,coeff_verify} and
# readers.coeff_tpc importable, and that coeff_verify was an OLDER copy missing
# the source_root gap fix (74cfe5f) -- silently wrong answers from a module that
# is supposed to be gone. build/ is gitignored, so it is invisible to every check
# that looks at the repo. Installing from a git URL sidesteps it entirely.
"$UV" pip uninstall --python "$VENV/bin/python" pimm-data || true
"$UV" pip install --python "$VENV/bin/python" --no-deps \
    "pimm-data @ git+https://github.com/OmarAlterkait/pimm-data@${PIMM_DATA_REV}"

# ONE install for everything else. It was four separate uv invocations, each
# re-resolving and re-extracting; the build died partway through the last of them.
# One call also means one consistent resolution.
#
#   PyWavelets  helix's DSP half -- its only dep not already here.
#   pytest      because the first thing anyone does with a new environment is ask
#               whether it works, and the base image has no runner.
#   jax         $JAX_EXTRA. The hazard is NOT the GPU: torch 2.10+cu126 brings 16
#               nvidia-*-cu12 wheels (cudnn 9.10.2.21, cublas 12.6.4.1, ...) and a
#               CUDA jax depends on the same wheel family. If the resolver moves
#               any of them to satisfy jax, torch is what breaks, six hours into a
#               training run. So the versions are captured before and compared
#               after, and a MOVE fails the build.
#
#               cuda12_local is not an option: it wants a system cuDNN and this
#               image has none -- /usr/local/cuda has the runtime and nvcc, but
#               libcudnn arrives only as torch's pip wheel.
# Does the base ship an MPI-enabled h5py? The NERSC base (pimm-nersc) rebuilds
# h5py from source against a parallel HDF5, and a binary wheel pulled in later
# would silently replace it with a serial one -- the image would still import,
# and collective I/O would simply not be available. Recorded here, asserted after.
H5PY_MPI_BEFORE=$("$VENV/bin/python" -c \
    "import h5py; print(int(bool(h5py.get_config().mpi)))" 2>/dev/null || echo 0)
[ "$H5PY_MPI_BEFORE" = "1" ] && echo "base ships MPI-enabled h5py; will verify it survives"

"$VENV/bin/python" - <<'PYSNAP' > /tmp/nvidia_before.txt
import importlib.metadata as m
for d in sorted(m.distributions(), key=lambda d: d.metadata["Name"] or ""):
    n = d.metadata["Name"] or ""
    if n.startswith("nvidia-") or n in ("torch",):
        print(n, d.version)
PYSNAP
cat /tmp/nvidia_before.txt

"$UV" pip install --python "$VENV/bin/python" \
    PyWavelets pytest pytest-benchmark "jax[${JAX_EXTRA}]"

"$VENV/bin/python" - <<'PYSNAP' > /tmp/nvidia_after.txt
import importlib.metadata as m
for d in sorted(m.distributions(), key=lambda d: d.metadata["Name"] or ""):
    n = d.metadata["Name"] or ""
    if n.startswith("nvidia-") or n in ("torch",):
        print(n, d.version)
PYSNAP

# Compare only what existed BEFORE. An ADDITION is fine and expected:
# jax[cuda12] brings its own nvidia-cuda-nvcc-cu12 (12.9) for ptxas, beside
# torch's 12.6 runtime, and touches nothing torch resolved. What must never happen
# is a package torch already had being MOVED or REMOVED to satisfy jax. The first
# version of this guard failed on the addition and would have forced jax[cpu] for
# no reason.
JAX_EXTRA="$JAX_EXTRA" "$VENV/bin/python" - <<'PYGUARD'
import os


def read(p):
    return dict(l.split() for l in open(p) if l.strip())


before, after = read("/tmp/nvidia_before.txt"), read("/tmp/nvidia_after.txt")
moved = {k: (v, after.get(k)) for k, v in before.items() if after.get(k) != v}
added = sorted(set(after) - set(before))
if added:
    print("  jax added (fine):", ", ".join(f"{k} {after[k]}" for k in added))
if moved:
    for k, (was, now) in sorted(moved.items()):
        print(f"  MOVED {k}: {was} -> {now}")
    raise SystemExit(
        f"FATAL: installing jax[{os.environ['JAX_EXTRA']}] moved a package torch "
        "had pinned.\n"
        "       Rebuild with JAX_EXTRA=cpu, or pin jax to a release whose "
        "nvidia-* requirements match torch 2.10+cu126.")
print("  torch's own CUDA wheels: unchanged")
PYGUARD

# h5py must still be the one the base built. Nothing helix installs depends on
# h5py -- pimm-data goes in with --no-deps, and PyWavelets/pytest/jax do not want
# it -- so this should never fire. It is here because if it ever does, the symptom
# at NERSC is not an error: it is collective I/O quietly being unavailable.
if [ "$H5PY_MPI_BEFORE" = "1" ]; then
    H5PY_MPI_AFTER=$("$VENV/bin/python" -c \
        "import h5py; print(int(bool(h5py.get_config().mpi)))" 2>/dev/null || echo 0)
    [ "$H5PY_MPI_AFTER" = "1" ] || {
        echo "FATAL: h5py lost its MPI support during this install."
        echo "       The base built h5py from source against a parallel HDF5 and"
        echo "       something here replaced it with a binary wheel. Find what"
        echo "       pulled h5py in and give it --no-deps."; exit 1; }
    echo "  h5py: still MPI-enabled"
fi

df -h "$TMPDIR" | tail -1

# Drop the cache so it is not captured into the image, but NEVER fail the build
# over it. Under apptainer's fakeroot overlay `rm -rf` can report
#
#   rm: cannot remove '.../archive-v0/.../_pytest/_io': Is a directory
#
# on a tree it is perfectly able to walk -- an overlay whiteout quirk, not a
# permissions or logic error. With `set -e` that aborted a six-minute build one
# line from the end, after every install had already succeeded. Cleanup is an
# image-size optimisation; it is not a correctness requirement, and it has no
# business being fatal. If it does not fully clear, say so rather than leave the
# size unexplained.
rm -rf "$UV_CACHE_DIR" 2>/dev/null || find "$UV_CACHE_DIR" -mindepth 1 -delete 2>/dev/null || true
if [ -d "$UV_CACHE_DIR" ] && [ -n "$(ls -A "$UV_CACHE_DIR" 2>/dev/null)" ]; then
    echo "note: uv cache not fully removed ($(du -sh "$UV_CACHE_DIR" 2>/dev/null | cut -f1)); the image carries it"
fi

# Fail the BUILD, not a training run six hours in.
JAX_EXTRA="$JAX_EXTRA" "$VENV/bin/python" - <<'PYCHECK'
import importlib.util as _iu
import os

import pywt
import pimm_data
from pimm_data.transform import TRANSFORMS

names = set(TRANSFORMS.module_dict)
# tuple, not string: "0.10" < "0.4" lexically
assert tuple(int(x) for x in pimm_data.__version__.split(".")[:2]) >= (0, 4), \
    f"stale pimm_data {pimm_data.__version__}"
assert "Densify" in names, "pimm_data lost Densify"
for gone in ("AddNoise", "Digitize"):
    assert gone not in names, f"pimm_data still registers {gone}: lockstep violated"

import pytest as _pt
import jax as _jx
import torch as _t

# NOT asserting a GPU is visible: this image is built on whatever node or runner
# is free, usually one with no GPU at all, and the jax wheels are arch-generic so
# the build host's hardware says nothing about where the image will run.

# Modules the boundary move DELETED. The registry check above cannot see these:
# they register nothing, so it passed while an image shipped all five.
for _gone in ("pimm_data.noise", "pimm_data.noise_jax", "pimm_data.coeff",
              "pimm_data.coeff_verify", "pimm_data.readers.coeff_tpc"):
    assert _iu.find_spec(_gone) is None, (
        f"{_gone} was deleted from pimm-data but is installed -- a stale "
        f"build/lib leaked into the wheel")

print(f"OK: pywt {pywt.__version__}, pimm_data {pimm_data.__version__}, "
      f"pytest {_pt.__version__}, jax {_jx.__version__} "
      f"({os.environ['JAX_EXTRA']}), torch {_t.__version__}, "
      f"{len(names)} transforms, forward model absent as expected")
PYCHECK
