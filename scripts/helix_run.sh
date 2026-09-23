#!/bin/bash
# Run a command inside this site's helix container, whatever that means here.
#
#   scripts/helix_run.sh python -m helix.paths
#   scripts/helix_run.sh python -m pytest tests -q
#   scripts/helix_run.sh python scripts/build_coeff_corpus.py --shard ... --out ...
#
# `python` in the command is rewritten to the IN-IMAGE interpreter
# (container.python), so a caller never names /opt/pimm/.venv/bin/python either.
#
# Why this exists. The two submit scripts each spelled the invocation out:
# `apptainer exec -B /sdf,/lscratch $HELIX_IMAGE /opt/pimm/.venv/bin/python`,
# nine times in one file. Every part of that is site-specific -- apptainer is not
# installed at NERSC, there is no /sdf to bind, and the image is a registry
# reference rather than a .sif -- so porting meant editing nine lines in two
# files and hoping none were missed. The four facts now live in
# helix/sites/<site>.yaml under `container:` and this script reads them.
#
# INTERACTIVE vs BATCH. A site may have different runtimes for the two: at NERSC,
# shifter is what pimm's launcher drives in a batch job and podman-hpc is what a
# shell session uses. `container.interactive_runtime` declares the second, and
# this script prefers it when stdin is a TTY or HELIX_INTERACTIVE=1. Pass
# HELIX_INTERACTIVE=0 to force the batch runtime.
set -euo pipefail

H=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=/dev/null
source "$H/scripts/helix_env.sh"

[ $# -gt 0 ] || { echo "usage: $0 <command> [args...]" >&2; exit 2; }

RUNTIME=${HELIX_CONTAINER_RUNTIME:-}
IMAGE=${HELIX_CONTAINER_IMAGE:-}
IN_PY=${HELIX_CONTAINER_PYTHON:-python3}
BINDS=${HELIX_CONTAINER_BINDS:-}

# pimm AND helix, in that order -- the same pair helix.paths.pythonpath() returns.
# Setting only helix meant anything importing pimm (smoke_train_fm, the pimm
# config contract, the trainer) died with "No module named 'pimm'" INSIDE the
# container while working fine outside it.
HPP="$HELIX_ROOT"
[ -n "${HELIX_PIMM_ROOT:-}" ] && [ -d "${HELIX_PIMM_ROOT}" ] && HPP="$HELIX_PIMM_ROOT:$HELIX_ROOT"

# Prefer the interactive runtime when we are plainly interactive.
if [ "${HELIX_INTERACTIVE:-auto}" = "1" ] || \
   { [ "${HELIX_INTERACTIVE:-auto}" = "auto" ] && [ -t 0 ]; }; then
  RUNTIME=${HELIX_CONTAINER_INTERACTIVE_RUNTIME:-$RUNTIME}
fi

[ -n "$RUNTIME" ] || { echo "helix_run.sh: no container.runtime for site ${HELIX_SITE:-<none>}" >&2; exit 1; }
[ -n "$IMAGE" ]   || { echo "helix_run.sh: no container.image for site ${HELIX_SITE:-<none>}" >&2; exit 1; }

# Rewrite a leading bare `python`/`python3` to the in-image interpreter. Only the
# FIRST token, so `... -c "import python"` is untouched.
CMD=("$@")
case "${CMD[0]}" in
  python|python3) CMD[0]=$IN_PY ;;
esac

# The CHECKOUTS are bound explicitly, by their own paths, rather than trusted to
# fall under one of the site's mount modules. They usually do -- and then one day
# they do not: at NERSC `--home` binds $HOME as /global/homes/<u>/<user> while
# helix.paths.repo() derives the canonical /global/u1/<u>/<user>, the SAME
# directory by a different mount alias, which simply does not exist inside the
# container. A checkout outside $HOME (a worktree on scratch, say) fails the same
# way for a plainer reason.
MOUNTS=()
for d in "$HELIX_ROOT" "${HELIX_PIMM_ROOT:-}"; do
  if [ -n "$d" ] && [ -d "$d" ]; then MOUNTS+=("$d"); fi
done

# Every HELIX_* the job needs, forwarded explicitly. Containers do not inherit
# the caller's environment uniformly across these three runtimes, and a job that
# silently loses HELIX_CORPUS reads whatever the site default is.
ENVS=()
while IFS='=' read -r k _; do
  case "$k" in HELIX_*) ENVS+=("$k") ;; esac
done < <(env | grep -E '^HELIX_' | sort)
# ...plus the site's own environment, whose names are NOT HELIX_* on purpose:
# h5py reads HDF5_USE_FILE_LOCKING, not a helix alias for it.
for k in ${HELIX_SITE_ENV_NAMES:-}; do
  [ -n "${!k:-}" ] && ENVS+=("$k")
done
# ...plus anything the caller names in HELIX_FORWARD_ENV (space-separated).
#
# This exists because its absence produced a WRONG ANSWER. Diagnosing the jax
# backend's accuracy, `JAX_DEFAULT_MATMUL_PRECISION=highest ./helix_run.sh ...`
# appeared to show that precision was not the cause -- but the variable was set
# on the host and never forwarded, so the container ran at the default. The real
# cause WAS precision. A silently dropped variable does not look like a dropped
# variable; it looks like evidence.
#
#   HELIX_FORWARD_ENV="JAX_DEFAULT_MATMUL_PRECISION XLA_FLAGS" scripts/helix_run.sh ...
for k in ${HELIX_FORWARD_ENV:-}; do
  if [ -z "${!k:-}" ]; then
    echo "helix_run.sh: HELIX_FORWARD_ENV names $k, which is unset -- not forwarded." >&2
  else
    ENVS+=("$k")
  fi
done

# Run from the checkout under every runtime (podman-hpc already did): Python
# puts the working directory ahead of PYTHONPATH, so a caller standing in another
# checkout would otherwise import that tree's code.
cd "$HELIX_ROOT"

case "$RUNTIME" in
  apptainer|singularity)
    ARGS=("$RUNTIME" exec --nv)
    if [ -n "$BINDS" ]; then ARGS+=(-B "$BINDS"); fi
    for d in "${MOUNTS[@]}"; do ARGS+=(-B "$d:$d"); done
    for k in "${ENVS[@]}"; do ARGS+=(--env "$k=${!k}"); done
    # PYTHONNOUSERSITE: -B of a home directory makes anything pip-installed under
    # ~/.local visible inside the image, so a "green" run can be one person's
    # alone. See CLAUDE.md.
    ARGS+=(--env PYTHONNOUSERSITE=1 --env "PYTHONPATH=$HPP")
    exec "${ARGS[@]}" "$IMAGE" "${CMD[@]}"
    ;;
  shifter)
    ARGS=(shifter "--image=$IMAGE")
    [ -n "${HELIX_CONTAINER_MODULE:-}" ] && ARGS+=("--module=${HELIX_CONTAINER_MODULE}")
    # shifter inherits the caller's environment, so the HELIX_* set is already
    # present; PYTHONPATH is the one thing train.sh would otherwise overwrite.
    export PYTHONNOUSERSITE=1 PYTHONPATH="$HPP${PYTHONPATH:+:$PYTHONPATH}"
    exec "${ARGS[@]}" -- "${CMD[@]}"
    ;;
  podman-hpc)
    ARGS=(podman-hpc run --rm)
    # interactive_flags carries the site's mount modules and the supplementary
    # group without which a group-restricted corpus is unreadable.
    IFS=',' read -r -a IFLAGS <<< "${HELIX_CONTAINER_INTERACTIVE_FLAGS:-}"
    for f in "${IFLAGS[@]}"; do [ -n "$f" ] && ARGS+=("$f"); done
    for d in "${MOUNTS[@]}"; do ARGS+=(-v "$d:$d"); done
    for k in "${ENVS[@]}"; do ARGS+=(-e "$k=${!k}"); done
    ARGS+=(-e PYTHONNOUSERSITE=1 -e "PYTHONPATH=$HPP")
    # NOT `-w $HELIX_ROOT`: podman validates the workdir against the IMAGE before
    # applying the mount modules, so a bound path fails with "workdir does not
    # exist on container". cd inside instead.
    exec "${ARGS[@]}" "$IMAGE" bash -lc "cd '$HELIX_ROOT' && exec $(printf '%q ' "${CMD[@]}")"
    ;;
  none)
    # Bare metal: the site runs helix directly in its own environment.
    export PYTHONPATH="$HPP${PYTHONPATH:+:$PYTHONPATH}"
    exec "${CMD[@]}"
    ;;
  *)
    echo "helix_run.sh: unknown container.runtime '$RUNTIME'." >&2
    echo "  Known: apptainer, singularity, shifter, podman-hpc, none." >&2
    exit 1
    ;;
esac
