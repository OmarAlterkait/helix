#!/bin/bash
# Submit a helix training run through pimm's launcher.
#
#   scripts/submit.sh CONFIG [--recipe preempt|interactive|PATH] [pimm submit flags...] [-- key=value ...]
#
#   scripts/submit.sh fm --resources.nnodes 4                      # FM stable phase, 16 GPUs, preempt chain
#   scripts/submit.sh fm_cooldown --chain.jobs 1
#   scripts/submit.sh denoise --run.name denoise_m6dc -- model.cond_charge=True
#   scripts/submit.sh fm_smoke --recipe interactive --resources.nnodes 2 --dry-run
#
# CONFIG is a file under configs/pimm/ (fm, fm_cooldown, fm_smoke, denoise) or a
# path. The run is $HELIX_EXP/<run name>, the name defaulting to CONFIG: other
# configs refer to runs by that path (fm_cooldown reads $HELIX_EXP/fm). A study
# of several arms is scripts/sweep.py.
#
# What this adds to `pimm submit`, and why it cannot live in a recipe: every
# site fact (pimm site, repo and exp roots, image, interpreter, account, QOS,
# constraint, GPU directive) comes from helix/sites/<site>.yaml via
# helix_env.sh, because launch YAML cannot read the environment. A recipe that
# sets its own QOS (interactive.yaml) keeps it. Submission runs OUTSIDE the
# container (sbatch is not in it), in the launcher-only environment
# scripts/make_host_env.sh builds: pimm's launch dependencies and helix's base
# install, no torch.
set -euo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=/dev/null
source "$H/scripts/helix_env.sh"
: "${HELIX_PIMM_ROOT:?helix.paths did not resolve HELIX_PIMM_ROOT; run python -m helix.paths}"
: "${HELIX_EXP:?helix.paths did not resolve HELIX_EXP; run python -m helix.paths}"

[ $# -gt 0 ] && [ "${1#-}" = "$1" ] || { sed -n '2,9p' "$0" >&2; exit 2; }
CONFIG=$1; shift
case "$CONFIG" in
  /*|*/*|*.py) CFG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "${CONFIG%.py}").py ;;
  *)           CFG=$H/configs/pimm/$CONFIG.py ;;
esac
[ -f "$CFG" ] || { echo "submit.sh: no config $CFG" >&2; exit 2; }

# The recipe: a name under configs/launch/ or a path; pimm wants it absolute.
RECIPE=$H/configs/launch/preempt.yaml
NAME=$(basename "${CFG%.py}")
REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --recipe)    RECIPE=$2; shift 2 ;;
    --recipe=*)  RECIPE=${1#--recipe=}; shift ;;
    --run.name)  NAME=$2; shift 2 ;;
    --run.name=*) NAME=${1#--run.name=}; shift ;;
    *)           REST+=("$1"); shift ;;
  esac
done
case "$RECIPE" in
  /*) ;;
  */*|*.yaml) RECIPE=$(cd "$(dirname "$RECIPE")" && pwd)/$(basename "$RECIPE") ;;
  *) RECIPE=$H/configs/launch/$RECIPE.yaml ;;
esac
[ -f "$RECIPE" ] || { echo "submit.sh: no recipe $RECIPE" >&2; exit 2; }

case "${HELIX_CONTAINER_RUNTIME:-}" in
  shifter|podman-hpc)    PIMM_SITE=${HELIX_PIMM_SITE:-nersc-container} ;;
  apptainer|singularity) PIMM_SITE=${HELIX_PIMM_SITE:-s3df-container} ;;
  *) PIMM_SITE=${HELIX_PIMM_SITE:?cannot map container runtime ${HELIX_CONTAINER_RUNTIME:-unset} to a pimm site; set HELIX_PIMM_SITE} ;;
esac

PY=${HELIX_PIMM_PYTHON:-${HELIX_HOST_VENV:-${SCRATCH:-/nonexistent}/helix-host-venv}/bin/python}
[ -x "$PY" ] || { echo "submit.sh: no launcher environment at $PY; run scripts/make_host_env.sh" >&2; exit 2; }
# Preflight reads the training config on this host; it imports helix.
export PYTHONPATH="$HELIX_PIMM_ROOT:$H${PYTHONPATH:+:$PYTHONPATH}"
export HELIX_ROOT=$H                       # the job's setup: line reads it (configs/launch/_common.yaml)

# Logs and submitit's pickles beside the run, not in the pimm checkout (pimm's default is relative
# to its repo root).
ARGS=(--site "$PIMM_SITE" --recipe "$RECIPE" --train.config "$CFG" --run.name "$NAME"
      --paths.repo-root "$HELIX_PIMM_ROOT" --paths.exp-root "$HELIX_EXP"
      --submit.folder "$HELIX_EXP/$NAME/submitit" --resources.output "$HELIX_EXP/$NAME/slurm-%j.out")
[ -n "${HELIX_CONTAINER_IMAGE:-}" ]  && ARGS+=(--container.image "$HELIX_CONTAINER_IMAGE")
[ -n "${HELIX_CONTAINER_PYTHON:-}" ] && ARGS+=(--container.interpreter "$HELIX_CONTAINER_PYTHON")
[ -n "${HELIX_SLURM_TRAIN_ACCOUNT:-}" ]    && ARGS+=(--resources.account "$HELIX_SLURM_TRAIN_ACCOUNT")
[ -n "${HELIX_SLURM_TRAIN_CONSTRAINT:-}" ] && ARGS+=(--resources.constraint "$HELIX_SLURM_TRAIN_CONSTRAINT")
[ -n "${HELIX_SLURM_TRAIN_PARTITION:-}" ]  && ARGS+=(--resources.partition "$HELIX_SLURM_TRAIN_PARTITION")
[ -n "${HELIX_SLURM_TRAIN_TIME_MIN:-}" ]   && ARGS+=(--resources.time-min "$HELIX_SLURM_TRAIN_TIME_MIN")
[ "${HELIX_SLURM_TRAIN_GPUS_FLAG:-}" = gpus-per-node ] && ARGS+=(--resources.gpu-directive gpus-per-node)
# The recipe's own shape: an interactive recipe takes the site's interactive QOS, any other the
# training QOS; its node topology is part of what a run is (below).
read -r R_INTER R_NNODES R_NPROC < <("$PY" - "$RECIPE" <<'PY'
import sys, yaml
def load(p):
    import os
    r = yaml.safe_load(open(p)) or {}
    out = {}
    for b in r.get("_base_") or []:
        out.update(load(os.path.join(os.path.dirname(p), b)))
    for k, v in r.items():
        out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
r = load(sys.argv[1]); res = r.get("resources") or {}
print(int(bool(r.get("interactive"))), res.get("nnodes", 1), res.get("nproc_per_node", 4))
PY
)
if [ "$R_INTER" = 1 ]; then
  : "${HELIX_SLURM_TRAIN_INTERACTIVE_QOS:?site $HELIX_SITE declares no scheduler.train.interactive_qos; use the preempt recipe}"
  ARGS+=(--resources.qos "$HELIX_SLURM_TRAIN_INTERACTIVE_QOS")
elif [ -n "${HELIX_SLURM_TRAIN_QOS:-}" ]; then
  ARGS+=(--resources.qos "$HELIX_SLURM_TRAIN_QOS")
fi

# What this run is: the same NAME continues only the same config, training overrides and node
# topology (a resubmitted chain resumes from $HELIX_EXP/NAME -- below -- and the config derives
# its batch and LR from the world size). Other scheduling flags may change. Recorded only once
# pimm has accepted the submission, so a rejected one does not lock the name.
TRAIN=(); seen=0; NN=$R_NNODES; NP=$R_NPROC; prev=
for a in "${REST[@]}"; do
  if [ "$seen" = 1 ]; then TRAIN+=("$a"); continue; fi
  case "$prev" in --resources.nnodes) NN=$a ;; --resources.nproc-per-node) NP=$a ;; esac
  case "$a" in --) seen=1 ;; --resources.nnodes=*) NN=${a#*=} ;; --resources.nproc-per-node=*) NP=${a#*=} ;; esac
  prev=$a
done
WHAT="${CFG#"$H"/} nodes=$NN x $NP ${TRAIN[*]}"       # relative: a frozen run tree's path is not the run's identity
RUN=$HELIX_EXP/$NAME
DRY=0; case " ${REST[*]} " in *" --dry-run "*) DRY=1 ;; esac
if [ "$DRY" = 0 ] && [ -s "$RUN/submit.cmd" ] && [ "$(cat "$RUN/submit.cmd")" != "$WHAT" ]; then
  echo "submit.sh: $RUN is a run of: $(cat "$RUN/submit.cmd")" >&2
  echo "           not of:            $WHAT  (pass --run.name for a new run)" >&2
  exit 2
fi

# A run with a checkpoint resumes. pimm resumes a chain's later links, but its FIRST link takes the
# config's train.resume (False), and train.sh's fresh branch then re-snapshots the code and writes
# over model/last (pimm/launch/submit.py:234-236).
[ -d "$RUN/model/last" ] && ARGS+=(--train.resume)

echo "submit.sh: $NAME <- $CFG, recipe $(basename "$RECIPE"), site $HELIX_SITE ($PIMM_SITE), exp $HELIX_EXP" >&2
"$PY" -m pimm.cli submit "${ARGS[@]}" "${REST[@]}" || exit $?
if [ "$DRY" = 0 ]; then mkdir -p "$RUN" && echo "$WHAT" > "$RUN/submit.cmd"; fi
