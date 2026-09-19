#!/bin/bash
# Submit a helix training run through `pimm submit`, with every site fact
# injected from helix.paths.
#
#   scripts/submit_helix.sh --recipe configs/launch/nersc-preempt.yaml
#   scripts/submit_helix.sh --recipe configs/launch/nersc-premium.yaml \
#       --resources.qos premium --dry-run
#
# Anything after the flags this script sets is passed straight to `pimm submit`,
# so --dry-run, --chain.jobs, --run.name and the rest work as documented.
#
# WHAT THIS EXISTS TO DO, and why it is not just an alias.
#
# pimm's launch YAML cannot read the environment. `format_string`
# (pimm/launch/config.py) resolves {placeholders} against the launch config only
# and hard-exits on an unknown key; `env:` values are shlex.quote'd at render
# time (local.py), so a literal $VAR is exported as the characters `$VAR`. The
# CLI is therefore the ONLY seam that accepts a computed value -- and every value
# below is computed from helix/sites/<site>.yaml rather than written down twice.
#
# Without this, the three recipes each carried
#   paths.exp_root: /global/cfs/cdirs/m5238/users/oalter/exp/helix
#   env.PYTHONPATH: /global/homes/o/oalter/nu/helix
# which put one value in four places and hardcoded one user's home directory into
# the repository.
set -euo pipefail

H=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=/dev/null
source "$H/scripts/helix_env.sh"

: "${HELIX_PIMM_ROOT:?helix.paths did not resolve HELIX_PIMM_ROOT; run python -m helix.paths}"
: "${HELIX_EXP:?helix.paths did not resolve HELIX_EXP; run python -m helix.paths}"

# --site is NOT optional and must be explicit. `pimm submit` defaults it to
# "s3df" (pimm/cli/submit.py) and load_config takes `site or recipe.get("site")`,
# so the CLI value always wins and a recipe's own `site:` key is unreachable.
# Omitting it renders an s3df job -- singularity, wrong account -- with no
# warning at all. Derived from the site profile's container runtime so that
# "which pimm site profile matches this helix site" is answered in one place.
case "${HELIX_CONTAINER_RUNTIME:-}" in
  shifter|podman-hpc) PIMM_SITE=${HELIX_PIMM_SITE:-nersc-container} ;;
  apptainer|singularity) PIMM_SITE=${HELIX_PIMM_SITE:-s3df-container} ;;
  none) PIMM_SITE=${HELIX_PIMM_SITE:-${HELIX_SITE:-local}} ;;
  *) PIMM_SITE=${HELIX_PIMM_SITE:?cannot map container runtime to a pimm site; set HELIX_PIMM_SITE} ;;
esac

ARGS=(
  --site "$PIMM_SITE"
  # repo_root drives the shifter bind, the `cd`, the slurm log directory, the
  # submitit folder and the watchdog state. UNSET it defaults to "." -- whatever
  # directory you happened to submit from -- which with shifter binds THAT at
  # /opt/pimm/src and then runs /opt/pimm/src/scripts/train.sh. Submit from the
  # helix checkout and the job dies inside the allocation, because helix has no
  # scripts/train.sh. It also scatters slurm_logs/ into whichever repo you were
  # standing in, and helix's .gitignore does not cover that.
  --paths.repo-root "$HELIX_PIMM_ROOT"
  # exp_root defaults to {repo_root}/exp, i.e. INSIDE the pimm checkout, and
  # train.sh then passes --options save_path=$EXP_DIR which OVERRIDES the
  # save_path helix's config computed from helix.paths. helix has been bitten by
  # the equivalent before: "a relative one wrote a run INTO the repo and four
  # files were committed".
  --paths.exp-root "$HELIX_EXP"
)

# The image, from the site profile. pimm's nersc-container.yaml names ITS OWN
# base image (pimm-nersc), which has no helix, no PyWavelets and a pimm-data on
# the wrong side of the forward-model boundary -- so this override is not a
# preference, it is the difference between running helix and not.
[ -n "${HELIX_CONTAINER_IMAGE:-}" ] && ARGS+=(--container.image "$HELIX_CONTAINER_IMAGE")
[ -n "${HELIX_CONTAINER_PYTHON:-}" ] && ARGS+=(--container.interpreter "$HELIX_CONTAINER_PYTHON")

# Scheduler facts for the TRAIN job kind. A recipe may still override any of
# these on the command line (later flags win), which is how the premium and
# interactive strategies select a non-default queue.
[ -n "${HELIX_SLURM_TRAIN_ACCOUNT:-}" ]    && ARGS+=(--resources.account "$HELIX_SLURM_TRAIN_ACCOUNT")
[ -n "${HELIX_SLURM_TRAIN_QOS:-}" ]        && ARGS+=(--resources.qos "$HELIX_SLURM_TRAIN_QOS")
[ -n "${HELIX_SLURM_TRAIN_CONSTRAINT:-}" ] && ARGS+=(--resources.constraint "$HELIX_SLURM_TRAIN_CONSTRAINT")
[ -n "${HELIX_SLURM_TRAIN_PARTITION:-}" ]  && ARGS+=(--resources.partition "$HELIX_SLURM_TRAIN_PARTITION")
# pimm spells this `gpu_directive` and accepts gres | gpus-per-node.
[ -n "${HELIX_SLURM_TRAIN_GPUS_FLAG:-}" ] && \
  [ "${HELIX_SLURM_TRAIN_GPUS_FLAG}" = "gpus-per-node" ] && \
  ARGS+=(--resources.gpu-directive gpus-per-node)

# pimm loads the training config on the SUBMITTING host during preflight
# (config.py), and helix's config imports helix at module scope. So helix must be
# importable here, not only in the job.
export PYTHONPATH="$HELIX_PIMM_ROOT:$HELIX_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Make --recipe absolute. pimm's load_config does
#   recipe_path = Path(recipe); if not is_absolute(): recipe_path = ROOT / recipe_path
# where ROOT is PIMM's root -- so a relative recipe is looked for inside the pimm
# checkout, not relative to where you typed it, and helix's recipes are never
# found: "Missing launch config: <pimm>/configs/launch/nersc-preempt.yaml".
# Absolute paths ARE honoured, which is what lets the recipes live in helix at
# all, so the fix is to always pass one. Relative paths are taken from $PWD, or
# from the helix checkout when that does not resolve, so both
#   scripts/submit_helix.sh --recipe configs/launch/nersc-preempt.yaml
# from anywhere and an explicit absolute path work.
FIXED=()
_want_recipe=0
for a in "$@"; do
  if [ "$_want_recipe" = 1 ]; then
    _want_recipe=0
    if [ "${a#/}" = "$a" ]; then
      if [ -f "$PWD/$a" ]; then a="$PWD/$a"; elif [ -f "$H/$a" ]; then a="$H/$a"; fi
    fi
    FIXED+=("$a"); continue
  fi
  case "$a" in
    --recipe) _want_recipe=1; FIXED+=("$a") ;;
    --recipe=*)
      _r=${a#--recipe=}
      if [ "${_r#/}" = "$_r" ]; then
        if [ -f "$PWD/$_r" ]; then _r="$PWD/$_r"; elif [ -f "$H/$_r" ]; then _r="$H/$_r"; fi
      fi
      FIXED+=("--recipe=$_r") ;;
    *) FIXED+=("$a") ;;
  esac
done
set -- "${FIXED[@]}"

PIMM_PY=${HELIX_PIMM_PYTHON:-python3}
echo "submit_helix.sh: site=$HELIX_SITE pimm-site=$PIMM_SITE" >&2
echo "                 exp_root=$HELIX_EXP repo_root=$HELIX_PIMM_ROOT" >&2
exec "$PIMM_PY" -m pimm.cli submit "${ARGS[@]}" "$@"
