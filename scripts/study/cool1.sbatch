#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --time-min=02:00:00
#SBATCH --requeue
#SBATCH --gpus-per-node=4
# One 10% (1-sqrt p) cooldown branched from a checkpoint, at the batch of the nodes
# given. --export: TAG SRC(weights.pth) LR STEPS [D CEXTRA]. CEXTRA carries the run's
# model options (qk_norm must match or its weights do not load). The 5% re-warm is
# there because a cold AdamW at full LR broke two B=256 cooldowns. A .done marker
# stops a requeue from redoing a finished cooldown.
set -uo pipefail
H=${HELIX_ROOT:?}; L=${HELIX_STUDY_DIR:?}
cd "$H" || exit 2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mapfile -t N < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[ -f "$L/cooldown/$TAG/.done" ] && { echo "$TAG done already"; exit 0; }
srun --nodes=${#N[@]} --ntasks-per-node=1 --gpus-per-node=4 \
  env HELIX_INTERACTIVE=0 OUT="$L/cooldown" CD_EVALS=5 D="${D:-}" \
      EXTRA="model.compile_blocks=False hooks.CoeffFMEvaluator.mask_ratio=0.75 scheduler.warmup=$(( STEPS / 20 )) ${CEXTRA:-}" \
      "$H/scripts/helix_run.sh" bash "$H/scripts/study/arm_cooldown.sh" "$TAG" "$SRC" "$STEPS" "$LR" "${#N[@]}" 4 "${N[0]}" $(( 20000 + SLURM_JOB_ID % 9000 )) \
  >> "$L/${TAG}-${SLURM_JOB_ID}.log" 2>&1 && touch "$L/cooldown/$TAG/.done"
