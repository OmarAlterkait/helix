#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --time-min=02:00:00
#SBATCH --requeue
#SBATCH --gpus-per-node=4
# One training arm, sized to fill its preempt slot. Submit through common.sh's
# study_sbatch, which supplies account, constraint, QOS and output path from the
# site profile. Parameterised entirely by --export:
#   TAG LR STEPS [WARMUP EVERY CKPT_EVERY SEED CONFIG EXTRA EXTRA_FILE BRANCH_FROM EPOCH MAXLEN]
#
# A preempt slot is billed at its --time-min whether or not the work fills it, so
# STEPS should fill it (slot.py). --time-min is a floor, not the grant: Slurm may
# give anything between it and --time, and a TIMEOUT does NOT requeue -- give
# slower arms a larger --time-min (e.g. 03:00:00) rather than trusting the floor.
# RESUME=auto in arm_steps.sh makes a requeued or resubmitted arm continue.
set -uo pipefail
H=${HELIX_ROOT:?HELIX_ROOT unset: submit through scripts/study/common.sh}
L=${HELIX_STUDY_DIR:?HELIX_STUDY_DIR unset: submit through scripts/study/common.sh}
cd "$H" || exit 2          # never the submission dir: a local file can shadow a stdlib module
[ -n "${EXTRA_FILE:-}" ] && EXTRA="${EXTRA:-} $(cat "$EXTRA_FILE")"   # options containing commas
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1
mapfile -t N < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NN=${#N[@]}
PORT=$(( 20000 + SLURM_JOB_ID % 9000 ))          # below the ephemeral range
echo "=== $SLURM_JOB_NAME : $NN nodes, B=$((NN*4)), ${STEPS} steps, $(date)"
srun --nodes="$NN" --ntasks-per-node=1 --gpus-per-node=4 \
  env HELIX_INTERACTIVE=0 OUT="${OUT:-$L/sweep}" EVERY="${EVERY:-500}" CKPT_EVERY="${CKPT_EVERY:-0}" \
      WARMUP="${WARMUP:-}" SEED="${SEED:-0}" CONFIG="${CONFIG:-}" EXTRA="${EXTRA:-}" \
      BRANCH_FROM="${BRANCH_FROM:-}" EPOCH="${EPOCH:-}" MAXLEN="${MAXLEN:-}" \
      "$H/scripts/helix_run.sh" bash "$H/scripts/study/arm_steps.sh" "$TAG" "$LR" "$STEPS" "$NN" 4 "${N[0]}" "$PORT" \
  >> "$L/${TAG}-${SLURM_JOB_ID}.log" 2>&1
rc=$?
echo "=== $SLURM_JOB_NAME exit=$rc $(date)"
exit $rc
