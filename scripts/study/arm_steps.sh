#!/bin/bash
# One training arm, inside the container on every node:
#   arm_steps.sh <TAG> <LR> <STEPS> <NNODES> <NPROC> <MASTER_ADDR> <PORT>
# Global batch = NNODES*NPROC (one event per GPU). Writes $OUT/$TAG.
#
#   WARMUP      fixed warmup steps (default 4% of STEPS; prefer 500*sqrt(d/128))
#   EVERY       eval cadence (default STEPS/20); CKPT_EVERY checkpoint cadence (STEPS/4)
#   SEED        seed (default 0) -- it used to be a literal, so "replicates" were identical
#   RESUME=auto continue from the latest complete checkpoint under $OUT/$TAG/model, so a
#               requeue or resubmit never restarts and overwrites a longer run.
#               pimm reads the checkpoint from `weight`; resume=True alone dies with
#               "No weight found at: None".
#   BRANCH_FROM start from a checkpoint's weights with a fresh optimizer and schedule
#   EPOCH/MAXLEN extend a stable run: keep MAXLEN, raise EPOCH, so resume maps the saved
#               step onto the same (epoch, iter)
#   EXTRA       whitespace-separated key=value overrides. A typo in a model key is
#               refused by build_coeff_fm; a non-model key that does not exist is added
#               silently by pimm, so check it.
#   CHECK=1     print the options and exit
set -euo pipefail
TAG=$1; LR=$2; STEPS=$3; NNODES=$4; NPROC=$5; MADDR=$6; MPORT=$7
OUT=${OUT:?}; SEED=${SEED:-0}; EVERY=${EVERY:-0}; CKPT_EVERY=${CKPT_EVERY:-0}; RESUME=${RESUME:-auto}
CFG=${CONFIG:-$HELIX_ROOT/configs/pimm/coeff_fm_train_8run.py}
B=$(( NNODES * NPROC )); NEED=$(( STEPS * B ))
[ -n "${WARMUP:-}" ] || WARMUP=$(( STEPS * 4 / 100 > 0 ? STEPS * 4 / 100 : 1 ))
[ "$EVERY" -gt 0 ] || EVERY=$(( STEPS / 20 ))
[ "$CKPT_EVERY" -gt 0 ] || CKPT_EVERY=$(( STEPS / 4 ))
# The training-event count is declared once, in the config; read it, never retype it.
AVAIL=${TRAIN_EVENTS:-$(python -c "from pimm.utils.config import Config; print(Config.fromfile('$CFG').N_TRAIN_EVENTS)" 2>/dev/null | tail -1)}
[ -n "${EPOCH:-}" ]  || EPOCH=$(( (NEED + AVAIL - 1) / AVAIL ))
[ -n "${MAXLEN:-}" ] || MAXLEN=$(( NEED / EPOCH ))

RESUME_OPT=()
if [ "$RESUME" = auto ] || [ "$RESUME" = True ]; then
  WEIGHT=$(python -m pimm.utils.path latest-checkpoint "$OUT/$TAG/model" 2>/dev/null || true)
  if [ -n "$WEIGHT" ]; then RESUME=True; RESUME_OPT=(resume=True "weight=$WEIGHT")
  else [ "$RESUME" = True ] && { echo "[$TAG] resume=True but no checkpoint under $OUT/$TAG/model" >&2; exit 2; }; RESUME=False; fi
fi
if [ -n "${BRANCH_FROM:-}" ]; then RESUME=False; RESUME_OPT=(resume=False "weight=$BRANCH_FROM"); fi

[ "${SLURM_NODEID:-0}" = "0" ] && [ -z "${CHECK:-}" ] && \
  echo "[$TAG] B=$B steps=$STEPS lr=$LR events=$NEED epoch=$EPOCH max_len=$MAXLEN warmup=$WARMUP resume=$RESUME"
OPTS=(save_path="$OUT/$TAG" "${RESUME_OPT[@]}" epoch="$EPOCH" seed="$SEED" optimizer.lr="$LR"
      scheduler.warmup="$WARMUP" data.train.max_len="$MAXLEN" data.val.max_len=256 data.test.max_len=256
      hooks.CheckpointSaver.save_freq="$CKPT_EVERY" hooks.WeightEMA.save_freq="$CKPT_EVERY"
      hooks.CoeffFMEvaluator.every_n_steps="$EVERY")
if [ -n "${EXTRA:-}" ]; then read -r -a _x <<< "$EXTRA"; OPTS+=("${_x[@]}"); fi
if [ -n "${CHECK:-}" ]; then printf '%s\n' "${OPTS[@]}"; exit 0; fi
exec python -m torch.distributed.run --nnodes="$NNODES" --nproc_per_node="$NPROC" \
  --node_rank="${SLURM_NODEID:-0}" --master_addr="$MADDR" --master_port="$MPORT" \
  -m pimm.train --config-file "$CFG" --options "${OPTS[@]}"
