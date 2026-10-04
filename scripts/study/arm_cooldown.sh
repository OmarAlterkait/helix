#!/bin/bash
# A WSD cooldown branched from a checkpoint (weights only, fresh optimizer):
#   arm_cooldown.sh <TAG> <CKPT> <STEPS> <LR> <NNODES> <NPROC> <MASTER_ADDR> <PORT>
# EXTRA as in arm_steps.sh; D sets width and heads (head dim 64).
set -euo pipefail
TAG=$1; CKPT=$2; STEPS=$3; LR=$4; NNODES=$5; NPROC=$6; MADDR=$7; MPORT=$8
OUT=${OUT:?}; SEED=${SEED:-0}
CFG=${CONFIG:-$HELIX_ROOT/configs/pimm/coeff_fm_train_8run.py}
[ -f "$CKPT" ] || { echo "[$TAG] no checkpoint at $CKPT" >&2; exit 2; }
B=$(( NNODES * NPROC )); NEED=$(( STEPS * B ))
AVAIL=${TRAIN_EVENTS:-$(python -c "from pimm.utils.config import Config; print(Config.fromfile('$CFG').N_TRAIN_EVENTS)" 2>/dev/null | tail -1)}
EPOCH=$(( (NEED + AVAIL - 1) / AVAIL )); MAXLEN=$(( NEED / EPOCH ))
EVERY=$(( STEPS / ${CD_EVALS:-15} > 0 ? STEPS / ${CD_EVALS:-15} : 1 ))
[ "${SLURM_NODEID:-0}" = "0" ] && [ -z "${CHECK:-}" ] && \
  echo "[$TAG] cooldown B=$B steps=$STEPS lr=$LR seed=$SEED from $CKPT epoch=$EPOCH max_len=$MAXLEN"
OPTS=(save_path="$OUT/$TAG" weight="$CKPT" resume=False epoch="$EPOCH" seed="$SEED" optimizer.lr="$LR"
      scheduler.type=WSDCooldownLR scheduler.warmup=0 scheduler.floor=1e-3
      data.train.max_len="$MAXLEN" data.val.max_len=256 data.test.max_len=256
      hooks.CheckpointSaver.save_freq="$STEPS" hooks.WeightEMA.save_freq="$STEPS"
      hooks.CoeffFMEvaluator.every_n_steps="$EVERY")
if [ -n "${D:-}" ]; then OPTS+=(model.d="$D" model.heads="$(( D / 64 ))"); fi
if [ -n "${EXTRA:-}" ]; then read -r -a _x <<< "$EXTRA"; OPTS+=("${_x[@]}"); fi
if [ -n "${CHECK:-}" ]; then printf '%s\n' "${OPTS[@]}"; exit 0; fi
exec python -m torch.distributed.run --nnodes="$NNODES" --nproc_per_node="$NPROC" \
  --node_rank="${SLURM_NODEID:-0}" --master_addr="$MADDR" --master_port="$MPORT" \
  -m pimm.train --config-file "$CFG" --options "${OPTS[@]}"
