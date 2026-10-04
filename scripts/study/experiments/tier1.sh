#!/bin/bash
# Tier 1, width at equal compute (2026-09-28): ~9,500 s of training per width on
# 16 nodes (B=64), QK-norm, LR 4.4e-3, warmup 500*sqrt(d/128), 10% in-run cooldown.
# Measured: d768 2.620, d1024 2.579, d1536 2.579, d2048 2.616 (docs/SCIENCE.md s9).
# d1536 and up need activation checkpointing on a 40 GB A100. These ran 2-3 h and
# three of the four TIMEOUT-ed under a 2 h floor; resubmitting resumes (RESUME=auto).
#   DRY=0 scripts/study/experiments/tier1.sh
source "$(dirname "$0")/../common.sh"
tier1() {  # tier1 <d> <steps> <warmup> [extra]
  local d=$1 st=$2 wu=$3 x=${4:-}
  local w="model.d=$d model.heads=$(( d / 64 )) model.qk_norm=True $x scheduler.type=WSDCooldownLR scheduler.stable_frac=0.9 scheduler.floor=1e-3 hooks.CoeffFMEvaluator.mask_ratio=0.75"
  study_sbatch 16 "t1_d$d" slot.sbatch "TAG=t1_d$d,LR=4.4e-3,WARMUP=$wu,STEPS=$st,EVERY=$(( st / 30 )),CKPT_EVERY=$(( st / 20 )),EXTRA=$w" --time-min=03:00:00
}
tier1 768 47000 1225
tier1 1024 33200 1414
tier1 1536 17700 1732 model.act_ckpt=True
tier1 2048 11500 2000 model.act_ckpt=True
