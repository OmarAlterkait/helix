#!/bin/bash
# Kernel-package quality A/B (2026-10-02/03): d768 QK-norm recipe, 4 nodes (B=16),
# 29,200 steps, WSD cooldown over the last 10%, same seed and data order; only the
# package flags differ. Results: docs/PERFORMANCE.md section 11.
#   DRY=0 scripts/study/experiments/kernel_ab.sh
source "$(dirname "$0")/../common.sh"
BASE="model.d=768 model.heads=12 model.qk_norm=True scheduler.type=WSDCooldownLR scheduler.stable_frac=0.9 scheduler.floor=1e-3 hooks.CoeffFMEvaluator.mask_ratio=0.75"
RUN="LR=4.4e-3,STEPS=29200,WARMUP=1225,EVERY=900,CKPT_EVERY=7300,SEED=0"
declare -A ARMS=(
  [ab_ref]=""
  [ab_varlen]="model.varlen=True model.fused_qk=True"
  [ab_pkg]="model.varlen=True model.fused_qk=True model.bf16_params=True optimizer.type=FlatAdamW"
  [ab_pkg_f32s]="model.varlen=True model.fused_qk=True model.bf16_params=True model.fp32_stream=True optimizer.type=FlatAdamW"
)
for tag in ab_ref ab_varlen ab_pkg ab_pkg_f32s; do
  study_sbatch 4 "$tag" slot.sbatch "TAG=$tag,$RUN,EXTRA=$BASE ${ARMS[$tag]}" --time-min=03:00:00
done
