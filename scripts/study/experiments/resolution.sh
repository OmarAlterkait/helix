#!/bin/bash
# Patch size vs resolution (2026-10-03): the kernel_ab.sh reference recipe, only the
# patch geometry differs. The pw16 seed-0 reference is kernel_ab.sh's ab_ref.
# Score with scripts/dump_resolution_truth.py + scripts/eval_resolution.py.
# Results: docs/SCIENCE.md section 10.
#   DRY=0 scripts/study/experiments/resolution.sh
source "$(dirname "$0")/../common.sh"
V="$HELIX_ROOT/configs/pimm/variants"
COMMON="model.qk_norm=True scheduler.type=WSDCooldownLR scheduler.stable_frac=0.9 scheduler.floor=1e-3 hooks.CoeffFMEvaluator.mask_ratio=0.75"
D768="model.d=768 model.heads=12"; D1024="model.d=1024 model.heads=16"
#  tag | seed | config | width | warmup
RUNS=(
  "res_pw16_s1|1||$D768|1225"
  "res_pw8|0|$V/coeff_fm_8run_pw8.py|$D768|1225"
  "res_pw32|0|$V/coeff_fm_8run_pw32.py|$D768|1225"
  "res_pw32_s1|1|$V/coeff_fm_8run_pw32.py|$D768|1225"
  "res_pt4|0|$V/coeff_fm_8run_pt4.py|$D768|1225"
  "res_pt16|0|$V/coeff_fm_8run_pt16.py|$D768|1225"
  "res_pw32_d1024|0|$V/coeff_fm_8run_pw32.py|$D1024|1414"
)
for r in "${RUNS[@]}"; do
  IFS='|' read -r tag seed cfg width wu <<< "$r"
  exp="TAG=$tag,LR=4.4e-3,STEPS=29200,WARMUP=$wu,EVERY=900,CKPT_EVERY=7300,SEED=$seed,EXTRA=$width $COMMON"
  [ -n "$cfg" ] && exp="$exp,CONFIG=$cfg"
  study_sbatch 4 "$tag" slot.sbatch "$exp" --time-min=03:00:00   # 2 h floor TIMEOUTs the slower patch sizes
done
