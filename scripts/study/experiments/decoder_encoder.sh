#!/bin/bash
# Decoder budget and mask units (2026-10-05): the kernel_ab.sh reference recipe
# (d768, B=16, 29,200 steps, seed 0); only the named option differs. The pw16
# references are ab_ref (seed 0) and res_pw16_s1 (seed 1). Scored with
# scripts/eval_resolution.py. Arms run interactively are listed for the record
# and skipped here (RUN_INTERACTIVE=1 submits them too).
#   DRY=0 scripts/study/experiments/decoder_encoder.sh
source "$(dirname "$0")/../common.sh"
COMMON="model.d=768 model.heads=12 model.qk_norm=True scheduler.type=WSDCooldownLR scheduler.stable_frac=0.9 scheduler.floor=1e-3 hooks.CoeffFMEvaluator.mask_ratio=0.75"
V="$HELIX_ROOT/configs/pimm/variants"
#  tag | where | options | config
RUNS=(
  "res_dec3|interactive|model.dec_frac=0.3333"                 # uniform partial decode
  "res_dech|interactive|model.d_dec=384"                       # half-width decoder
  "res_decb|preempt||$V/coeff_fm_8run_decb.py"                # all A4/D4, a third of D3/D2, 1/p weights
  "res_mm2|preempt|model.n_masks=2"                            # two complementary masks per event
  "res_loc|interactive|model.mask_mode=location"               # every band of a location masked together
)
for r in "${RUNS[@]}"; do
  IFS='|' read -r tag where opt cfg <<< "$r"
  [ "$where" = interactive ] && [ -z "${RUN_INTERACTIVE:-}" ] && continue
  exp="TAG=$tag,LR=4.4e-3,STEPS=29200,WARMUP=1225,EVERY=900,CKPT_EVERY=7300,SEED=0,EXTRA=$COMMON $opt"
  [ -n "$cfg" ] && exp="$exp,CONFIG=$cfg"
  study_sbatch 4 "$tag" slot.sbatch "$exp" --time-min=03:00:00
done
