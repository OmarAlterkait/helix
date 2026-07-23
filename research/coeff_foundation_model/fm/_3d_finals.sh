SHARD=$1; N=4; i=0
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
OUT=res3d_shard$SHARD.jsonl; : > $OUT
run() {  # ckpt mnll blocks layer tag
  if [ $((i % N)) -eq $SHARD ]; then
    echo ">>> $5" >&2
    python3 pb_aw.py --arm nll --ckpt $1 --d 512 --heads 8 --blocks $3 --mnll $2 --layer $4 --steps 3000 --tag "$5" 2>/dev/null | grep "^AWFINAL" >> $OUT
  fi; i=$((i+1))
}
run ckpt_nll_long_snap300000.pt 1 12 8  nll_long_L8
run ckpt_nll_long_snap300000.pt 1 12 12 nll_long_L12
run ckpt_nll_data_snap300000.pt 1 12 8  nll_data_L8
run ckpt_nll_data_snap300000.pt 1 12 12 nll_data_L12
run ckpt_nll_deep_snap300000.pt 1 24 12 nll_deep_L12
run ckpt_nll_deep_snap300000.pt 1 24 24 nll_deep_L24
run ckpt_mse_long_snap300000.pt 0 12 8  mse_long_L8
run ckpt_mse_long_snap300000.pt 0 12 12 mse_long_L12
echo "SHARD$SHARD DONE" >&2
