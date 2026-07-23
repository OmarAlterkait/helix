SHARD=$1; N=4; i=0
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
OUT=results_shard$SHARD.jsonl; : > $OUT
run() {  # kind ckpt mnll layer_or_budget tag
  if [ $((i % N)) -eq $SHARD ]; then
    echo ">>> shard$SHARD: $1 $5" >&2
    case $1 in
      aw) python3 pb_aw.py    --arm nll --ckpt $2 --d 512 --heads 8 --mnll $3 --layer $4 --steps 3000 --tag "$5" 2>/dev/null | grep "^AWFINAL" >> $OUT ;;
      pb) python3 pb_probe.py --arm nll --ckpt $2 --d 512 --heads 8 --mnll $3 --nox --layer $4 --steps 3000 --tag "$5" 2>/dev/null | grep "^PROBE"   >> $OUT ;;
      dc) python3 deconv_ft.py --arm ckpt --ckpt $2 --budget $4 --tag "$5" 2>/dev/null | grep "^FT" >> $OUT ;;
    esac
  fi; i=$((i+1))
}
for L in 4 8 12; do
  run aw ckpt_mae_mse_s0_snap50000.pt  0 $L mse_50k_L$L
  run aw ckpt_mae_mse_s0_snap150000.pt 0 $L mse_150k_L$L
  run aw ckpt_mae_nll_s0_snap50000.pt  1 $L nll_50k_L$L
  run aw ckpt_mae_nll_s0_snap150000.pt 1 $L nll_150k_L$L
done
for L in 8 12; do
  run pb ckpt_mae_mse_s0_snap150000.pt 0 $L mse_150k_L$L
  run pb ckpt_mae_nll_s0_snap150000.pt 1 $L nll_150k_L$L
done
run dc ckpt_mae_mse_s0.pt 0 240  mse_240
run dc ckpt_mae_mse_s0.pt 0 2000 mse_2000
run dc ckpt_mae_nll_s0.pt 1 240  nll_240
run dc ckpt_mae_nll_s0.pt 1 2000 nll_2000
echo "SHARD $SHARD DONE" >&2
