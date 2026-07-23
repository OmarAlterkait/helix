SHARD=$1; N=4; i=0
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
run() {  # probe ckpt mnll layer tag
  if [ $((i % N)) -eq $SHARD ]; then
    echo ">>> $1 $5 layer=$4"
    if [ "$1" = "aw" ]; then
      python3 pb_aw.py    --arm nll --ckpt "$2" --d 512 --heads 8 --mnll $3 --layer $4 --steps 4000 --tag "$5" 2>&1 | grep -E "^AW " | tail -1
    else
      python3 pb_probe.py --arm nll --ckpt "$2" --d 512 --heads 8 --mnll $3 --nox --layer $4 --steps 4000 --tag "$5" 2>&1 | grep -E "^PROBE" | tail -1
    fi
  fi; i=$((i+1))
}
# 3D probe (pb_aw): layers x {MSE,NLL} x {50k,150k}
for L in 4 8 12; do
  run aw ckpt_mae_mse_s0_snap50000.pt  0 $L mse_50k
  run aw ckpt_mae_mse_s0_snap150000.pt 0 $L mse_150k
  run aw ckpt_mae_nll_s0_snap50000.pt  1 $L nll_50k
  run aw ckpt_mae_nll_s0_snap150000.pt 1 $L nll_150k
done
# group content (pb_probe D/F/B1): final ckpts, 2 layers
for L in 8 12; do
  run pb ckpt_mae_mse_s0_snap150000.pt 0 $L mse_150k
  run pb ckpt_mae_nll_s0_snap150000.pt 1 $L nll_150k
done
echo "SHARD $SHARD DONE"
