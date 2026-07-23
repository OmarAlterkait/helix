cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
echo "### DECONV-FT label-efficiency (scratch vs MSE vs NLL, budgets 240/2000)"
for arm in "scratch" "ckpt --ckpt ckpt_mae_mse_s0.pt" "ckpt --ckpt ckpt_mae_nll_s0.pt"; do
  for b in 240 2000; do
    echo ">>> deconv_ft arm='$arm' budget=$b"
    python3 deconv_ft.py --arm $arm --budget $b --tag probe 2>&1 | grep -E "^FT |^arm|Error|Traceback" | tail -6 || echo "  FAILED"
  done
done
echo "### GROUP PROBES at final ckpt (pb_aw 3D + pb_probe D/F/B1), MSE vs NLL"
echo ">>> pb_aw MSE";  python3 pb_aw.py    --arm nll --ckpt ckpt_mae_mse_s0.pt --d 512 --heads 8 --mnll 0 --tag mse 2>&1 | grep -E "^AW |within|Error|Traceback" | tail -4 || echo FAILED
echo ">>> pb_aw NLL";  python3 pb_aw.py    --arm nll --ckpt ckpt_mae_nll_s0.pt --d 512 --heads 8 --mnll 1 --tag nll 2>&1 | grep -E "^AW |within|Error|Traceback" | tail -4 || echo FAILED
echo ">>> pb_probe MSE"; python3 pb_probe.py --arm nll --ckpt ckpt_mae_mse_s0.pt --d 512 --heads 8 --mnll 0 --nox --tag mse 2>&1 | grep -E "^PROBE|Error|Traceback" | tail -4 || echo FAILED
echo ">>> pb_probe NLL"; python3 pb_probe.py --arm nll --ckpt ckpt_mae_nll_s0.pt --d 512 --heads 8 --mnll 1 --nox --tag nll 2>&1 | grep -E "^PROBE|Error|Traceback" | tail -4 || echo FAILED
echo "### PROBES DONE"
