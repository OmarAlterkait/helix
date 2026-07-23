cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
echo "### DECONV-FT pretrained arms (MSE, NLL) — scratch already have (240=32.6, 2000=45.2)"
for ck in ckpt_mae_mse_s0.pt ckpt_mae_nll_s0.pt; do
  for b in 240 2000; do
    echo ">>> $ck budget=$b"
    python3 deconv_ft.py --arm ckpt --ckpt $ck --budget $b --tag probe 2>&1 | grep -E "^FT |Error|Traceback" | tail -4
  done
done
echo "### DECONV DONE"
