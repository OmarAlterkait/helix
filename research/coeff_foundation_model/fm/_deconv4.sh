G=$1
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
case $G in
  0) CK=ckpt_mae_mse_s0.pt B=240 ;;
  1) CK=ckpt_mae_mse_s0.pt B=2000 ;;
  2) CK=ckpt_mae_nll_s0.pt B=240 ;;
  3) CK=ckpt_mae_nll_s0.pt B=2000 ;;
esac
echo ">>> deconv $CK budget=$B on gpu$G" >&2
python3 deconv_ft.py --arm ckpt --ckpt $CK --budget $B --tag probe 2>/dev/null | grep "^FT" > deconv_g$G.jsonl
echo "gpu$G done" >&2
