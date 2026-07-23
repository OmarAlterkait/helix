G=$1
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
case $G in
  0) CK=ckpt_nll_long_snap300000.pt M=1 B=12 L=12 T=nll_long ;;
  1) CK=ckpt_nll_data_snap300000.pt M=1 B=12 L=12 T=nll_data ;;
  2) CK=ckpt_nll_deep_snap300000.pt M=1 B=24 L=24 T=nll_deep ;;
  3) CK=ckpt_mse_long_snap300000.pt M=0 B=12 L=12 T=mse_long ;;
esac
echo ">>> $T (blocks=$B layer=$L)" >&2
python3 pb_aw.py --arm nll --ckpt $CK --d 512 --heads 8 --blocks $B --mnll $M --layer $L --steps 1500 --tag $T 2>/dev/null | grep "^AWFINAL" > res3df_g$G.jsonl
echo "$T done" >&2
