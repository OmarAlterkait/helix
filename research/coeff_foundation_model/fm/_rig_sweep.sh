cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
run() { python3 probe_3d_rigor.py --ckpt $1 --layer $2 --arms trained --tag $3 --seeds 4 --out probe_3d_scaling.jsonl 2>&1 | grep -E "SUMMARY|seed0|seed3"; }
for step in 150000 300000; do
  run ckpt_nll_long_snap${step}.pt 12 nll_long@${step}
  run ckpt_nll_data_snap${step}.pt 12 nll_data@${step}
  run ckpt_nll_deep_snap${step}.pt 24 nll_deep@${step}
  run ckpt_mse_long_snap${step}.pt 12 mse_long@${step}
done
echo "SCALING 3D SWEEP DONE"
