cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
C="--heads 4 --blocks 12 --layer 12 --key teacher --seeds 4 --out probe_3d_jepa.jsonl"
# plain JEPA final (120k) WITH random-init floor for heads=4 arch
python3 probe_3d_rigor.py --ckpt ckpt_sc_jepa_d512.pt          --arms trained,random --tag jepa_plain@120k $C 2>&1 | grep SUMMARY
python3 probe_3d_rigor.py --ckpt ckpt_sc_jepa_d512_snap100000.pt --arms trained      --tag jepa_plain@100k $C 2>&1 | grep SUMMARY
python3 probe_3d_rigor.py --ckpt ckpt_sc_jepa_sig_d512.pt        --arms trained      --tag jepa_sig@60k   $C 2>&1 | grep SUMMARY
echo "JEPA 3D PROBE DONE"
