cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
for ro in "run_0027575768 40000" "run_0027587649 80000" "run_0027587652 120000" "run_0027587654 160000"; do
  set -- $ro; echo ">>> $1 -> off $2"; python -u cache_ext.py --split $1 --start 0 --n 20000 --out_start $2
done; echo "GPU1_ALLDONE"
