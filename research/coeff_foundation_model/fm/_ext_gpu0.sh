cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm
for ro in "run_0027575767 20000" "run_0027575769 60000" "run_0027587651 100000" "run_0027587653 140000" "run_0027651460 180000"; do
  set -- $ro; echo ">>> $1 -> off $2"; python -u cache_ext.py --split $1 --start 0 --n 20000 --out_start $2
done; echo "GPU0_ALLDONE"
