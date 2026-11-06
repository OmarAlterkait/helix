#!/bin/bash
# M12 convergence: the k = 1 run's kept checkpoints (steps 2000 / 4000 / 6000) vs the k = 0 control's, paired (c13).
set -uo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd); D=/global/cfs/cdirs/m5238/users/oalter/helix_runs/denoise; M=/global/cfs/cdirs/m5238/users/oalter/helix_runs/m7tests
cd $H
J=""; until [ -n "$J" ]; do
  J=$(timeout 21600 salloc --no-shell --qos=interactive --account=m5238_g --constraint="gpu&hbm80g" --nodes=1 --gpus-per-node=4 \
      --time=02:00:00 --job-name=hx-m12conv 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
  [ -z "$J" ] && sleep 30
done
echo "[m12conv] got $J at $(date)"
S=(srun --jobid=$J -N1 -n1 --overlap -c 16 --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE)
"${S[@]}" CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/c13_eval_models.py --bins 1000,4000,16000,64000,256000 \
  --ckpt w0_s2000=$D/M12w0/step2000.pt --ckpt w1_s2000=$D/M12w1/step2000.pt --ref w0_s2000 --out $M/c13_m12_s2000.json > $M/c13_m12_s2000.log 2>&1 &
"${S[@]}" CUDA_VISIBLE_DEVICES=1 scripts/helix_run.sh python baselines/uboone_sp/causes/c13_eval_models.py --bins 1000,4000,16000,64000,256000 \
  --ckpt w0_s4000=$D/M12w0/step4000.pt --ckpt w1_s4000=$D/M12w1/step4000.pt --ref w0_s4000 --out $M/c13_m12_s4000.json > $M/c13_m12_s4000.log 2>&1 &
wait
scancel $J; echo "[m12conv] released $J at $(date)"
