#!/bin/bash
# One denoiser training run on ONE node of an interactive allocation (B=4):
#   denoise_node.sh <jobid> <node> <tag> <port> [train_denoise.py args...]
# Output: $HELIX_STUDY_DIR/denoise/<tag>, log $HELIX_STUDY_DIR/denoise/<tag>.log
set -euo pipefail
J=$1; N=$2; TAG=$3; PORT=$4; shift 4
HELIX=$(cd "$(dirname "$0")/../.." && pwd)
W=${HELIX_STUDY_DIR:-/pscratch/sd/o/oalter/helix_work}
mkdir -p $W/denoise
cd $HELIX
exec srun --jobid=$J -N1 -n1 -w $N --overlap --gpus-per-node=4 env HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE HELIX_ROOT=$HELIX \
  scripts/helix_run.sh python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$PORT \
  scripts/train_denoise.py --truth-root ${DENOISE_TRUTH:-/pscratch/sd/o/oalter/denoise_truth} --out $W/denoise/$TAG "$@" \
  > $W/denoise/$TAG.log 2>&1
