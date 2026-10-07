#!/bin/bash
# One denoiser training run across EVERY node of an interactive allocation (B = 4 x nodes):
#   denoise_multi.sh <jobid> <tag> <port> [train_denoise.py args...]
# Output: $HELIX_STUDY_DIR/denoise/<tag>; log $HELIX_STUDY_DIR/denoise/<tag>-<jobid>.log. Resumes from last.pt.
set -euo pipefail
J=$1; TAG=$2; PORT=$3; shift 3
HELIX=$(cd "$(dirname "$0")/../.." && pwd)
W=${HELIX_STUDY_DIR:-/pscratch/sd/o/oalter/helix_work}
NN=$(squeue -j $J -h -o %D); HEAD=$(scontrol show hostnames $(squeue -j $J -h -o %N) | head -1)
mkdir -p $W/denoise; cd $HELIX
ARGS=$(printf '%q ' "$@")
exec srun --jobid=$J --nodes=$NN --ntasks-per-node=1 --gpus-per-node=4 --overlap env HELIX_INTERACTIVE=0 \
  HDF5_USE_FILE_LOCKING=FALSE HELIX_ROOT=$HELIX TORCH_NCCL_ASYNC_ERROR_HANDLING=1 scripts/helix_run.sh bash -c \
  "python -m torch.distributed.run --nnodes=$NN --nproc_per_node=4 --node_rank=\$SLURM_NODEID --master_addr=$HEAD --master_port=$PORT \
   scripts/train_denoise.py --truth-root ${DENOISE_TRUTH:-/pscratch/sd/o/oalter/denoise_truth} --out $W/denoise/$TAG $ARGS" \
  > $W/denoise/$TAG-$J.log 2>&1
