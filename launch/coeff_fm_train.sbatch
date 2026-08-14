#!/bin/bash
# Coefficient-FM pretraining: 25 epochs / 118,962 steps on 4 A100s (~7 h).
#
#   sbatch /sdf/group/neutrino/omara/helix-extraction/launch/coeff_fm_train.sbatch
#
# This exists because `pimm submit` cannot run on this cluster. It needs ONE
# environment with both a working sbatch and the ability to import the training
# config (finalize_config does Config.fromfile and SystemExits if it cannot),
# and those are mutually exclusive here: the image has every import but cannot
# be a Slurm client (its sbatch cannot resolve the configless controller, and
# the layers below that end at LDAP/SSSD user resolution), while the host has
# sbatch but cannot build pimm's CUDA-compiled deps (torch_scatter, and behind
# it spconv-cu126 / flash-attn) with no CUDA toolkit on a login node.
#
# What `pimm submit` would have given us is narrow: requeue on preemption, and
# resume. Slurm provides the first directly (--requeue), and train.sh already
# owns the second. So the body below is the command pimm's own `--dry-run`
# rendered, verbatim, plus a resume guard. Everything downstream — the code
# snapshot, the config dump, checkpoint discovery, trainer-state restore — is
# still pimm's, unchanged.
#
# The recipe in launch/coeff_fm_train.yaml stays as the source of these values
# and remains usable the moment a host-side pimm environment exists.

#SBATCH --job-name=coeff-fm-train
#SBATCH --account=mli:cider-ml
#SBATCH --partition=ampere
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
# normal QoS, NOT preemptable. That choice is forced by pimm, not preference:
# RESUME IS BROKEN in this build, so an evicted run cannot pick up where it left
# off. checkpoints.py:940 loads a legacy checkpoint with
# `map_location=lambda storage, loc: storage.cuda()`, which moves the saved RNG
# ByteTensor onto the GPU; torch.set_rng_state then rejects it with
# "TypeError: RNG state must be a torch.ByteTensor". The other format is no help:
# DCP's save planner gathers on the NCCL-only default group and dies at the first
# save (see below). Measured both.
#
# Saving is fine, so a 7 h run inside a 12 h non-preemptable slot simply never
# needs to resume. Requeue is kept only for a node failure, and the guard below
# then does a WARM START rather than a resume.
#SBATCH --requeue
# Append, don't truncate: a requeued attempt reuses %j, and the default would
# erase the log of the attempt that got evicted.
#SBATCH --open-mode=append
#SBATCH --output=/sdf/data/neutrino/omara/exp/logs/coeff-fm-train-%j.out

set -uo pipefail

IMAGE=/sdf/data/neutrino/youngsam/images/pimm-latest.sif
PIMM=/sdf/group/neutrino/omara/pimm-fm
EXP_ROOT=/sdf/data/neutrino/omara/exp
RUN_NAME=coeff-fm-train
# train.sh builds EXP_DIR as ${EXP_ROOT}/${DATASET}/${EXP_NAME}, and DATASET is
# the directory part of the config ref (`helix/coeff_fm_train` -> `helix`).
EXP_DIR=${EXP_ROOT}/helix/${RUN_NAME}

mkdir -p "${EXP_ROOT}/logs"

# Warm start from the last weights if this job was requeued after a node
# failure. Deliberately NOT `-r true`: pimm's legacy resume path crashes
# restoring RNG state (see above), so a real resume would just fail. A warm start
# keeps the learned weights and resets optimizer moments and the step counter —
# survivable on a flat-LR WSD phase, and strictly better than starting over.
#
# It must not be `-r true` for a second reason: train.sh's resume branch exits 2
# when it finds no checkpoint, so that flag would kill the very first attempt.
WARM_ARGS=""
if [ -e "${EXP_DIR}/model/model_last.pth" ]; then
  WARM_ARGS="-w ${EXP_DIR}/model/model_last.pth"
  echo "[wrapper] requeued: warm-starting from ${EXP_DIR}/model/model_last.pth"
  echo "[wrapper] NOTE optimizer state and step counter reset (pimm resume is broken)"
else
  echo "[wrapper] no checkpoint under ${EXP_DIR}/model — starting from scratch"
fi

echo "[wrapper] job ${SLURM_JOB_ID:-?} restart count ${SLURM_RESTART_COUNT:-0}"

# `-c helix/coeff_fm_train` resolves to <pimm>/configs/helix/coeff_fm_train.py,
# a symlink into helix's configs/pimm. train.sh has no absolute-config path, so
# the config must be reachable from pimm's configs/; the symlink keeps helix the
# single source of truth. It points into /sdf, which is bound below.
#
# checkpoint_format=legacy is NOT a preference. pimm defaults a multi-GPU run to
# DCP, whose save planner calls dist.gather_object on the default process group,
# and pimm initialises that group NCCL-only (comm.py:183-184). NCCL has no
# gather: the first save died with "NCCL Error 1: unhandled cuda error" and every
# rank then hung until the 600 s watchdog fired. legacy writes one rank-0 file,
# which is complete under DDP and which train.sh's resume lookup already accepts.
singularity exec --nv -B /sdf,/lscratch,"${PIMM}":/opt/pimm/src "${IMAGE}" \
  bash --noprofile --norc -c "
    export EXP_ROOT=${EXP_ROOT}
    export NCCL_SOCKET_IFNAME='^docker0,lo'
    export OMP_NUM_THREADS=4
    cd /opt/pimm/src
    sh /opt/pimm/src/scripts/train.sh \
      -m 1 -g 4 -c helix/coeff_fm_train -n ${RUN_NAME} \
      -p /opt/pimm/.venv/bin/python -a ${RUN_NAME} ${WARM_ARGS} \
      -- --options checkpoint_format=legacy
  "
rc=$?
echo "[wrapper] train.sh exited ${rc}"
exit ${rc}
