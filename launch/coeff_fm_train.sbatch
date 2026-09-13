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

# THE THREE SITE-SPECIFIC VALUES ARE HERE: --account, --output and --partition.
# #SBATCH directives cannot read shell variables, so they cannot be resolved
# through helix.paths the way every other external path is. They are overridable
# on the command line instead, and sbatch's later flag wins:
#
#   sbatch --account=<facility>:<repo> --output=<your log dir>/coeff-fm-%j.out \
#          launch/coeff_fm_train.sbatch
#
# launch/chain_submit.sh forwards "$@" to every link of the chain, so the same
# two flags work there. Both defaults below point into one person's allocation
# and one person's directory: a different account fails at submission, and an
# unwritable --output fails the job with no log to say why.
#SBATCH --job-name=coeff-fm-train
#SBATCH --account=mli:cider-ml
#SBATCH --partition=ampere
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
# preemptable. NOTE the babysitter is NOT --requeue: this cluster runs
# PreemptMode=CANCEL (QoS preemptable: "within,cancel"), so a preempted job is
# CANCELLED outright and never returns to the queue, whatever --requeue says.
# Confirmed the hard way: a run reached epoch 11/25 (step 51,170, 3h20m) and then
# simply vanished. Continuation therefore needs something OUTSIDE the job --
# submit a chain of `--dependency=afterany` jobs, each carrying ALLOW_RESUME=1
# (see launch/chain_submit.sh). --requeue is kept because it still covers node
# failure, which is requeued rather than cancelled. This is only safe because
# resume now works: pimm's loader moved the saved RNG state to the GPU
# (checkpoints.py:940 map_location) and torch.set_rng_state rejects a CUDA
# tensor, so `resume=True` used to die before step 1 and an evicted run could
# only warm-start with its optimizer and step counter reset -- which on a
# 25-epoch schedule means restarting the schedule on every eviction.
# helix.integrations.pimm._patch_rng_restore_to_cpu moves it back; verified
# end-to-end ("Resuming train from structured state: global_step=32").
#
# So an eviction now costs at most SAVE_EVERY=238 steps, and --requeue puts the
# SAME job id back in the queue automatically.
#SBATCH --requeue
# Append, don't truncate: a requeued attempt reuses %j, and the default would
# erase the log of the attempt that got evicted.
#SBATCH --open-mode=append
#SBATCH --output=/sdf/data/neutrino/omara/exp/logs/coeff-fm-train-%j.out

set -uo pipefail

IMAGE=${HELIX_IMAGE:-/sdf/data/neutrino/omara/images/helix-train.sif}
PIMM=${HELIX_PIMM_ROOT:-/sdf/group/neutrino/omara/pimm-fm}
EXP_ROOT=${HELIX_EXP_ROOT:-/sdf/data/neutrino/omara/exp}

# Resolving the checkout: HELIX_ROOT, else the submission directory, else this
# script's own location — the same order, and for the same reason, as
# scripts/submit_coeff_corpus.sh. ${BASH_SOURCE[0]} is RIGHT under `srun bash`
# and WRONG under `sbatch`, which copies the script to
# /var/spool/slurmd/scripts/ and runs the copy.
if [ -n "${HELIX_ROOT:-}" ]; then
  H=$HELIX_ROOT
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/helix/paths.py" ]; then
  H=$SLURM_SUBMIT_DIR
else
  H=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
[ -f "$H/helix/paths.py" ] || {
  echo "FATAL: no helix/paths.py under H=$H."
  echo "       Set HELIX_ROOT to the checkout, or sbatch from it."
  exit 1; }

# pimm's train.sh takes `-c helix/coeff_fm_train` and resolves it under its OWN
# configs/, with no absolute-config option — so the configs must be reachable
# from there, and helix stays the single source of truth by linking rather than
# copying. This link used to be made BY HAND and left untracked, pointing at one
# person's worktree by absolute path; it did not exist in any fresh checkout and
# nothing recorded that it should. Deriving it from the running checkout is what
# makes that reproducible.
CFG_LINK="${PIMM}/configs/helix"
CFG_WANT="${H}/configs/pimm"
[ -d "$CFG_WANT" ] || {
  echo "FATAL: no configs/pimm under H=$H — cannot link pimm's configs/helix."
  exit 1; }
if [ ! -e "$CFG_LINK" ] && [ ! -L "$CFG_LINK" ]; then
  ln -s "$CFG_WANT" "$CFG_LINK"
  echo "[wrapper] linked ${CFG_LINK} -> ${CFG_WANT}"
elif [ -L "$CFG_LINK" ] && [ "$(readlink "$CFG_LINK")" != "$CFG_WANT" ]; then
  # Repointed, not left alone: a stale link silently trains from ANOTHER
  # checkout's configs, which is the failure this whole block exists to stop.
  echo "[wrapper] configs/helix pointed at $(readlink "$CFG_LINK"), repointing to ${CFG_WANT}"
  ln -sfn "$CFG_WANT" "$CFG_LINK"
elif [ ! -L "$CFG_LINK" ]; then
  echo "FATAL: ${CFG_LINK} exists and is not a symlink. Remove it, or set" >&2
  echo "       HELIX_PIMM_ROOT to a pimm checkout that does not have one." >&2
  exit 1
fi
# Overridable, because EXP_DIR is derived from it: reusing a name silently
# points a NEW run at a FINISHED run's directory.
RUN_NAME=${RUN_NAME:-coeff-fm-train}
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
RESUME_ARGS=""
RESTARTS=${SLURM_RESTART_COUNT:-0}
# Explicit opt-in for continuing an EXISTING run directory. Needed because a
# preempt-cancel means the continuation is a brand-new job with
# SLURM_RESTART_COUNT=0, which is indistinguishable from a run-name collision
# without being told which one this is.
ALLOW_RESUME=${ALLOW_RESUME:-0}
if [ -e "${EXP_DIR}/model/model_last.pth" ] || [ -d "${EXP_DIR}/model/last" ]; then
  if [ "$RESTARTS" -gt 0 ] || [ "$ALLOW_RESUME" = "1" ]; then
    # TRUE resume: train.sh derives the checkpoint and pimm restores optimizer,
    # scheduler, global_step, dataloader and RNG. Not a warm start.
    RESUME_ARGS="-r true"
    echo "[wrapper] resuming ${EXP_DIR} (full trainer state; restarts=${RESTARTS}," \
         "allow_resume=${ALLOW_RESUME})"
  else
    echo "ERROR: ${EXP_DIR}/model already holds a checkpoint, this is a FIRST" >&2
    echo "       attempt, and ALLOW_RESUME is not set. Continuing would silently" >&2
    echo "       train from THAT model. Set ALLOW_RESUME=1 to continue it, or" >&2
    echo "       pick a different RUN_NAME." >&2
    exit 2
  fi
else
  echo "[wrapper] no checkpoint under ${EXP_DIR}/model - starting from scratch"
fi

echo "[wrapper] job ${SLURM_JOB_ID:-?} restart count ${SLURM_RESTART_COUNT:-0}"

# `-c helix/coeff_fm_train` resolves to <pimm>/configs/helix/coeff_fm_train.py,
# the symlink established above into THIS checkout's configs/pimm. It points
# into /sdf, which is bound below.
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
      -p /opt/pimm/.venv/bin/python -a ${RUN_NAME} ${RESUME_ARGS} \
      -- --options checkpoint_format=legacy
  "
rc=$?
echo "[wrapper] train.sh exited ${rc}"

# Slurm's --requeue covers PREEMPTION and node failure, not a non-zero exit. But
# on a preemptable queue the most common startup failure is transient
# infrastructure -- e.g. "CUDA error: CUDA-capable device(s) is/are busy or
# unavailable" at torch.cuda.set_device, which happens when a node is still
# releasing a just-preempted job. Left alone, that silently ends a 7 h run within
# a minute of submission.
#
# So requeue ourselves on failure, but only while there is no checkpoint yet
# (i.e. the failure is at startup, not mid-training where --requeue already
# applies) and only a bounded number of times, so a DETERMINISTIC failure -- a
# bad config, a missing corpus -- fails loudly instead of looping forever.
MAX_STARTUP_REQUEUE=${MAX_STARTUP_REQUEUE:-3}
if [ "${rc}" -ne 0 ] \
   && [ ! -e "${EXP_DIR}/model/model_last.pth" ] \
   && [ "${RESTARTS}" -lt "${MAX_STARTUP_REQUEUE}" ]; then
  # NOTE a plain requeue can land on the SAME node, which is useless when the
  # node itself is the problem. Seen: sdfampere010 reporting state=idle,
  # reason=none while every job placed on it died at torch.cuda.set_device with
  # "CUDA-capable device(s) is/are busy or unavailable" -- broken but not
  # drained, so the scheduler kept handing it out. If the log shows the same node
  # twice, resubmit with --exclude=<node> rather than waiting out the retries.
  echo "[wrapper] startup failure on ${SLURMD_NODENAME:-?} with no checkpoint" \
       "(attempt $((RESTARTS+1)) of ${MAX_STARTUP_REQUEUE}); requeueing" \
       "${SLURM_JOB_ID}" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  sleep 30          # let the requeue register before the step exits
  exit 0
fi
if [ "${rc}" -ne 0 ]; then
  echo "[wrapper] NOT requeueing: either a checkpoint exists (so --requeue owns" \
       "this) or ${MAX_STARTUP_REQUEUE} startup attempts are exhausted" >&2
fi
exit ${rc}
