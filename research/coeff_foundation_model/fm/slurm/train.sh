#!/bin/bash
# =============================================================================
# train.sh — SLURM driver for helix FM training (one GPU job per config).
#
# Mirrors JAXTPC/slurm: SELF-SUBMITTING — run it on a login node (do NOT sbatch
# it). It sbatch's one job per YAML config onto `ampere` (1 A100-40GB / job),
# running the trainer inside the PIMM container. A job asks for HOURS but frees
# the node the instant training returns.
#
#   ./slurm/train.sh configs/deconv_perc_d24.yaml configs/deconv_perc_d48.yaml
#   HOURS=12 ./slurm/train.sh configs/deconv_perc_d24.yaml      # wall-clock ceiling
#   ACCT=nu PART=ampere ./slurm/train.sh configs/deconv_perc_d24.yaml
#   ./slurm/train.sh status                                     # your queued/running jobs
#
# The config YAML carries a `script:` key (which trainer to run) and the
# hyperparameters (loaded as argparse defaults by `<script> --config`).
# Account/image/partition mirror ~/gpu-setup.sh.
# =============================================================================
#SBATCH --job-name=helix_fm
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=230016M
#SBATCH --time=12:00:00
set -euo pipefail

# ============================= CONFIG (edit me) ==============================
WORKDIR=${WORKDIR:-/sdf/group/neutrino/omara/helix}
FMDIR="$WORKDIR/research/coeff_foundation_model/fm"
IMAGE=${IMAGE:-/sdf/data/neutrino/youngsam/containers/pimm.sif}   # gpu-setup.sh IMAGES[pimm]
export TMPDIR=${TMPDIR:-/lscratch/$USER/tmp}
HOURS=${HOURS:-}            # empty => use each config's `hours:` (env HOURS overrides it)
PART=${PART:-ampere}
# account aliases (mirror gpu-setup.sh): cider-ml -> mli:cider-ml (default)
ACCT=${ACCT:-cider-ml}
case "$ACCT" in
  nu)       ACCOUNT=mli:nu-ml-dev;    ACCT_QOS=  ;;
  cider-ml) ACCOUNT=mli:cider-ml;     ACCT_QOS=  ;;
  cider-nu) ACCOUNT=neutrino:cider-nu; ACCT_QOS= ;;
  default)  ACCOUNT=mli:default;      ACCT_QOS=preemptable ;;   # mli:default only offers preemptable
  *)        ACCOUNT=$ACCT;            ACCT_QOS=  ;;
esac
QOS=${QOS:-$ACCT_QOS}    # env QOS wins; else per-account default
# =============================================================================

# ---- STATUS -----------------------------------------------------------------
if [[ "${1:-}" == "status" ]]; then
    squeue -u "$USER" -o "%.10i %.16j %.9P %.8T %.11M %.4D %R"
    exit 0
fi

# ---- WORKER MODE: inside one job (HELIX_CFG set via --export) ----------------
if [[ -n "${SLURM_JOB_ID:-}" && -n "${HELIX_CFG:-}" ]]; then
    SCRIPT=$(awk '/^script:/{print $2}' "$FMDIR/$HELIX_CFG")   # whitespace-split: $2=value, ignores inline comment
    [ -z "$SCRIPT" ] && SCRIPT=deconv_perc_batch.py
    GPUS=$(awk '/^gpus:/{print $2}' "$FMDIR/$HELIX_CFG"); GPUS=${GPUS:-1}   # config-driven GPU count
    if [ "$GPUS" -gt 1 ]; then LAUNCH="torchrun --standalone --nproc_per_node=$GPUS"; else LAUNCH="python3"; fi
    echo "============================================================"
    echo " helix FM job $SLURM_JOB_ID : $LAUNCH $SCRIPT --config $HELIX_CFG   (gpus=$GPUS)"
    echo " host $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}  $(date)"
    echo "============================================================"
    mkdir -p "$TMPDIR"
    rc=0
    singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch "$IMAGE" \
        bash -lc "cd '$FMDIR' && PYTHONPATH='' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                  $LAUNCH '$SCRIPT' --config '$HELIX_CFG' --resume" || rc=$?   # --resume: idempotent, continues a preempted/crashed run
    echo "job $SLURM_JOB_ID finished at $(date) with exit code $rc"
    exit $rc
fi

# ---- shared submitter -------------------------------------------------------
SCRIPT_PATH="$(readlink -f "$0")"; mkdir -p "$FMDIR/slurm_logs"
QOS_ARG=(); [ -n "$QOS" ] && QOS_ARG=(--qos="$QOS")
submit_cfg() {                                              # $1 = config path
    local CFG="$1" rel name cfg_hours job_hours
    rel="${CFG#$FMDIR/}"; [ -f "$FMDIR/$rel" ] || { echo "ERROR: config not found: $CFG"; return 1; }
    name=$(basename "$CFG" .yaml)
    cfg_hours=$(awk '/^hours:/{print $2}' "$FMDIR/$rel"); job_hours=${HOURS:-${cfg_hours:-12}}
    local cfg_gpus job_gpus
    cfg_gpus=$(awk '/^gpus:/{print $2}' "$FMDIR/$rel"); job_gpus=${cfg_gpus:-1}   # config-driven; 1 node, N GPUs
    sbatch --job-name="fm_$name" --partition="$PART" --account="$ACCOUNT" \
        ${QOS_ARG[@]+"${QOS_ARG[@]}"} --requeue --nodes=1 \
        --gpus="$job_gpus" --cpus-per-task=28 --mem=230016M --time="${job_hours}:00:00" \
        --export="ALL,HELIX_CFG=$rel,WORKDIR=$WORKDIR,IMAGE=$IMAGE,ACCT=$ACCT" \
        --output="$FMDIR/slurm_logs/${name}_%j.out" --error="$FMDIR/slurm_logs/${name}_%j.err" \
        "$SCRIPT_PATH"
}

# ---- BABYSIT: keep configs alive through preemption (resubmit dead-incomplete) ----
if [[ "${1:-}" == "babysit" ]]; then
    shift; INTERVAL=${INTERVAL:-600}
    echo "babysitting $# config(s) every ${INTERVAL}s  (acct=$ACCT qos=${QOS:-<default>})"
    while true; do
        for CFG in "$@"; do
            name=$(basename "$CFG" .yaml); ck=$(awk '/^ckpt:/{print $2}' "$FMDIR/${CFG#$FMDIR/}")
            squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "fm_$name" && continue   # already active
            [ -f "$FMDIR/${ck}.done" ] && continue                                        # already complete
            echo "$(date +%H:%M:%S) resubmit fm_$name"; submit_cfg "$CFG" >/dev/null
        done
        sleep "$INTERVAL"
    done
fi

# ---- SUBMIT MODE: run on a login node ---------------------------------------
[ $# -lt 1 ] && { echo "usage: $0 <config.yaml> [more...]  |  $0 status  |  $0 babysit <config.yaml>..."; exit 1; }
echo "Account  : $ACCOUNT (acct=$ACCT)   qos=${QOS:-<default>}   Partition: $PART   Image: $IMAGE"
for CFG in "$@"; do
    name=$(basename "$CFG" .yaml); echo "  submit  $CFG  ->  job fm_$name"; submit_cfg "$CFG"
done
