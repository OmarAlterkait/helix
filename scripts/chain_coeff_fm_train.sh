#!/bin/bash
# Submit a CHAIN of dependent training links.
#
# WHY A CHAIN, and not --requeue. This cluster runs PreemptMode=CANCEL (QoS
# preemptable = "within,cancel"): a preempted job is cancelled outright and
# never returns to the queue, whatever --requeue says. Confirmed twice -- once
# by a run that reached epoch 11/25 (step 51,170, 3h20m) and simply vanished,
# and again on 2026-09-16 when job 38395161 carried --requeue, was preempted at
# 1:47, and stayed dead. Continuation therefore needs something OUTSIDE the job:
# each link starts when the previous one ENDS for any reason
# (--dependency=afterany) and resumes from the last checkpoint, so a preemption
# costs at most SAVE_EVERY steps instead of the whole run.
#
# --requeue is still worth setting on the link itself, because it covers NODE
# FAILURE, which slurm does requeue.
#
# This is only safe because resume works: pimm's loader moved the saved RNG
# state to the GPU and torch.set_rng_state rejects a CUDA tensor, so `resume=True`
# used to die before step 1 and an evicted run could only warm-start with its
# optimizer and step counter reset -- which on a 25-epoch schedule means
# restarting the schedule on every eviction.
# helix.integrations.pimm._compat moves it back.
#
# (This paragraph came from launch/coeff_fm_train.sbatch, which knew it first
# and was retired into this launcher.)

N=${1:-4}
# Locating the checkout: NOT by name. Three helix checkouts have existed side by
# side, and an old hardcoded default of `helix-consolidate` silently ran code "32
# commits behind, missing the MAD median fix, the packaged noise spectrum, the
# m113 anchoring and the whole pimm integration" (see build_coeff_corpus.py).
#
# ${BASH_SOURCE[0]} is RIGHT under `srun bash scripts/...` and WRONG under
# `sbatch`, which copies the script to /var/spool/slurmd/scripts/ and runs the
# copy — so it cannot be the only source. SLURM_SUBMIT_DIR is where sbatch was
# invoked, trusted only if it actually looks like a checkout.
if [ -n "${HELIX_ROOT:-}" ]; then
  H=$HELIX_ROOT
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/build_coeff_corpus.py" ]; then
  H=$SLURM_SUBMIT_DIR
else
  H=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
[ -f "$H/helix/paths.py" ] || { echo "FATAL: $H is not a helix checkout (set HELIX_ROOT)"; exit 1; }
CFG=${2:-$H/configs/pimm/coeff_fm_train_8run.py}
ACCT=${ACCOUNT:-mli:default}
# EXCLUDE=node[,node] for nodes known to be bad. A node whose GPUs are held
# still reports healthy to SLURM, so it keeps being offered: sdfampere010 failed
# links 2, 3, 4 and 5 in about a minute each with "CUDA-capable device(s) is/are
# busy or unavailable". The launcher's preflight now requeues instead of burning
# a link, but excluding a known-bad node saves the round trip.
EXC=${EXCLUDE:-}
# QOS: empty means "whatever submit_coeff_fm_train.sh declares", which is
# preemptable. On S3DF that costs roughly HALF the scheduling priority --
# measured in one 409-deep ampere queue, preemptable jobs sat at 9,592 while the
# same user's normal-QoS jobs on the SAME account were at 19,591. There was no
# way to run a chain at normal QoS at all: ACCOUNT and EXCLUDE were overridable
# and this was not, so switching accounts to get headroom silently kept the low
# priority that made the switch pointless.
#
# Not defaulted to normal: preemptable is the right choice for a long chain when
# the allocation is shared, and `normal` is not granted on every account
# (`sacctmgr -n show assoc user=$USER format=Account,Partition,QOS` lists yours;
# mli:default and neutrino:default carry preemptable only).
QOS=${QOS:-}
# Under the run's own experiment root, not a personal scratch directory. This
# defaulted to /sdf/data/neutrino/omara/exp/_diag/trainlogs -- one person's
# diagnostics folder on one cluster, which a recipient cannot write to and which
# put the logs for every 8-run link somewhere the run directory does not mention.
LOGS=${LOGDIR:-${HELIX_EXP:-$PWD/exp}/trainlogs}
mkdir -p "$LOGS"

case "$CFG" in /*) ;; *) CFG="$H/$CFG" ;; esac
[ -f "$CFG" ] || { echo "FATAL: no config at $CFG"; exit 1; }

# Name the jobs after the CONFIG, not "coeff8". Every link of every chain used
# to be `coeff8_<i>`, so two chains running different experiments were
# indistinguishable in squeue -- and their logs collided in the same
# coeff8_<i>_%j.out namespace. Deciding which of two chains to cancel then meant
# reasoning from submission order, which is exactly the kind of thing that is
# wrong at 2am. The config name is the one thing that actually differs.
TAG=$(basename "${CFG%.py}")
TAG=${TAG#coeff_fm_}            # coeff_fm_train_8run_plane -> train_8run_plane

PREV=""
for i in $(seq 1 "$N"); do
  DEP=""
  [ -n "$PREV" ] && DEP="--dependency=afterany:$PREV"
  JID=$(sbatch --parsable $DEP ${EXC:+--exclude=$EXC} ${QOS:+--qos=$QOS} \
      --account="$ACCT" \
      --job-name="${TAG}_$i" \
      --output="$LOGS/${TAG}_${i}_%j.out" \
      --error="$LOGS/${TAG}_${i}_%j.out" \
      --export=ALL,HELIX_ROOT=$H,CFG=$CFG \
      "$H/scripts/submit_coeff_fm_train.sh")
  printf "link %2d/%s -> job %s%s\n" "$i" "$N" "$JID" "${PREV:+  (after $PREV)}"
  PREV=$JID
done
echo
echo "logs: $LOGS"
echo "watch: squeue -u \$USER -n ${TAG}_1 ; tail -f $LOGS/${TAG}_1_*.out"
