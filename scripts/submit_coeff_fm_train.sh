#!/bin/bash
# One LINK of the coefficient-FM stable phase, on preemptable GPUs.
#
# A link is short by design and the run is a CHAIN of them. Preemptable QoS has
# priority 1 against normal's 10000, so a long job is not a job that runs for
# long — it is a job that waits. Short links start sooner, and losing one costs
# at most SAVE_EVERY steps.
#
# Recovery is the CHAIN, not --requeue. This cluster runs PreemptMode=CANCEL
# (QoS preemptable = "within,cancel"), so a preempted job is cancelled outright
# and never returns to the queue whatever --requeue says. Measured 2026-09-16:
# job 38395161 carried --requeue, was preempted on sdfampere040 at 1:47, and
# stayed dead. The retired launch/ launcher had this right and this file did
# not; the header used to claim "--requeue puts the SAME job back on preemption".
#
#   --dependency=afterany   the NEXT link starts when this one ends for ANY
#                           reason -- preempted, completed, or out of wall clock.
#                           This is what actually continues a run.
#   --requeue               kept only for NODE FAILURE, which slurm does requeue.
#
# Both land here and are handled by the same rule below: resume if a checkpoint
# exists, start fresh if not. Nothing has to know WHY it restarted.
#
# Usage — submit a chain of N links:
#   scripts/chain_coeff_fm_train.sh <N> [config]
# or a single link by hand:
#   sbatch --export=ALL,CFG=<config> scripts/submit_coeff_fm_train.sh
#
#SBATCH --job-name=coeff8
#SBATCH --partition=ampere
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --requeue
# NO --signal=B:USR1: SLURM delivers it, pimm installs no handler, and the
# default disposition for SIGUSR1 is to TERMINATE. Link 1 died at 03:54:40 of a
# 4 h limit with State=FAILED ExitCode=0:10 — signal 10 is SIGUSR1. It bought
# nothing (there is no graceful-checkpoint path to trigger) and cost five
# minutes plus a FAILED that reads like a crash.
set -euo pipefail

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
CFG=${CFG:-$H/configs/pimm/coeff_fm_train_8run.py}
IMG=${IMG:-${HELIX_IMAGE:-/sdf/data/neutrino/omara/images/helix-train.sif}}
PIMM=${PIMM_ROOT:-/sdf/group/neutrino/omara/pimm-fm}

# K=128 at full event size needs Ampere: the logits are (n_cells, n_slot, K),
# ~2.4 GB for one event, and the backward doubles it. An 11 GB Turing card is
# where the smoke config lives, not this.
[ -f "$CFG" ] || { echo "FATAL: no config at $CFG"; exit 1; }

# ASK THE CONFIG, do not re-derive from its text. Grepping out save_path and
# recomputing STEPS in shell would be a second source of truth for numbers the
# config already computes — and a `_base_`-derived config keeps most of them in
# its parent, where the grep would not see them at all.
# NO pimm-data checkout here. The image installs it at the PINNED revision and
# container/helix-train.def's %post fails the build if it is stale, so shadowing
# it with a checkout only created a second authority: the pin governed the image
# while whatever was checked out governed the run. Set PYTHONPATH yourself if you
# are deliberately testing an unreleased pimm-data, and know that the run's
# provenance.json is then the only record of it.
export PYTHONPATH="$PIMM:$H"
read -r SAVE TOTAL < <(apptainer exec -B /sdf,/lscratch "$IMG" \
  env PYTHONPATH="$PYTHONPATH" /opt/pimm/.venv/bin/python - "$CFG" <<'PY'
import sys
from pimm.utils.config import Config
c = Config.fromfile(sys.argv[1])
print(c.save_path, int(c.STEPS))
PY
) || { echo "FATAL: could not resolve $CFG"; exit 1; }
[ -n "$SAVE" ] && [ -n "$TOTAL" ] || { echo "FATAL: config gave no save_path/STEPS"; exit 1; }
echo "config: $CFG -> $SAVE, $TOTAL steps"

# SNAPSHOT HELIX INTO THE RUN DIRECTORY.
#
# provenance.json records helix's commit and dirty flag, which is the right
# thing and was not enough. The 8-run's eleven links all recorded commit
# 87248d88 on branch `extraction`, clean -- and that hash resolves in NO
# repository today: the 2026-09-14 consolidation purged 517 research/ files and
# rewrote every commit, so the recorded identity ceased to exist. The cooldown
# is worse: three different commits across six links, one of them dirty.
#
# A hash is a pointer into a history someone may rewrite. A copy is not. pimm's
# own train.sh has snapshotted ITS code into <save_path>/code since forever --
# helix, the repo that actually defines the model and the tokenizer, had no
# equivalent. This is it.
#
# Written once per run, not per link: links share a save_path, and the first one
# is the code the run started from. ~13 MB, against checkpoints of 236 MB each.
if [ ! -d "$SAVE/helix-code" ]; then
  mkdir -p "$SAVE"
  if command -v git >/dev/null && git -C "$H" rev-parse --git-dir >/dev/null 2>&1; then
    # `git archive` takes the COMMITTED tree; a dirty tree would silently ship
    # something the commit does not describe, so fall back to a copy and say so.
    if [ -z "$(git -C "$H" status --porcelain)" ]; then
      mkdir -p "$SAVE/helix-code"
      git -C "$H" archive HEAD | tar -x -C "$SAVE/helix-code" \
        && echo "snapshot: $SAVE/helix-code <- git archive $(git -C "$H" rev-parse --short HEAD)"
    else
      echo "snapshot: WORKING TREE IS DIRTY — copying it verbatim, because the" >&2
      echo "          commit does not describe what will run" >&2
      mkdir -p "$SAVE/helix-code"
      tar -C "$H" --exclude=.git --exclude='*.pyc' --exclude=__pycache__ -cf - . \
        | tar -x -C "$SAVE/helix-code"
    fi
  else
    echo "snapshot: $H is not a git checkout; copying verbatim" >&2
    mkdir -p "$SAVE/helix-code"
    tar -C "$H" --exclude='*.pyc' --exclude=__pycache__ -cf - . | tar -x -C "$SAVE/helix-code"
  fi
fi

# Resume iff there is something to resume from. This is what makes a preempted
# link, a wall-clock link and a fresh start all the same case — the alternative
# is a flag the submitter has to get right on every link but the first, which is
# a rule that holds until the one time it does not.
# ASK PIMM which checkpoint to resume from. `pimm.utils.path latest-checkpoint`
# checks `last`, `last.prev` AND `model_last.pth`, validates each properly
# (`.complete` sentinel plus weights.pth plus a complete trainer.dcp), and picks
# the newest by mtime.
#
# This replaces a hand-rolled `[ -d last ] && [ -f last/.complete ]`, which was a
# strictly weaker subset in three ways, one of them dangerous:
#   * it knew only `last/`, so it silently missed the flat `model_last.pth` the
#     older run writes (that was one of two bugs that restarted a chain at step 1);
#   * it checked only the sentinel, not that weights.pth and trainer.dcp were
#     actually there;
#   * with a TORN `last/` — sentinel absent mid-save — it fell through to
#     `resume=False` and would have retrained from ZERO, where pimm falls back to
#     `last.prev`. The comment above it claimed the opposite.
#
# `resume=True` still needs `weight=` beside it: pimm resumes from `cfg.weight`
# (utils/checkpoints.py:1088, and logs "No weight found at:" at :1109, both inside
# `load_weight_and_resume`) without it. That is why pimm's own train.sh passes
# `resume=$RESUME weight=$WEIGHT` together.
#
# The exit status is checked, and stderr is NOT discarded. `2>/dev/null || true`
# collapsed two different outcomes into the same empty string: "no checkpoint
# exists yet" (the resolver exits 0 and prints nothing) and "the resolver
# CRASHED" (a rename upstream, an apptainer flake, an OOM). The second then read
# as the first, `resume=False`, and the chain restarted from step 1 -- which is
# the 3.9 GPU-hours this commit's own message is about. pimm's train.sh:230-235
# runs the identical call bare and hard-errors on an empty result; this does the
# same, while still allowing the legitimate empty case through.
_RESOLVER_ERR=$(mktemp)
set +e
WEIGHT=$(apptainer exec -B /sdf,/lscratch "$IMG" \
  env PYTHONPATH="$PYTHONPATH" /opt/pimm/.venv/bin/python -m pimm.utils.path \
  latest-checkpoint "$SAVE/model" 2>"$_RESOLVER_ERR")
_RESOLVER_RC=$?
set -e
if [ "$_RESOLVER_RC" -ne 0 ]; then
  echo "ERROR: checkpoint resolver exited $_RESOLVER_RC for $SAVE/model." >&2
  echo "       Refusing to launch: an unresumed run would silently restart at step 1." >&2
  sed 's/^/       /' "$_RESOLVER_ERR" >&2
  rm -f "$_RESOLVER_ERR"
  exit 2
fi
rm -f "$_RESOLVER_ERR"
if [ -n "$WEIGHT" ]; then
  # WHOSE run is this? Resume is decided purely by "a checkpoint exists at
  # $SAVE/model", so pointing this launcher at a save_path that already belongs
  # to another experiment silently CONTINUES that experiment's weights under this
  # config. save_path comes from a config whose run name is a LITERAL, so a new
  # HELIX_EXP with an unchanged name lands on an existing run -- and if HELIX_EXP
  # fails to reach the job at all, on the production one.
  #
  # The retired launch/ launcher had an ALLOW_RESUME guard for exactly this and
  # this launcher, the one the chain actually uses, had none.
  #
  # pimm dumps the RESOLVED config to <save_path>/config.py, so the check is
  # cheap and exact: if the run already there was produced by a different config,
  # refuse. Comparing the resolved dumps (not the source files) means a config
  # reached through different _base_ paths still compares equal.
  _PREV_CFG="$SAVE/config.py"
  if [ -f "$_PREV_CFG" ] && [ "${ALLOW_RESUME:-0}" != "1" ]; then
    _MINE=$(apptainer exec -B /sdf,/lscratch "$IMG" \
      env PYTHONPATH="$PYTHONPATH" /opt/pimm/.venv/bin/python - "$CFG" <<'PY' 2>/dev/null
import sys, hashlib, re
from pimm.utils.config import Config
def _fingerprint(path):
    c = Config.fromfile(path)
    t = c.pretty_text
    # save_path is excluded BY DESIGN: two configs that differ only in where they
    # write are the same experiment, and resuming one from the other is correct.
    t = re.sub(r"^save_path\s*=.*$", "", t, flags=re.M)
    return hashlib.sha256(t.encode()).hexdigest()[:16]
print(_fingerprint(sys.argv[1]))
PY
)
    _THEIRS=$(apptainer exec -B /sdf,/lscratch "$IMG" \
      env PYTHONPATH="$PYTHONPATH" /opt/pimm/.venv/bin/python - "$_PREV_CFG" <<'PY' 2>/dev/null
import sys, hashlib, re
from pimm.utils.config import Config
def _fingerprint(path):
    c = Config.fromfile(path)
    t = c.pretty_text
    # save_path is excluded BY DESIGN: two configs that differ only in where they
    # write are the same experiment, and resuming one from the other is correct.
    t = re.sub(r"^save_path\s*=.*$", "", t, flags=re.M)
    return hashlib.sha256(t.encode()).hexdigest()[:16]
print(_fingerprint(sys.argv[1]))
PY
)
    if [ -n "$_MINE" ] && [ -n "$_THEIRS" ] && [ "$_MINE" != "$_THEIRS" ]; then
      echo "ERROR: $SAVE already holds a run from a DIFFERENT config." >&2
      echo "         there: $_THEIRS" >&2
      echo "         here : $_MINE" >&2
      echo "       Resuming would continue THAT run's weights under THIS config." >&2
      echo "       Use a distinct run name, or set ALLOW_RESUME=1 if this is" >&2
      echo "       deliberate." >&2
      exit 2
    fi
  fi
  OPTS="resume=True weight=$WEIGHT"
  # A FLOOR on progress, not the step. `iter_N.pth` is written on the
  # CheckpointSaver's cadence, so N is always a multiple of SAVE_EVERY and the
  # true step is somewhere in [N, N+SAVE_EVERY). The exact step lives in
  # `last/trainer.dcp`, which cannot be read without building a trainer, and
  # `iter_N.pth` is not the resumable artifact anyway — `last/` is.
  #
  # That asymmetry is what makes this safe: the floor can only UNDER-report, so
  # the test below can miss a finished run (costing one link, which pimm itself
  # then exits in seconds with "Training already complete") but can never skip an
  # unfinished one. Report it as a floor rather than as the step — reading
  # "step 112500 of 112679" as real progress is what produced a spurious
  # "chain exhausted" on a run that had in fact completed at 112,677.
  FLOOR=$(ls "$SAVE"/model/iter_*.pth 2>/dev/null |
          sed 's/.*iter_\([0-9]*\)\.pth/\1/' | sort -n | tail -1)
  FLOOR=${FLOOR:-0}
  echo "resuming from $WEIGHT (at least step ${FLOOR} of ${TOTAL}; pimm decides)"
  if [ "$FLOOR" -ge "$TOTAL" ]; then
    echo "training already complete (>=${FLOOR}/${TOTAL}) — nothing to do"
    exit 0
  fi
else
  OPTS="resume=False"
  echo "no complete checkpoint under $SAVE/model — starting fresh, target ${TOTAL} steps"
fi

mkdir -p "$SAVE"
export APPTAINERENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
# Rank 0 of a 4-rank job is the only one that writes; the others must not race
# it for the same HDF5 page cache. Left at pimm's default otherwise.
export HDF5_USE_FILE_LOCKING=FALSE

# PREFLIGHT. A node whose GPUs are held reports healthy to SLURM and then fails
# the job in ~60 s — and with `afterany` chaining that eats one link per minute:
# links 2, 3 and 4 all died on sdfampere010 with
#   torch.AcceleratorError: CUDA-capable device(s) is/are busy or unavailable
# having consumed three links in three minutes while the node still showed
# State=MIXED.
#
# So check before committing the link, and REQUEUE rather than exit: a bad node
# then costs a trip through the queue instead of a place in the chain. The sleep
# is not politeness — without it a requeue that lands on the same node hot-loops.
if ! apptainer exec --nv -B /sdf,/lscratch "$IMG" /opt/pimm/.venv/bin/python -c '
import sys, torch
n = torch.cuda.device_count()
assert n >= 4, f"only {n} GPU(s) visible"
for i in range(4):
    torch.zeros(8, device=f"cuda:{i}")      # a context, not just a count
' 2>&1; then
  echo "PREFLIGHT FAILED on $(hostname) — requeueing rather than burning a chain link"
  sleep 60
  scontrol requeue "$SLURM_JOB_ID" || exit 1
  exit 0
fi

srun --kill-on-bad-exit=1 apptainer exec --nv -B /sdf,/lscratch "$IMG" \
  /opt/pimm/.venv/bin/python -m pimm.train \
    --config-file "$CFG" --num-gpus 4 --options $OPTS
