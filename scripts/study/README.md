# Preempt study harness

The scripts behind the scaling, design and A/B studies in `docs/SCIENCE.md` §9-10
and `docs/PERFORMANCE.md` §11. Each experiment is a short submitter in
`experiments/`; each job is one training arm sized to fill a preempt slot.

```bash
scripts/study/experiments/resolution.sh            # dry run: prints the sbatch lines
DRY=0 scripts/study/experiments/resolution.sh      # submit
```

| file | what it is |
|---|---|
| `common.sh` | sourced by submitters: site facts via `helix_env.sh`, `HELIX_STUDY_DIR`, `study_sbatch` |
| `slot.sbatch` | one training arm on N nodes (B = 4N), logs to `$HELIX_STUDY_DIR/<tag>-<jobid>.log` |
| `arm_steps.sh` | the in-container launcher: options, resume, epoch/max_len from the config |
| `cool1.sbatch`, `arm_cooldown.sh` | a 10% WSD cooldown branched from a checkpoint |
| `slot.py` | sizes STEPS to fill a 2 h slot from measured step times |
| `experiments/*.sh` | the runs as submitted: `kernel_ab`, `resolution`, `tier1` |

Runs, logs and checkpoints go under `HELIX_STUDY_DIR` (default
`$HELIX_SCRATCH/helix_work`; at NERSC that is Lustre scratch, purged if unused,
so copy anything to keep to `HELIX_EXP` or `HELIX_ARCHIVE`).

What these encode, each learned the hard way:

* **Fill the slot, and give slow arms more floor.** Preempt is billed at its
  `--time-min`. That is a floor, not the grant: Slurm may give any time between it
  and `--time`, and a TIMEOUT does not requeue. Arms longer than ~1.7 h pass
  `--time-min=03:00:00`.
* **Resume, never restart.** `RESUME=auto` resumes from the latest complete
  checkpoint, so a requeue or a resubmit continues the run instead of
  overwriting it.
* **Cool in-run.** Compare and fit on cooled loss
  (`scheduler.type=WSDCooldownLR scheduler.stable_frac=0.9`).
* **Seed explicitly.** `SEED` is passed through. It used to be a literal, so
  "replicates" were identical.
* **Scheduler facts come from the site profile.** Account, constraint and QOS
  reach `sbatch` as flags from `helix/sites/<site>.yaml`, because `#SBATCH` lines
  cannot read variables.
