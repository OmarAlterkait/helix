# Launch recipes

Three strategies for getting a training run onto a scheduler, plus the shared
shape they inherit. **None of them contains an absolute path, an account, or an
image**; every site fact is injected by `scripts/submit_helix.sh` from
`helix/sites/<site>.yaml` via `helix.paths`.

```bash
scripts/submit_helix.sh --recipe configs/launch/nersc-preempt.yaml --dry-run
```

| recipe | shape |
|---|---|
| `nersc-preempt.yaml` | 24h wall / 2h guaranteed floor, `chain.jobs: 8`. Submitit requeue. The default. |
| `nersc-interactive-chain.yaml` | 4 nodes x 4h, `chain.jobs: 32`, scron watchdog. Pass `--interactive --resources.qos interactive`. |
| `nersc-premium.yaml` | highest priority, no floor, no chain. Pass `--resources.qos premium`. |
| `_common.yaml` | what all three inherit: the config pointer and the runtime `setup:` |

## The three-way split, which is the whole design

pimm's launcher takes configuration from three places with three different
resolution rules, and each of helix's values has to go to the right one.

| what | where it can live | why |
|---|---|---|
| **recipe** | here, in helix | `--recipe PATH` accepts an absolute path (`pimm/launch/config.py` resolves it with `Path(recipe)`), so helix keeps its own launch shapes. |
| **site profile** | pimm's `launch/sites/` | `--site` takes a bare NAME resolved under pimm's own directory. helix cannot supply one, so it selects pimm's and overrides what it must. |
| **training config** | pimm's `configs/` | `train.sh -c` resolves only beneath that checkout. Hence the two-line pointer at `pimm-private/configs/coeff_fm/train_8run.py`, which computes its `_base_` from `$HELIX_ROOT`. |

## Launch YAML cannot read the environment

This is the constraint that shapes everything above, and it is worth stating
because the obvious fix does not work:

* `format_string` (`pimm/launch/config.py`) resolves `{placeholders}` against
  the launch config only, and hard-exits with `Unknown placeholder` otherwise.
  There is no `os.environ` anywhere in the resolution path.
* `env:` values are `shlex.quote`d at render time (`pimm/launch/local.py`), so
  `PYTHONPATH: $HELIX_ROOT` exports the eleven literal characters `$HELIX_ROOT`.
* `env` is not exposed on the CLI either (`schema.py` marks it `Suppress`).

Two seams do take a computed value, and both are used:

* **`setup:`** — its lines are joined and handed to `bash -c`, so they ARE
  shell, evaluated inside the job. `_common.yaml` uses this for `PYTHONPATH`.
* **the CLI** — `--paths.exp-root`, `--paths.repo-root`, `--container.image`,
  `--resources.*`. `scripts/submit_helix.sh` fills all of these.

## `--site` is not optional, even though every recipe names one

`pimm submit` defaults `site` to **s3df** (`pimm/cli/submit.py`), and
`load_config` takes `site or recipe.get("site")` — so the CLI value is always
truthy and a recipe's own `site:` key is never reached. Omitting it renders an
s3df job, with singularity and the wrong account, and says nothing.
`submit_helix.sh` always passes it, derived from the site profile's container
runtime. The `site:` key in `_common.yaml` is documentation.

## Two things `submit_helix.sh` sets that are easy to miss

**`paths.repo_root`.** Unset it defaults to `"."` — whatever directory you
submitted from. With shifter that binds *that* at `/opt/pimm/src` and runs
`/opt/pimm/src/scripts/train.sh`; submit from the helix checkout and the job
dies inside the allocation, because helix has no `scripts/train.sh`. It also
decides where `slurm_logs/`, the submitit folder and watchdog state land, and
helix's `.gitignore` covers `exp/` but not `slurm_logs`.

**`paths.exp_root`.** Defaults to `{repo_root}/exp`, inside the pimm checkout.
`train.sh` then passes `--options save_path=$EXP_DIR`, which **overrides** the
`save_path` helix's config computed from `helix.paths`.

## Not settled

* `signal_delay_s` defaults to 120, but NERSC's preempt `GraceTime` is 60s — so
  on preemption a checkpoint has one minute, not two. Measure a real checkpoint
  write at model size before trusting either.
* The code snapshot copies `scripts tools pimm` and **not** helix, so a chain
  freezes pimm and not helix. Point `HELIX_ROOT` at a git worktree pinned to one
  commit for the duration of a chain — that is what the variable is for.
* Shifter keeps its own image store: `shifterimg pull <image>` once before the
  first batch submit. podman-hpc's store is separate again.
