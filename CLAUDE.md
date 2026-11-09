# CLAUDE.md

Guidance for Claude Code working in this repository.

HELIX: LArTPC wire data -> coherent-noise removal and wavelet sparsification -> a coefficient
corpus -> a masked-autoencoder FM trained on noisy coefficients -> a fine-tuned denoiser,
evaluated against simulation truth. Read `docs/ARCHITECTURE.md` (the code), `docs/RUNBOOK.md`
(commands) and `docs/STATUS.md` (numbers, decisions, open work) before changing anything
substantial. History is in git and the tags `archive-2026-11` and `eval-probes-final`; docs
describe the code as it is, not how it got there.

## The boundary

`helix.core` and `helix.tpc` must NEVER import `pimm_data` (pimm-data requires torch>=2.5 and
hdf5plugin; the corpus builder must not depend on it). torch is allowed there and is imported
at module scope by `helix.core.robust` and the TPC chain. Only `helix/data/dataset.py` and
`helix/integrations/*` import pimm-data / pimm; `helix.build.SimSource` and
`helix.model.tokenize` defer one import each. `tests/test_boundary.py` also requires a `LIGHT`
list (the packages, `helix.paths`, `helix.data.{bins,verify,cells,denoise}`,
`helix.model.tokenize`, `helix.model.checkpoint`, `helix.eval`, `helix.eval.events`) to import
with neither torch nor pimm-data, because `pimm submit` loads training configs on a host without
them, and forbids the string `'cells'` in pretraining code. Read it before moving code.

## Commands

Everything runs in the site's container; nothing names it. Runtime, image, interpreter, binds,
roots and scheduler facts are in `helix/sites/<site>.yaml`.

```bash
source scripts/helix_env.sh && $HELIX_PYTHON -m helix.paths   # FIRST: site, roots, container, scheduler
scripts/helix_run.sh python -m pytest tests -q
scripts/submit.sh fm --resources.nnodes 4                     # training, through pimm submit, on the host
scripts/helix_run.sh python -m helix.eval RUN --out e.json    # evaluation (an A100)
```

`helix_run.sh` sets `PYTHONNOUSERSITE=1` (a bound home would leak `~/.local` packages into the
image), `PYTHONPATH=pimm:helix`, and the site's `env:`. Submitting happens on the host (sbatch is
not in the container), in the venv `scripts/make_host_env.sh` builds. pimm is pinned per site
(`scheduler.train.pimm_rev`); `submit.sh` refuses another or a dirty pimm checkout.

**Two runtimes at NERSC.** podman-hpc is interactive and single-node (no NCCL plugin: 1.23 GiB/s
across nodes against shifter's 4.9). shifter is the batch runtime pimm's launcher drives.
`helix_run.sh` picks podman-hpc on a TTY; `HELIX_INTERACTIVE=0` forces shifter.

## Patterns worth knowing

- **The corpus is compression; the tokenizer is interpretation.** Shards store raw
  coefficients plus a `norm_sigma` table applied at tokenize. Changing the wavelet, gate,
  threshold or `tau` is a full rebuild and a new `basis_digest`, incomparable with every
  checkpoint; changing patch size, `n_bands` or masking is a config change and a retrain.
  Every shard carries its format stamp; there are no legacy read paths.
- **Configs state intent.** No step counts (`"auto"` cadence from `FMTrainer`), no world size;
  a variant is a `key=value` override. Warm starts are `model.init=<run>` / `model.fm=<run>`,
  never pimm's `weight=`.
- **Reading a run has one owner:** `helix.model.checkpoint.load_run` on pimm's run directory
  (`resolved_config.json`, `model/last`, `provenance.json`). Nothing is exported.
- **`plane_id` serves four roles** (FiLM index, RoPE axis, masking group, plane identity). A
  second modality would separate them.

## Traps that have cost real time

- **The global batch IS the rank count.** One event per rank (no event separation);
  `FMTrainer` refuses anything else. Configs hold one rank's batch; `submit.sh` appends
  `batch_size=<nodes x GPUs>`. Never write the global batch into a config. `ft_eval` also
  requires exactly 4 ranks.
- **Provenance refuses a dirty tree.** A corpus built from an uncommitted checkout records
  `git_dirty: True` and `helix.data.verify` refuses it; this once cost a full rebuild. Build
  and train from a frozen worktree.
- **`#SBATCH` directives cannot read shell variables**, so scheduler facts live in the site
  profile, split by job kind (`train`, `eval`, `corpus`; at S3DF corpus and training use
  different accounts), and reach sbatch / pimm as CLI flags via `scripts/helix_env.sh`.
  `sacctmgr -n show assoc user=$USER format=Account,Partition,QOS` lists yours.
- **A root a site lacks is `null`**, and asking for it raises `SiteError` with a named cause
  (NERSC has no `HELIX_SIM_ROOT`). A test that only LOADS configs must catch `SiteError` and skip.
- **`save_path` must resolve through `helix.paths`**, never relative to the checkout: a relative
  one once wrote a run into the repo.
- **GPFS does not support the locks HDF5 and uv take.** At NERSC every shard read fails
  `OSError: [Errno 524] unable to lock file` without `HDF5_USE_FILE_LOCKING=FALSE` (the
  profile's `env:` sets it); point `UV_CACHE_DIR` at node-local `/tmp`.
- **Training and evaluation need sm80+** (flash varlen attention): A100, never turing.
  Corpora are built on one GPU type (d1m: turing).
- **Never bypass the `basis_digest` refusal.** A run trained at another site has its corpus
  and FM run relocated by `helix.eval` (or `--corpus` / `--fm`); a different digest means
  another corpus.
