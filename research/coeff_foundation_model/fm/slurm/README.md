# helix FM — SLURM training jobs

Run training as proper SLURM jobs (survive logout, queue for a GPU, logged) —
modeled on `JAXTPC/slurm`. One GPU job per config, run inside the **PIMM
container** (`gpu-setup.sh IMAGES[pimm]`) on `ampere` (A100-40GB).

## Run

`train.sh` is **self-submitting** — run it on a login node (do *not* `sbatch` it):

```bash
cd /sdf/group/neutrino/omara/helix/research/coeff_foundation_model/fm

# submit one job per config
./slurm/train.sh configs/deconv_perc_d24.yaml configs/deconv_perc_d48.yaml

# wall-clock ceiling (job frees the node the instant training returns)
HOURS=12 ./slurm/train.sh configs/deconv_perc_d24.yaml

# different account / partition (aliases mirror ~/gpu-setup.sh)
ACCT=nu ./slurm/train.sh configs/deconv_perc_d24.yaml

# check your jobs
./slurm/train.sh status
```

Logs: `slurm_logs/<config-name>_<jobid>.out` (training stdout) and `.err`.
Checkpoints: written to the `ckpt:` path in the config (this dir), every `eval_every`.

## Config

A YAML in `configs/`. Its `script:` key picks the trainer; the rest become
that script's argparse **defaults** (explicit CLI flags still override — same
mechanism as JAXTPC's `--production-config`). To add a run, copy a config and
edit. Keys must match the trainer's argparse dest names (`events`, `steps`,
`batch`, `depth`, `M`, `d`, `heads`, `lr`, `no_ckpt`, `eval_every`, `ckpt`, ...).

Run a config directly (interactive GPU, inside the container) without SLURM:

```bash
python3 deconv_perc_batch.py --config configs/deconv_perc_d24.yaml --steps 2000
```

## Resources (per gpu-setup.sh)

- partition `ampere`: 28 CPUs, 230 GB, A100-40GB, up to 4 GPUs.
- account `cider-ml` → `mli:cider-ml` (default); `nu`, `cider-nu`, `default` also aliased.
- image `pimm` = `/sdf/data/neutrino/youngsam/containers/pimm.sif` (torch 2.5).
