# helix

A foundation model for liquid-argon TPC wire data and the signal processing that feeds it.
Raw wire ADC goes through coherent-noise removal and a thresholded wavelet transform (coif3,
level 4) into a sparse **coefficient corpus**. A **masked autoencoder** is pretrained on the
noisy coefficients alone, because real data has no noise-free counterpart. Its encoder is then
**fine-tuned as a denoiser**: from noisy tokens to pre-response charge per 2-wire x 16-tick
cell, which is denoising and deconvolution in one, scored against simulation truth. The same
chain reads JAXTPC simulation and SBND data. An optical codec writes PMT / X-ARAPUCA shards
beside the TPC ones; the model does not read them yet.

## Install

Everything runs in one container per site. The site profile `helix/sites/<site>.yaml` names
the image, the roots and the scheduler; `scripts/helix_run.sh CMD` runs CMD in that container
with helix and pimm (`HELIX_PIMM_ROOT`, pinned per site) on `PYTHONPATH`. Both are
bind-mounted, so editing either needs no rebuild. The host that submits jobs needs only a small
venv (`scripts/make_host_env.sh`). Outside the container, `pip install -e .` gives the paths,
the verifier and the tokenizer; the DSP and the model need `.[torch]`, the dataset, training and
evaluation `.[pimm]` plus a pimm checkout, and the truth step `.[probe]`.

## The short path

    scripts/make_host_env.sh && source scripts/helix_env.sh    # host venv; exports HELIX_*
    $HELIX_PYTHON -m helix.paths                               # site, roots, container, scheduler

    scripts/corpus.sh all sim $HELIX_CORPUS_ROOT/coeff_tpc_new $HELIX_SIM_ROOT/j00*   # plan, calibrate, build
    scripts/helix_run.sh python -m helix.data.verify $HELIX_CORPUS_ROOT/coeff_tpc_new
    export HELIX_CORPUS=$HELIX_CORPUS_ROOT/coeff_tpc_new/j000000                   # train on it
    scripts/helix_run.sh python scripts/build_truth.py --corpus $HELIX_CORPUS      # cells truth, per run (S3DF)

    scripts/submit.sh fm --resources.nnodes 4                   # pretrain: the stable phase
    scripts/submit.sh fm_cooldown --resources.nnodes 4          # then the cooldown
    scripts/submit.sh ft_eval --run.name ft_cool --resources.nnodes 1 -- model.fm=$HELIX_EXP/fm_cooldown
    scripts/helix_run.sh python -m helix.eval $HELIX_EXP/ft_cool --out $HELIX_EXP/ft_cool/eval.json
    scripts/helix_run.sh python -m helix.eval --uprobe $HELIX_EXP/fm_cooldown --out uprobe.json

    $HELIX_PYTHON scripts/sweep.py configs/sweeps/ft_eval.yaml --dry-run   # a whole study from one file

Training and evaluation need an A100 (sm80+). `docs/RUNBOOK.md` gives each step with its Slurm
form and the NERSC / S3DF differences.

## Documents

| | |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | data flow, packages and the boundary, the TPC chain, shard formats, model, denoiser, training, evaluation, sites |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | procedures as commands: host env, corpus, verify, truth, train, evaluate |
| [docs/STATUS.md](docs/STATUS.md) | current corpus, runs and numbers; standing decisions; open work |
| [CLAUDE.md](CLAUDE.md) | guidance and traps for agents working in this repository |

History (retired designs, study records, the frozen probes) is in git and the tags
`archive-2026-11` and `eval-probes-final`.
