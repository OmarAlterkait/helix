# helix

A foundation model for liquid-argon TPC wire data, and the signal processing that
produces what it trains on: raw wire ADC -> coherent-noise removal and wavelet
sparsification -> a coefficient corpus -> a masked autoencoder trained on the noisy
coefficients alone (real data has no noise-free counterpart) -> a supervised denoiser
on its encoder, and an evaluation of both against simulation truth. The same chain reads
JAXTPC simulation and SBND data; an optical codec compresses PMT and X-ARAPUCA
waveforms into shards beside the TPC ones (not yet fed to the model).

## Layout

| path | what |
|---|---|
| `helix/build.py` | the build pipeline: event loop, TPC chain, optical codec, shard writer, calibration, task plan, one CLI (`main(Source)`); knows no source |
| `helix/core` | wavelet transform, robust statistics, hysteresis, the shard format (`coeff_io`), provenance, training cadence |
| `helix/tpc` | the TPC chain (`chain`), coherent gate, thresholding, noise model, readouts, JAXTPC and SBND readers |
| `helix/optical` | the optical codec, response calibration, optical shards and tokens |
| `helix/data` | corpus reader, dataset and verifier; load-time transforms; bin grid; the `cells` truth; the denoiser's labels; identity guard |
| `helix/model` | the FM: tokenizer, blocks, masking, loss, muP; the denoiser on it; reading a run (`checkpoint.load_run`) |
| `helix/eval` | evaluating a run: `python -m helix.eval RUN --metrics recon,floor,uprobe,labels,denoise` |
| `helix/integrations/pimm` | adapters into pimm's trainer |
| `helix/sites` | per-site roots, container and scheduler (`s3df.yaml`, `nersc.yaml`) |
| `scripts/build_{sim,sbnd,goop}.py` | the sources, one script each; `scripts/corpus.sh` runs a corpus on the scheduler |
| `scripts/build_truth.py` | the simulation truth (`cells`) beside a corpus run |
| `scripts/derive_coeff_bins.py` | the categorical head's bin grid |
| `scripts/train.sh` | training (FM or denoiser) as a chain of jobs |
| `configs/pimm/` | `fm.py` and its variants `fm_cooldown`, `fm_smoke`; `denoise.py`, the supervised denoiser on an FM run |
| `container/` | the image recipe (`install.sh`) and its apptainer / Docker front-ends |
| `tests/` | the suite |

## The short path

Paths, container and scheduler come from `helix/sites/<site>.yaml`; `source
scripts/helix_env.sh` exports them as `$HELIX_*`. The full procedures are in
[docs/RUNBOOK.md](docs/RUNBOOK.md).

    scripts/make_host_env.sh               # once per machine: a host Python with PyYAML
    source scripts/helix_env.sh            # exports HELIX_* (and HELIX_PYTHON)
    $HELIX_PYTHON -m helix.paths           # site, roots, container, scheduler

    # build a corpus: plan, calibrate norm_sigma, build (Slurm arrays)
    scripts/corpus.sh all sim $HELIX_CORPUS_ROOT/coeff_tpc_new $HELIX_SIM_ROOT/j00*

    # verify it against its plan
    scripts/helix_run.sh python -m helix.data.coeff_verify $HELIX_CORPUS_ROOT/coeff_tpc_new

    # truth beside each run (RUNBOOK section 7; on a CPU node)
    scripts/helix_run.sh python scripts/build_truth.py --corpus $HELIX_CORPUS

    # pretrain on HELIX_CORPUS's corpus: the stable phase as 8 chained jobs, then the cooldown
    LINKS=8 scripts/train.sh fm
    scripts/train.sh fm_cooldown

    # the denoiser on the pretrained encoder
    LINKS=4 scripts/train.sh denoise model.fm=$HELIX_EXP/fm_cooldown

    # evaluate (on an A100 node: RUNBOOK section 7)
    scripts/helix_run.sh python -m helix.eval $HELIX_EXP/fm_cooldown --metrics recon,floor,uprobe,labels --out eval.json
    scripts/helix_run.sh python -m helix.eval $HELIX_EXP/fm_cooldown --metrics denoise --denoiser $HELIX_EXP/denoise --out dn.json

The bin grid the configs read is `$HELIX_ARCHIVE/coeff_bins_d1m_j000000_noisy.pt`
(RUNBOOK section 5).

## Documents

| | |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | modules, data flow, the shard format |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | procedures as commands |
| [docs/DECISIONS.md](docs/DECISIONS.md) | standing decisions and the evidence behind them |
| [docs/OPTICAL.md](docs/OPTICAL.md) | the optical codec |
| [TODO.md](TODO.md) | what is open |
| `docs/archive/` | retired documents and study records (design notes, measurements, the denoising and encoder-pooling studies, handover), kept unchanged |

## Install

    pip install -e .                 # base: numpy, h5py, PyWavelets, scipy, PyYAML
    pip install -e ".[torch]"        # + torch (the TPC chain and the optical codec run in torch)
    pip install -e ".[pimm]"         # + pimm-data, for the corpus reader and training
    pip install -e ".[probe]"        # + hdf5plugin and torch: the truth step and helix.eval

Training also needs a pimm checkout (`HELIX_PIMM_ROOT`). The site's container has
every dependency; `scripts/helix_run.sh` runs a command in it with helix and pimm on
`PYTHONPATH`.
