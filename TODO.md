# Deferred work

Things consciously left undone, with enough context to pick up cold. Ordered by
what blocks what, not by size.

Larger records get their own file: `MULTI_EVENT_BATCHING.md` (why the FM is one
event per forward), `NOISE_BANDS.md` (what the noise model does to each band, and
why m113 is out-of-distribution here).

**Running anything.** This tree needs two containers, and neither is the other:

```bash
# tests + DSP (torch, pytest, pywt)
singularity exec -B /sdf,/lscratch /sdf/group/neutrino/images/develop.sif \
    bash -lc 'PYTHONPATH=<helix>:<pimm-data>/src python -m pytest'

# anything importing pimm (pyarrow + addict; NOT containers/pimm.sif, which lacks pyarrow)
singularity exec --nv -B /sdf,/lscratch \
    /sdf/data/neutrino/youngsam/images/pimm-latest.sif ...
```

Short GPU work goes to the preemptable pool: `--account=mli:default
--qos=preemptable`, partition `turing` for wiring, `ampere` for anything with a
K=128 head at full event size.

---

## 1. Retire the research bundle — now unblocked

13 modules, ~3,700 lines, one connected graph anchored by `fm/mae_ddp.py`:

```
fm/     mae_ddp  model  model_serial  train  data
top     star_tpc  measure_coeffs  vit_tpc  star_model  vit_model
        baseline_tpc  doraemon_optical  onfly_optical
```

The gate was "FMTrainer parity". FMTrainer now exists and has trained on a GPU,
and `tests/test_training_parity.py` pins loss, gradients, optimizer state and a
full multi-step trajectory bit-exactly against the research implementation —
a stronger licence than a live A/B could have given, since `mae_ddp`'s recorded
numbers are on the white-noise distribution (`NOISE_BANDS.md`).

`research/goldens/capture.py` depends on this bundle (it calls
`star_tpc.prep_tpc_rows`), so the DSP golden retires with it.

## 2. Merge the branches

Held at instruction, not because anything is unfinished: helix `extraction`,
pimm-data `coeff-corpus`. `main` carries only the two corpus data fixes.

## 3. Probes and eval

`CoeffFMEvaluator` covers the training metric. What is missing is anything that
says whether the representation is GOOD: the 3D probe, charge closure, per-band
variance explained. The research versions were deleted deliberately (they were
written against the old npz cache); pimm has `EventProbeSuiteEvaluator` and
`hooks/eval/pretrain/probes/` to rebuild them into.

## 4. A real pretraining run

The 1500-step run was a demonstration: it reached bce 0.096 and categorical CE
2.29 from ln(128)=4.85, on 1500 of 19,999 events, in under 5 minutes. A real
pretrain is a different scale of job and wants a decision on steps, LR schedule
and corpus size.

## 5. Corpus scale

One run, 20k events, 45 GB. The plan was ~160k events across 8 runs. Everything
downstream works at either size; this is a compute decision.

## 6. Mirror the jax forward ops

`helix/tpc/{noise,dense_ops,geometry}.py` mirror pimm-data for numpy and torch,
pinned by `tests/test_forward_mirror.py`. The jax path (`noise_jax`,
`dense_ops_jax`) is not mirrored, so `build_coeff_corpus.py --backend jax` still
imports pimm-data. torch is the production backend, so this is a loose end.

## 7. Housekeeping

* `tests/test_coeff_dataset.py` in pimm-data pins the cross-repo codec golden to
  the hardcoded path `/sdf/group/neutrino/omara/helix-consolidate`, so as
  `extraction` diverges it compares against the wrong tree.
* ~~Version split: `pyproject.toml` 0.1.0 vs `helix/__init__.py` 0.2.0~~ — FIXED (0362328): pyproject reads the module via `[tool.setuptools.dynamic]`.
* The 10 back-compat flat shims (`helix/_backend.py`, `helix/io.py`, ...) are
  imported only by 6 scripts and 4 tests, all internal. helix has no external
  consumer.

## 8. Courtesy report to pimm's author

`engines/train.py::run_step` does
`if "offset" in input_dict: input_dict["coord"].shape[0]` — a batch with an
offset and no coord raises KeyError AFTER the forward. The sibling accounting at
line ~454 is guarded; this is not. One line. We do not hit it (`CoeffCollect`
emits no offset, asserted by a test), but it affects any non-point-cloud model.
