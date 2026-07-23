# Live-import freeze list (Phase 0.3 of the helix consolidation)

In-flight training jobs (`fm_b4c_80k`, the babysat `fm_dscale600b4_20k`)
execute `torchrun … fm/mae_ddp.py`. Spawn DataLoader workers **re-import**
this closure on every worker (re)start, and a slurm requeue re-imports all
of it — so editing these files in the main checkout can change a running
experiment mid-flight.

**Policy (decided 2026-07-23): edit these files only in a git worktree**;
merge to main only at a boundary you choose (run finished, or an accepted
requeue point). Everything outside this list may be edited in place.

Import closure of `fm/mae_ddp.py` (mechanically derived, static):

    fm/mae_ddp.py        fm/data.py          fm/train.py
    fm/model.py          fm/model_serial.py
    star_tpc.py          measure_coeffs.py   vit_tpc.py
    baseline_tpc.py      star_model.py       vit_model.py
    doraemon_optical.py  onfly_optical.py

(The last five are pulled in transitively via module-level imports even
though the cached-training path never calls them — an argument for making
those imports lazy during consolidation, which would shrink this list.)

Also effectively frozen while runs are live: `fm/configs/b4c_80k.yaml`,
`fm/configs/dscale600b4_20k.yaml`, `fm/slurm/{train.sh,babysit_job.sh}`,
and the append-target `fm/fm_curve.jsonl` (never rewrite, only append).

Regenerate the closure after changing imports:
`python - <<'EOF' …` (see git log for the one-liner, or re-run the snippet
in research/goldens/capture-era notes).
