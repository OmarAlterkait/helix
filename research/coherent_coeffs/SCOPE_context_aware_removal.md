# Scope — context-aware coherent removal (beating the kgate frontier)

Written 2026-08-15, after the recheck in `research/r2_qualification/REPORT.md`
("RECHECK on the FM corpus noise"). This scopes the work needed to remove the
leftover coherent blips WITHOUT paying the fidelity cost that every `kgate`
setting demands. It is a scoping document, not a decision.

## 1. The problem, precisely

`gate_band` sets `Mc = 0` wherever `|M| >= kgate*sigc`, so in those cells it
subtracts NOTHING and the whole coherent component survives. The leftovers are
therefore exactly one `group_size`-wide block, and they land where the true clean
image is empty.

One scalar `sigc` per band has to serve two opposite regimes:

  * empty region — a large `|M|` is coherent, and refusing to subtract is wrong;
  * dense deposit — a large `|M|` is partly signal, and subtracting is wrong.

Measured consequence (1,200 event-plane pairs, all 100 shards, F0 vs the TRUE
clean). No setting gives 3.0's fidelity with the strong blips gone:

    setting      F0        dF0      off>5    ratio   strong blips
    3.0      0.9056          -     23.44M    1.00x   present
    2.5->3.5 0.9046    -0.0009     11.76M    1.99x   PRESENT (population halved)
    2.5->4.0 0.9023    -0.0032      9.51M    2.46x   gone
    2.5->4.5 0.9001    -0.0055      9.53M    2.46x   gone (k2 saturated)
    3.0 soft 0.8885    -0.0170     11.13M    2.11x   -

## 2. What is already known (do not re-derive)

* **The cheap coefficient-space fix was tried and mostly failed.** `smart.py`'s
  docstring, §6p: an `occ_map`-driven "coeff-space spatial mask-dilation
  recovered <20% of de2's edge at a coeff cost", and a noise-anchored sigma was a
  no-op. So `occ_map` reweighting is NOT expected to close this.
* **`gate_soft` is dominated**, on fidelity (REPORT.md: U 0.839 / V 0.869 /
  Y 0.936) and now on the artefact axis too (-0.0170 F0 for 2.11x, against
  -0.0032 for 2.46x from 2.5->4.0). Clipping subtracts `t` from every cell
  including signal-dense ones, bleeding a constant off all real charge.
* **`de2_clamp` already exists and already solves it** (`DE2_CLAMP.md`,
  `induction.py::iterate`). Detect signal by 2D hysteresis + connected
  components, estimate the common mode per (block, tick) from the UN-FLAGGED
  wires only, interpolate where fewer than `minc` clean wires remain, then clamp
  to the smart anchor +-4 ADC. Measured, 12 events:

      Y  F0 0.9635  40k coeffs   nz_in 1.93 -> nz_out 1.60
      U  F0 0.8926  45k          2.83 -> 1.66   (ties helix F0 at 45k vs 55k)
      V  F0 0.8975  37k          2.15 -> 1.66   (dominates smart AND helix)

  "nz_out hits the intrinsic floor (~1.6) everywhere" — i.e. no leftover coherent
  off-signal, which is exactly the property the gate cannot deliver.
* **Its disposition was "opt-in only"** (REPORT.md, record §6n): de2_clamp does
  not beat R1 on U. That verdict was reached on fidelity/win-rate. It was never
  scored on leftover coherent, which is the axis that motivates this work — so
  the disposition should be revisited, not assumed.

## 3. Options

### A. Pick a frontier point. No code change.
Choose 2.5->3.5 (fidelity-first, blip population halved, strong blips remain) or
2.5->4.0 (blips gone, -0.0032 F0). Cost: a corpus rebuild (~25 min of array) plus
recalibration and bins. Zero new algorithm risk. **This is the fallback and it is
already fully validated.**

### B. `occ_map`-weighted gate. Coefficient-space, stays on GPU.
Make the threshold per-(block, position) instead of one scalar per band: where
signal occupancy is low, the cell is trustworthy coherent, so raise the threshold
(subtract); where occupancy is high, keep today's conservative behaviour.
`block_common_mode` already computes the occupancy for free (`return_occ`).
- Pros: small change, no new dependency, stays in torch/GPU, keeps the corpus
  build at 0.8 s/event, and reuses machinery that already exists.
- Cons: §6p says this class of fix recovered <20% of de2's edge. Expect a partial
  win at best.
- Effort: ~1 day to implement in the three backends + the 1,200-pair sweep.

### C. Port `de2_clamp` into production.
- Pros: the only approach measured to reach the intrinsic floor; documented
  knobs, sensitivities and failure modes; ties/beats current F0 at fewer
  coefficients, which also shrinks the corpus.
- Cons, and they are real:
  * **~12x more compute.** Measured on real plane shapes: the detect+estimate
    loop costs ~9.7 s/event (6 planes, n_iter=4) against a current whole-pipeline
    0.8 s/event. In the 100-task array (32 concurrent, 200 events/task) that is
    ~35 min/task, ~2 h wall — acceptable, but it is 12x the CPU-hours.
  * **CPU-only detector.** `scipy.ndimage.label` has no GPU path, so the DSP
    stops being end-to-end on-device. Note this arguably HELPS reproducibility
    (the current pipeline is pinned to turing precisely because GPU float
    reduction order is architecture-sensitive), but it is a structural change to
    how the corpus is built.
  * **Sample space, not coefficient space.** The corpus stores coefficients; de2
    estimates coherent in sample space and the anchor is the smart estimate. The
    pipeline would gain a stage rather than swap one.
  * **More knobs to own**: `klo` (most sensitive, [0.5,0.9], percolates below
    0.45), `dilate` (plane-dependent: U ~15-31, V ~11-15, Y indifferent),
    `clamp` (~4). Three plane-dependent parameters where the gate has one.
- Effort: ~3-5 days (port + backend parity + tests + sweep + a corpus rebuild).

### D. Do nothing. Ship 3.0.
Defensible: 3.0 is the qualified default (REPORT.md, n=129 stratified) and the
blips are ~10-20 ADC against ~150-500 ADC signal peaks. The argument against is
that they are STRUCTURED and block-aligned, so a masked-prediction model can
learn them as a shortcut, which a diffuse fidelity loss cannot become.

## 4. Recommended sequence

1. **Decide the corpus now from Option A**, so training is not blocked on
   algorithm work. 2.5->4.0 if the blips matter, 2.5->3.5 if fidelity does.
2. **Re-score de2_clamp on the artefact axis** before any porting — it was
   dispositioned on fidelity alone. This is CHEAP: the research code runs today,
   and the existing `_diag/kgate_variants.py` harness already measures
   off-signal >5 ADC / rms / F0-vs-true-clean on 1,200 pairs. Half a day.
   Kill criterion: if de2_clamp does not beat 2.5->4.0 on off-signal at equal or
   better F0, stop — Option A is the answer and C is not worth 12x compute.
3. Only if (2) passes, run Option B as the cheap-alternative check (1 day), since
   a GPU-resident partial win may beat a CPU-bound full win in practice.
4. Port whichever survives, then A/B by PROBE score, not by F0 — the point is
   representation quality, and the existing probe suite plus the identity-based
   holdout (bit-identical across corpora) makes that a clean comparison.

## 5. Validation plan (identical for every option)

* 1,200 (event, plane) pairs, 200 events across all 100 shards, paired.
* F0 against the TRUE clean — NEVER against the stored `coeff_clean`. The
  co-supported target reverses the kgate ranking (favours 4.0 on 90% of pairs
  while the true clean favours 3.0 on 93%), because it is missing ~2.7% of true
  charge and its reconstruction leaks nonzero pixels ~5x too widely.
* Off-signal >5 ADC, rms and max, always on the TRUE signal mask.
* Residual-centred crops, not signal-centred: an aggregate ratio is not an empty
  region, which is how 2.5->3.5 hid its surviving strong blips.
* Coefficient count, since it sets corpus size and token cost.

## 6. Open risks

* de2's numbers are n=12 events, on research's own noise realisations. The
  recheck harness uses ours (200 events, colored spectrum, the corpus seeds).
  They may not transfer.
* `klo` percolation (<=0.45 "blows up") is a silent failure mode — a bad
  detection event over-removes. The clamp exists to bound it; a corpus build has
  no human in the loop, so any port needs a per-event guard and a recorded
  diagnostic.
* Three plane-dependent knobs is a qualification burden: the gate's single
  `kgate` took n=129 stratified events to settle.
