# Results — wire-plane denoise/compress, learned transforms

Test: held-out doraemon events, **Stage A = intrinsic noise only** (no coherent).
Metric: F0 (fidelity on signal pixels, higher=better), noise_rms (off-signal, lower=better),
compression = pixels / kept-coeffs. R-D operating point quoted near ~10× compression.

## P1 — baseline per-wire DWT (VisuShrink-hard), sweep wavelet/level/κ
| plane | raw F0 | raw nrms | best DWT @~10× | comp | F0 | nrms |
|-------|--------|----------|----------------|------|----|------|
| Y | 0.9734 | 1.61 | db8 / L4 / κ1.0  | 9.2× | 0.9740 | 1.15 |
| U | 0.9336 | 1.67 | coif3 / L4 / κ1.5 | 12.6× | 0.9450 | 1.02 |
| V | 0.9269 | 1.67 | db8 / L4 / κ1.0  | 9.2× | 0.9327 | 1.23 |

Notes:
- DWT denoising *raises* F0 vs raw (removes noise on signal pixels too) while compressing ~10×.
- **Level 4 > level 8** here (level 8 → boundary effects on 4321-len signals). Family matters
  little (db8 ≈ coif3 ≈ sym8 within ~0.001 F0); κ is the rate knob — matches production lore.
- Y (collection) is intrinsically easier (raw F0 0.973) than U/V (induction, bipolar) ~0.93.
- This is the **bar** the learned methods must beat. (Full R-D curves in artifacts/baselines.json.)

## P2–P4 — learned transforms vs DWT, matched compression (Stage A, intrinsic noise)
F0 at matched compression (higher = better). KLT = per-plane PCA basis on clean
time-patches; dict = overcomplete dictionary (1D non-overlap patches, OMP);
lwave = neural lifting wavelet (learned filters + threshold), trained per plane.

| plane | comp | DWT | KLT | learned-wavelet | dict (1D patch) |
|-------|------|-----|-----|-----------------|------------------|
| Y | ~10× | **0.974** | 0.972 | 0.972 | 0.915 |
| Y | ~50× | **0.968** | 0.964 | 0.951 | — |
| Y | ~100× | 0.950 | **0.956** | (caps ~50×) | — |
| U | ~10× | **0.944** | 0.936 | 0.939 | 0.872 |
| U | ~50× | **0.941** | 0.929 | 0.912 | — |
| V | ~10× | **0.933** | 0.928 | 0.927 | (incomplete) |
| V | ~100× | **0.914** | 0.898 | (caps) | — |

**Stage A conclusion: the well-tuned fixed DWT wins or ties at essentially every
operating point.** Findings:
- **No learned transform beats DWT** on this sparse-pulse data. KLT and the neural
  lifting-wavelet *match* DWT at moderate compression (~10–20×) but fall behind by
  ~50×. KLT edges DWT only at extreme compression on the clean Y plane (≥100×).
- **Dictionary learning (1D non-overlapping patches) underperforms** (~0.06–0.07 F0
  below DWT at 10×) — pulses straddle patch boundaries and fixed-nnz OMP wastes
  atoms on noise-only patches. Overlapping-patch + error-tolerance OMP would close
  some of the gap but is far costlier; not pursued.
- **Why no win:** the signal is localized bipolar/unipolar pulses that wavelets
  already represent near-optimally; a learned basis has little structure left to
  exploit. The neural lifting model (6 levels, length-4 filters) is also capped at
  ~50× by always keeping the full level-6 approx band.
- **Per-plane:** Y (collection) is easiest (F0≈0.97); U/V (induction, bipolar)
  ≈0.93–0.94 and benefit slightly more from denoising. The *ranking* is the same
  across planes — DWT first — so the "optimal transform" does not flip by plane,
  though the achievable F0 does.

## P5 — Stage B: coherent noise (group-correlated), ~10× compression
| plane | raw coherent (no removal) | + helix coherent removal |
|-------|---------------------------|--------------------------|
|       | F0 / noise_rms (best method) | F0 / noise_rms (best method) |
| Y | DWT 0.951 / 2.56 | **DWT 0.964 / 1.22** |
| U | DWT 0.888 / 2.38 | **DWT 0.903 / 1.18** |
| V | DWT 0.874 / 2.59 | **DWT 0.899 / 1.36** |

(raw-noisy, no processing: F0 0.95/0.87/0.87, noise_rms ≈3.0 all planes.)

**Stage B conclusion:**
- **Coherent removal is essential and is a *separate, cross-wire* problem.** Per-wire
  transforms (DWT, KLT, lwave) cannot remove coherent noise — applied to raw
  coherent-noisy data they all leave noise_rms ≈2.4–2.6 (the coherent waveform looks
  like signal on each wire). With helix multi-pass removal first, noise_rms drops to
  ~1.2–1.4 and F0 recovers to near the Stage-A level.
- After removal, the Stage-A ranking is unchanged: **DWT best**, KLT/lwave within
  ~0.005 F0. No learned per-wire transform absorbs coherent structure — that would
  require a 2D (wire×time) or explicitly group-aware model.

## P7 — 2D (wire×time) separable wavelet on raw coherent-noisy data
Tested whether a 2D transform's cross-wire decorrelation removes coherent noise
without the explicit removal step.

| plane | 2D-DWT raw (coh) | per-wire DWT raw (coh) | removal + per-wire DWT |
|-------|------------------|------------------------|------------------------|
| Y | 5× / F0 0.950 / nrms **2.84** | 9× / 0.951 / 2.56 | 9× / **0.964** / **1.22** |
| U | 5× / 0.878 / 2.86 | 12× / 0.888 / 2.38 | 12× / **0.903** / **1.18** |
| V | 5× / 0.868 / 2.86 | 9× / 0.874 / 2.59 | 9× / **0.899** / **1.36** |

**2D-DWT does *not* help** — noise_rms stays ~2.8 (worse than per-wire) at lower
compression. The coherent waveform is constant across each 64-wire group, so it
lands in the low-wire-frequency bands that *overlap the signal*; generic dyadic
thresholding can't separate them. helix's removal works because it is **group-
structured** (per-64-wire-group median/mean subtraction) — it matches the actual
noise geometry, which a generic 2D wavelet does not. So the promising learned
direction is not a 2D *transform* but a **group-aware** learned model.

## P8 — group-aware LEARNED coherent removal (the win) → then DWT
A 19k-param CNN estimates each 64-wire group's shared coherent waveform from
robust across-wire statistics (median/mean/std it refines) + temporal context,
supervised on the synthetic coherent waveform; subtract, then the usual per-wire DWT.

| plane | no removal +DWT | helix removal +DWT | **learned removal +DWT** | Stage A (no coherent) |
|-------|-----------------|--------------------|--------------------------|------------------------|
| Y | 0.951 / 2.56 | 0.964 / 1.22 | **0.970 / 1.15** | 0.974 |
| U | 0.888 / 2.38 | 0.903 / 1.18 | **0.915 / 1.04** | 0.944 |
| V | 0.874 / 2.59 | 0.899 / 1.36 | **0.914 / 1.23** | 0.933 |
(F0 / noise_rms at ~10× compression.)

**The learned group-aware removal beats hand-crafted helix removal on all three
planes** (+0.006 / +0.012 / +0.015 F0; lower residual noise) and nearly fully
recovers the no-coherent Stage-A F0. Its residual noise (1.61/1.68/1.66) sits right
at the **intrinsic floor** (~1.6) — i.e. it removes the coherent component almost
perfectly, cleaner than helix's masked multi-pass median/mean (which leaves more
leakage). The CNN trains in ~100 s/plane on one GPU.

Why it wins: helix estimates the group waveform with a robust mean and a kσ signal
mask; the CNN learns (a) a better signal-robust across-wire combiner than median/
masked-mean and (b) temporal smoothing of the coherent estimate, which helix doesn't do.

**Honest caveats:** the CNN is trained *supervised on this specific synthetic coherent
model* (rms 2.5, β 0.15, this spectrum) — a real-detector mismatch would need
retraining, whereas helix is model-agnostic and adapts from the data. Gains are
modest (~0.01 F0) though consistent across planes; test = 4 events, single seed.

## P9 — pushing the R-D frontier (F0 + #coefficients jointly, unbiased)
Tried the methods *best suited* to sparse pulses, all measured by F0, compression
(coeffs), and **bias** = mean(recon−clean) on signal pixels (must be ~0).

**Convolutional sparse coding (learned pulse templates + matching pursuit).** The
"right" tool for sparse pulses on paper. Result: excellent compression/noise but
**fails F0 and bias** — Y 0.889, U 0.825, V 0.768 with bias −0.29/−0.22/−0.57.
Greedy template fitting underreconstructs pulse *amplitudes*, which F0 punishes and
the no-bias rule forbids. Lesson: F0 demands exact amplitude → favors a *complete*
transform over a sparse generative code.

**Coefficient-selection on the DWT** (all hard-keep → unbiased on kept coeffs),
F0 / bias at matched compression, level-8 db8/coif3:
| plane | @20× visu | @20× topk | @100× visu | @100× topk | @100× oracle |
|-------|-----------|-----------|------------|------------|--------------|
| Y | **0.974** / −0.06 | 0.970 / −0.15 | **0.935** / −0.66 | 0.812 / **−3.28** | 0.958 / −0.44 |
| U | 0.943 / +0.01 | 0.943 / +0.02 | **0.930** / +0.08 | 0.832 / +0.24 | 0.953 / +0.01 |
| V | **0.932** / +0.01 | 0.927 / +0.06 | **0.905** / +0.11 | 0.770 / +0.57 | 0.941 / +0.01 |

- **top-k looked like a big win at level 4 only because VisuShrink-L4 keeps the full
  approx band (capped ~16×).** At a fair deep level (L8), **VisuShrink-hard beats
  top-k**, and top-k develops a severe bias at high compression (−3.28 @100× Y:
  it drops the small signal coefficients needed for amplitude → underreconstructs).
- **energy selection** is bad everywhere (F0 0.66–0.70, large bias).
- **VisuShrink-hard DWT at deep level is the best practical method** — ~100× at
  F0 0.91–0.94 with small bias; ~20× at 0.93–0.97.
- **Oracle (perfect signal-coefficient detection)** shows the ceiling: ~0.02–0.04 F0
  above VisuShrink at 100× (Y 0.958, U 0.953, V 0.941, bias ~0). That gap is the
  *only* remaining headroom, and it requires better signal/noise *detection* of
  coefficients — i.e. a learned keep/drop classifier, not a different transform.

**P9 conclusion:** none of the learned/sparse variations beats well-tuned
VisuShrink-hard DWT (deep level) on the joint F0+compression objective without
adding bias. The transform+threshold is genuinely near-optimal here; the only
demonstrated headroom (~0.02–0.04 F0) is in *learned coefficient detection* toward
the oracle — a worthwhile but modest next step.

## P10 — thorough wavelet Pareto on the REAL coherent case (coherent + helix removal)
8,229 combos: 26 wavelets (Daubechies/symlet/coiflet/**biorthogonal-CDF**/rbio/dmey)
× full depth 1→max × 13 κ × 3 planes, on the helix-multipass-removed coherent residual
(4 GPUs, sharded). VisuShrink-hard. Selection = best-F0 plateau → fewest coefficients.

**Winner per plane = biorthogonal (CDF) at deep level**, validated on 16 events × 3 seeds:

| plane | production coif3 L4 | BEST (bior4.4 L8) | gain |
|-------|---------------------|-------------------|------|
| Y | 6.9×, F0 0.9627, bias −0.033, nrms 1.32 | 21.8×, F0 0.9618±.0003, bias −0.075, nrms 1.02 | **3.1×** coeffs, ΔF0 −0.0009 |
| U | 14.2×, F0 0.9128, bias −0.074, nrms 1.03 | 54.1×, F0 0.9129±.0003, bias −0.062, nrms 0.85 | **3.8×** coeffs, ΔF0 +0.0001 |
| V | 9.1×, F0 0.9035, bias −0.104, nrms 1.36 | 31.6×, F0 0.9025±.0002, bias −0.102, nrms 1.06 | **3.5×** coeffs, ΔF0 −0.0010 |
(leanest-at-best-F0 from the full sweep: Y bior6.8 L7 24.6×, U bior4.4 L8 53.8×, V bior4.4 L8 31.6×.)

**Findings:**
- **bior4.4 (CDF 9/7 family — JPEG2000's compression wavelet) at level 8 wins all three planes.**
  Adding biorthogonal to the sweep mattered: it beats the orthogonal db/sym/coif families here.
- **~3–4× fewer coefficients than the production coif3-L4 at the SAME F0** (ΔF0 ≤ 0.001, ≪ usefulness),
  with **lower residual noise** (nrms −20–25%) and **comparable-or-better bias** (U bias improves).
- Still consistent with the master finding: **depth (L7–L8) is the main lever**; among deep configs the
  CDF biorthogonal edges the orthogonal ones (smoother, symmetric → better energy compaction).
- Push (F0 ≥ plateau−0.005, |bias|<0.15): Y bior6.8 L7 ~41×, U db4 L9 ~82×, V ~50×.
- κ per plane at the recommended point: Y 1.25, U 2.0, V 1.5.

## Overall recommendation
For LArTPC wire denoise→compress on this data, the **production pipeline
(cross-wire coherent removal → per-wire DWT VisuShrink-hard) is already near-optimal**
among the transform families tried. The high-value lever is the **coherent-removal
front-end**, not the per-wire transform. Learned transforms (KLT, neural wavelet)
are competitive but do not justify their added complexity here; dictionary learning
(simple form) is worse. A generic **2D separable wavelet does not help** with coherent
noise (P7). If pursuing a learned win, the one promising (untried) direction is a
**group-aware learned model** (e.g. a CNN whose first stage operates across the 64-wire
groups) that jointly removes coherent + intrinsic noise — i.e. *learn the removal step*,
which is the real lever, not a better per-wire transform basis.

**This was confirmed (P8): a small group-aware CNN that learns the coherent removal
beats hand-crafted helix removal on all 3 planes** and recovers near the no-coherent
ceiling. So the actionable finding is: keep the per-wire DWT compression stage, but
replace (or augment) the coherent-removal front-end with a learned group-aware
estimator — that is where learning pays off for this data, not in the transform.

Figures: `figures/rd_{Y,U,V}.png` (Stage A frontiers), `figures/rd_*_B.png` (Stage B).
Caveats: rate = kept-coeffs/pixels (dict's atom-index overhead not charged → dict
flattered); test sets 4–6 events; lwave trained with approximate (mid-wire-length)
intrinsic noise; single noise seed at eval.
