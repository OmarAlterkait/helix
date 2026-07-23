# How should learning rate scale with batch size for AdamW? (Literature review + recommendation for the coeff-FM)

**Date:** 2026-07-22.
**Context:** AdamW (β1=0.9, β2=0.95, wd=0.05), hard grad-clip at 1.0 with raw grad norms 3–9 (clip active on 20–100% of steps), µP (d_base=128, d=512, ~50M params), MAE-style Gaussian-NLL pretraining on wavelet-coefficient tokens, DP batch B = 2–8 events (~32k tokens/event), measured gradient noise scale **B_noise ≈ 31 events late** (bootstrap CI 18–38), ~80 early. Validated anchor: **lr = 1.6e-3 at B = 2** (crowned by a 150k-step sweep; an 8e-3 winner from a 30k sweep was worst at 150k).

**TL;DR:** For Adam-family optimizers the well-supported rule below the noise scale is **square-root-or-milder** scaling of LR with batch size — not linear, and specifically not `lr ∝ B/(B+B_noise)`, which is an SGD-on-a-quadratic result that even its own authors did not claim for Adam. Our B=4 failure at lr=3e-3 is exactly what the Adam literature predicts. Recommended: **lr(B=4) ≈ 2.2e-3** (surge-rule 2.1e-3, sqrt-rule 2.3e-3; medium-high confidence), **lr(B=8) ≈ 2.7e-3** (surge-rule with B_noise=31; sqrt gives 3.2e-3 as a ceiling; medium confidence). Keeping 1.6e-3 unchanged is a safe floor at both batch sizes. Details, derivations, and caveats below.

---

## 1. Theory

### 1.1 SGD baselines: where linear scaling and B/(B+B_noise) come from

- **McCandlish, Kaplan, Amodei & OpenAI Dota team, "An Empirical Model of Large-Batch Training," 2018, arXiv:1812.06162.** For *plain SGD* on a local quadratic, one step with a noisy gradient gives the optimal step
  `η_opt(B) = η_max / (1 + B_noise/B)`, with `η_max = |G|²/(GᵀHG)` and `B_noise = tr(HΣ)/(GᵀHG)`
  (Σ = per-sample gradient covariance). For B ≪ B_noise this reduces to *linear* LR scaling and predicts near-perfect step-for-compute exchange; B_noise is the critical batch size where returns halve.
  **Crucial caveat from the same paper:** for **Adam and RMSProp they explicitly report that the *measured* optimal LR follows a power law `η_opt ∝ B^α` with α between roughly 0.5 and 1.0 depending on task, then flattens** (discussed in their Appendix E.2). So `B/(B+B_noise)` was never the Adam rule even in the paper that introduced B_noise. The noise scale itself remains a good predictor of the *critical batch size* (compute/time tradeoff), which is a separate quantity from the LR rule.
- **Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," 2017, arXiv:1706.02677.** Linear scaling rule + warmup, for SGD+momentum on ImageNet. Empirically holds up to ~8k images, then breaks. SGD-specific.
- **Krizhevsky, "One weird trick for parallelizing convolutional neural networks," 2014, arXiv:1404.5997** and **Hoffer, Hubara & Soudry, "Train longer, generalize better," 2017, arXiv:1705.08741.** Both advocate **√B scaling** (Hoffer: to keep the per-step weight-update covariance constant). Also SGD, but the variance-matching argument is the ancestor of the modern Adam sqrt rule.
- **Smith et al., "Don't Decay the Learning Rate, Increase the Batch Size," 2018, arXiv:1711.00489.** Frames B and η as interchangeable through the SGD "noise temperature" `g ∝ ηN/B` — again SGD-specific.
- **Zhang, Li, Nado, Martens, Sachdeva, Dahl, Shallue & Grosse, "Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model," NeurIPS 2019, arXiv:1907.04164.** The NQM reproduces the whole phenomenology: perfect scaling → diminishing returns → plateau; **preconditioned optimizers (Adam, K-FAC) have substantially *larger* critical batch sizes than momentum-SGD**, and momentum only helps above a threshold B. Predicts the optimal LR curve flattens near/above the critical batch.

### 1.2 Adam: approximate scale invariance, the SDE sqrt rule, and the surge phenomenon

**Scale invariance.** Adam's update `Δθ ∝ m/(√v + ε)` is (for ε → 0) exactly invariant to multiplying *all* gradients by a constant scalar, and approximately invariant to a *slowly varying* scalar (changes on timescales longer than the EMA horizons 1/(1−β1) ≈ 10 and 1/(1−β2) ≈ 20 steps for our betas). This is why Adam's effective step size is nearly decoupled from the raw gradient *magnitude*; batch size affects Adam through the **signal-to-noise ratio** of the gradient, not its norm. A recent formalization: **"Why Adam Works Better with β1 = β2: The Missing Gradient Scale Invariance Principle," 2026, arXiv:2601.21739** (Adam is first-order gradient-scale-invariant iff β1=β2 — our 0.9/0.95 is near this regime).

**The SDE sqrt rule.** **Malladi, Lyu, Panigrahi & Arora, "On the SDEs and Scaling Rules for Adaptive Gradient Algorithms," NeurIPS 2022, arXiv:2205.10287.** First rigorous SDE approximations for RMSprop/Adam, yielding the **square-root scaling rule**: when B → κB,

- `η′ = η√κ`
- `β1′ = 1 − κ(1−β1)`, `β2′ = 1 − κ(1−β2)`
- `ε′ = ε/√κ`

**Validity regime:** the derivation *requires the noise-dominated regime* — "σ ≫ ‖∇f‖", i.e., per-example gradient noise dominates the mean gradient; they show no SDE approximation exists otherwise. **Our measured B_noise ≈ 31 ≫ B = 2–8 is precisely the statement that we are in this regime** (B_noise ≫ B ⇔ minibatch gradient variance ≫ squared mean), so the sqrt rule's assumptions are *met* in our setup, which is not always true for large-batch LLM runs. Note the rule also rescales the betas; at κ=2–4 with β2=0.95 the β2 correction (0.95 → 0.90/0.80) is secondary but real (see §5).

**The surge phenomenon (Adam's non-monotone LR-vs-B curve).** **Li, Liu, Zhou et al., "Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling," NeurIPS 2024, arXiv:2405.14578.** Extending the McCandlish one-step analysis to sign-like/Adam-style updates, they derive

`η_opt(B) = η_max / [ ½ ( √(B_noise/B) + √(B/B_noise) ) ]`

- For **B ≪ B_noise: η_opt ≈ (2η_max/√B_noise)·√B — square-root scaling** (agrees with Malladi).
- **Peak at B ≈ B_noise**, after which the optimal LR *decreases* — opposite of SGD's monotone rule. The peak location moves right as training progresses (B_noise grows during training in their experiments, as in McCandlish).
- Verified across CV and NLP tasks.

This is the cleanest theoretical object for our situation because it is parametrized by exactly the quantity we measured. With B_noise=31 events, the predicted LR *ratios* relative to B=2 are:

| B | surge factor (B_noise=31) | surge factor (B_noise=80) | pure √B factor | linear/McCandlish factor |
|---|---|---|---|---|
| 2 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4 | 1.33 | 1.38 | 1.41 | 1.89 |
| 8 | 1.69 | 1.87 | 2.00 | 3.39 |

(The surge factor is milder than pure √B because B=4–8 is already a non-negligible fraction of B_noise=31. Using the early-training B_noise≈80 moves it toward pure sqrt; the recommendation is insensitive to which we use.)

**Why sqrt and not linear, intuitively.** In the noise-dominated regime Adam behaves like (soft) sign descent / normalized descent: the *magnitude* of the update is ≈ η per coordinate regardless of gradient size, so doubling B does not double the useful step the way it does for SGD; it improves the *directional* SNR of the update by √2, and the tolerable step grows with that SNR. Larger B also mechanically shrinks `√v` (v ≈ noise variance ∝ 1/B when noise dominates), so **Adam already self-amplifies its effective step by ≈ √κ when B grows — part of the "LR increase" SGD would need is built in**. This is made precise by **Wang & Aitchison, "Batch size invariant Adam," 2024, arXiv:2402.18824 (OPT-ML @ NeurIPS 2024):** standard Adam averages micro-batch gradients *then* squares, making v batch-size dependent; in the regime where **gradient variance dominates the squared mean gradient (≡ our regime), √v ∝ 1/√B and standard Adam is *approximately batch-size invariant on its own* under the sqrt-LR rule**; their modified Adam (square-then-average) is batch-size invariant with *no* LR change at all. Either way: **nothing in the Adam-specific theory supports linear scaling below B_noise, and one credible line supports scaling even weaker than sqrt.**

### 1.3 What gradient clipping does and does not change

Setup fact: clipping is applied to the **raw gradient before Adam**, as a global-norm rescale `g → g·min(1, c/‖g‖)` — a *scalar* multiplication of the entire gradient vector.

- **Because Adam is (approximately) invariant to scalar gradient rescalings (§1.2), norm-clipping ahead of AdamW mostly does *not* cap the effective step size** — unlike SGD, where clipping directly bounds the update. Both m and v accumulate the same scalar; the ratio m/√v is unchanged for a constant rescale and only mildly affected when the rescale factor fluctuates faster than the EMA horizons. With clip=1.0 and raw norms 3–9, we are applying a fluctuating 3–9× attenuation; its main real effects on AdamW are (i) **relative re-weighting inside the EMAs** — high-norm (spiky) steps are down-weighted relative to quiet steps, a heavy-tail robustness mechanism, and (ii) a slight interaction with ε (clipped gradients are 3–9× smaller entering v; with ε at default 1e-8 this is harmless at our scale, cf. §1.4 on Adam-atan2).
- Theory of clipping per se: **Zhang, He, Sra & Jadbabaie, "Why Gradient Clipping Accelerates Training: A Theoretical Justification for Adaptivity," ICLR 2020, arXiv:1905.11881** (clipped GD ≈ adaptive step under relaxed (L0,L1)-smoothness) and **Zhang, Karimireddy, Veit, Kim, Reddi, Kumar & Sra, "Why are Adaptive Methods Good for Attention Models?," NeurIPS 2020, arXiv:1912.03194** (transformer gradient noise is heavy-tailed; clipping/adaptivity is what fixes SGD). Both concern SGD-side benefits; **neither implies a change to the LR-vs-B exponent for Adam**, because Adam's normalization already supplies the adaptivity clipping would otherwise provide.
- **Answer to "does active clipping mean LR should barely scale with B?": No, but it slightly strengthens the case for the conservative (≤ sqrt) end.** The batch-size dependence of Adam's optimum comes through the gradient SNR (surge/SDE analysis), which clipping does not remove. What clipping *does* mean: (a) the raw grad norm will shrink as B grows (noise-dominated ⇒ ‖g‖ roughly ∝ 1/√B), so the **clip fraction will drop at B=4/8**; monitor it — if it collapses toward 0, the spike-suppression you currently enjoy at B=2 is gone, and running a *hotter* LR simultaneously is compounding risk (this is a concrete reason to prefer surge-factor over pure-sqrt at B=8); (b) since clipping is already quasi-normalizing your update spikes, the marginal value of extra LR is smaller than in an unclipped run.
- Empirical stability context: **Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities," ICLR 2024 (oral), arXiv:2309.14322** — with standard mitigations in place, **"increasing the batch size from 256 to 512 or 1024 does not meaningfully change learning rate sensitivity"**: the loss-vs-LR basin barely moves with batch at small scale. Direct empirical support that for AdamW at B ≪ B_crit the optimum shifts *weakly* with B — i.e., between "no scaling" and "sqrt," not linear.

### 1.4 µP and batch-size transfer: what is actually claimed and how reliable it is

- **Yang, Hu et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer," NeurIPS 2021, arXiv:2203.03466.** µP's theory covers transfer **across width** only. Transfer across **batch size** (also depth, sequence length, training steps) is listed in their Table 1 with an asterisk = **"empirically validated only,"** shown on Wikitext-2 (Appendix G.2.1) with minimum thresholds ("HPs generally transfer ... if some minimum ... batch size (e.g., 32) ... are met"). µP's implicit claim for batch is therefore **η_opt(B) ≈ const over moderate B ranges** — an empirical observation on one small benchmark, not a theorem. Note also their transfer-across-*training-time* claim is empirical-only and is contradicted by the horizon studies in §2.3.
- **Lingle, "A Large-Scale Exploration of µ-Transfer," 2024, arXiv:2404.05728.** µ-transfer across width mostly works up to 10B params, with named failure cases (e.g., large batch + decoupled WD interactions); tests of 4× batch up/down at fixed tokens found transfer approximately holds, with a minimum-batch caveat.
- **Everett et al., "Scaling Exponents Across Parameterizations and Optimizers," ICML 2024, arXiv:2407.05872.** Tens of thousands of models: all four parametrizations can transfer LR across width once per-layer exponents and Adam-ε handling are correct (fitted base-LR exponents ≈ −0.02…−0.06, i.e., near-invariant); introduces Adam-atan2 to kill ε-underflow. Message: µP-style transfer works but is fragile to implementation details — consistent with our in-house finding that µP silently shrank the effective hidden LR ~4×.
- **Zhou et al., "How to Set the Learning Rate for Large-Scale Pre-training?," 2026, arXiv:2601.05049.** At production LLM scale, a fitted `lr(N,D) = 38.46·N^(−0.222)·D^(−0.351)` "Fitting Paradigm" **consistently outperforms µTransfer**. Optimal LR falls with *data* (horizon) even faster than with model size — µP has no story for this axis (it's the empirical-only asterisk again).
- **Bottom line for us:** µP does **not** license "keep base LR fixed while changing B" as a theorem; its batch/duration transfer claims are the weakest part of the package, and our own µP mishap (memory: effective hidden LR shrank 4×; prod runs under-trained) is in line with the fragility reported by Lingle/Everett. Treat µP as a width-transfer device only; tune LR-vs-B empirically at d=512.

### 1.5 Weight decay: the λ·η timescale, and what changing B does to it

- **Wang & Aitchison, "How to set AdamW's weight decay as you scale model and dataset size," ICML 2025, arXiv:2405.13698.** AdamW weights are an EMA of recent updates with timescale `τ_iter = 1/(η·λ)` iterations; **the optimal timescale measured in *epochs/data* is roughly constant** across model and dataset size. Therefore λ should be set through τ, not copied. Practical bounds: τ not ≪ 1 epoch and not ≫ total epochs.
- **Bergsma, Dey, Gosal, Gray, Soboleva & Hestness (Cerebras), "Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training," NeurIPS 2025, arXiv:2505.13738.** Defines the AdamW timescale `τ = B/(η·λ·D)`; finds **optimal λ scales *linearly* with B at fixed N, D and fixed η** (equivalently: keep τ fixed), τ_opt follows a power law in tokens-per-parameter, and both B_opt and B_crit follow **power laws in dataset size D, independent of N**.
- **Implication for us:** if η is raised by the sqrt/surge factor when B doubles, keeping the *data-denominated* timescale constant requires `η·λ ∝ B`, i.e., **λ ∝ B/η ≈ √B**: wd 0.05 → ~0.07 at B=4 → ~0.10 at B=8. This is a second-order correction (τ changes by only 1.4–2× if you leave wd alone) but the direction is unambiguous in both papers, and leaving λ fixed while shortening the run in steps makes regularization effectively *weaker* per epoch. Low-effort option: fold it in only if you change B permanently.

---

## 2. Empirics

### 2.1 Large measurement studies

- **Shallue, Lee, Antognini, Sohl-Dickstein, Frostig & Dahl, "Measuring the Effects of Data Parallelism on Neural Network Training," JMLR 2019, arXiv:1811.03600.** ~72M loss measurements, 35 workloads. Universal three-phase steps-vs-batch curve (perfect scaling → diminishing returns → plateau), but **"extremely large variation between workloads"; no single LR heuristic (linear, sqrt, or other) predicted the optimum across workloads** — the optimal LR had to be re-tuned per batch size, and the *optimal-LR-vs-B* curve was frequently sublinear and flattened well before the maximal useful batch. The canonical warning against trusting any closed-form lr(B) without a check.
- **Zhang et al. (NQM), NeurIPS 2019, arXiv:1907.04164** (see §1.1): Adam extends the perfect-scaling region vs momentum SGD; optimal LR flattens near critical batch.
- **You et al., "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes" (LAMB), ICLR 2020, arXiv:1904.00962:** BERT/Adam-family large-batch training empirically used **sqrt-of-batch LR scaling**, corroborated as superior for Adam in follow-ups (this is the empirical rule the Zhang CBS paper, below, adopts as background).

### 2.2 Critical batch size in transformer pretraining

- **Zhang, Morwani, Vyas, Wu, Zou, Ghai, Foster & Kakade, "How Does Critical Batch Size Scale in Pre-training?," ICLR 2025, arXiv:2410.21676.** 85M–1.2B autoregressive LMs on C4, per-B LR sweeps. **CBS scales with *data*, not model size:** `B* ≈ 22.9·D^0.47` (D in billions of tokens; ≈ N^0.087 at fixed data). Below CBS, doubling B ≈ halves steps to target loss — *provided LR is re-tuned per B*. They adopt/confirm the Adam-vs-SGD distinction (sqrt-type LR adjustment for Adam) and use EMA + schedule tricks to decouple horizon from tuning.
- **Merrill, Arora, Groeneveld & Hajishirzi, "Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training," NeurIPS 2025, arXiv:2505.23971.** Directly measures CBS during OLMo 1B/7B training: **CBS ≈ 0 at init, rises rapidly, then plateaus** (largely model-size independent). Motivates **batch-size warmup** (start small-B, grow B as CBS grows): trained OLMo-1B to slightly better loss with 43% fewer optimizer steps. Relevant to us: early training has a much smaller usable batch than late training.
- **Bergsma et al. "Power Lines," arXiv:2505.13738** (§1.5): B_opt and B_crit are power laws in D, independent of N — consistent with Zhang et al.
- **McCandlish et al. 1812.06162 / Kaplan et al., "Scaling Laws for Neural Language Models," 2020, arXiv:2001.08361:** B_noise ≈ B_crit measured from loss; for LMs B_crit ~ 1–2M tokens mid-training and grows as loss falls. Our measured B_noise ≈ 31 events ≈ 1.0M tokens sits exactly in this canonical range — the measurement looks trustworthy in scale. (One anomaly: our B_noise *fell* from ~80 early to ~31 late, whereas McCandlish/Merrill find noise scale/CBS *growing* during training. Possible causes: warmup/µP transients inflating the early estimate, or nonstationary curriculum. Worth re-measuring, but the recommendation below is insensitive to which value is used.)

### 2.3 The horizon effect ("LR sweep mirage") is real and documented

- **Bjorck et al., "Scaling Optimal LR Across Token Horizons," ICLR 2025, arXiv:2409.19913.** 250+ runs: **optimal LR falls with training-token horizon as a power law**; short-horizon sweeps systematically crown LRs that are too hot for long runs (they argue LLaMA-1's LR was too high for its horizon). This is precisely our observation (a): 8e-3 won at 30k steps and lost at 150k.
- **Zhou et al., arXiv:2601.05049** (§1.4): fitted `lr_opt ∝ D^(−0.35)` — an independent quantitative confirmation.
- **Porian, Wortsman, Jitsev, Schmidt & Carmon, "Resolving Discrepancies in Compute-Optimal Scaling of Language Models," NeurIPS 2024, arXiv:2406.19146.** The Kaplan-vs-Chinchilla disagreement is largely *hyperparameter tuning that must change with scale* (LR, batch, warmup). Two findings matter for us: (i) they fit power laws for optimal LR and batch vs N — and find an optimal batch size *below which performance degrades* (contradicting "smaller batch is always data-efficient"); (ii) **at small batch sizes, β2=0.95 is suboptimal: "as the batch size gets smaller, the squared gradients become noisier, and AdamW requires more smoothing to obtain a correct denominator"** — β2 ∈ {0.99, 0.999} was needed for clean trends at their smallest batches. Our B=2–8-events regime with per-step gradients this noisy (B ≪ B_noise) is exactly the regime where **β2=0.95 may be hurting**, independent of LR.
- **Hägele, Bakouch, Kosson, Ben Allal, von Werra & Jaggi, "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations," NeurIPS 2024, arXiv:2405.18392.** With constant-LR + decay (WSD-style) schedules, a large fraction of the final-loss gap closes only during the decay phase; runs at too-hot stable LR "catch up" during the anneal. This is our observation (b) — convergence back-loaded into the anneal is the signature of a stable-phase LR above the horizon-appropriate optimum.
- **Zhou et al., "How to Set the Batch Size for Large-Scale Pre-training?," 2026, arXiv:2601.05034.** Revises the McCandlish E(S) tradeoff for WSD schedules; defines B_min/B_opt and a dynamic batch schedule; further evidence that the 2018 quadratic-SGD framework does not transfer unmodified to modern AdamW pipelines.

---

## 3. Contradictions and open questions

1. **Sqrt (Malladi, You/LAMB, Surge) vs. near-constant (TP5 batch transfer, Wortsman, batch-invariant-Adam view).** Both camps have evidence. They are less contradictory than they look: over a 2–4× batch range, √B predicts a 1.4–2× LR shift, while Adam's loss-vs-LR basin is typically ~2–4× wide near the optimum — so "optimum moved by sqrt" and "old LR still near-optimal" can both be true. The camps genuinely diverge only over ≥8–16× batch changes. **Nobody credible defends linear scaling for Adam below B_noise.**
2. **Is `lr(B) = lr_max·B/(B+B_noise)` valid for AdamW? No.** It is derived for plain SGD on a quadratic; McCandlish's own appendix reports α ∈ [0.5, 1.0) power laws for Adam; Malladi's SDE and the surge derivation both replace it with sqrt-type behavior; our in-house B=4@3e-3 result is a direct falsification. B_noise remains valuable — but as a *critical-batch/compute-tradeoff* estimate and as the *location of the surge peak*, not as an LR formula.
3. **Does µP make LR batch-transferable?** Claimed empirically (asterisked) in TP5, weakly supported at 4× changes (Lingle), unsupported theoretically, and challenged at scale (Zhou 2601.05049). Open question; do not rely on it.
4. **How does persistent clipping shift the optimum?** No paper directly measures LR-vs-B curves under always-on norm clipping with AdamW. The scale-invariance argument (§1.3) says the effect on the *optimum's location* should be small; the effect on the *safe ceiling* (spike protection) is real and B-dependent (clip fraction falls with B). This is a genuine gap in the literature — treat our clip-fraction telemetry as the measurement.
5. **Nonstationary B_noise.** Literature says noise scale grows during training; ours shrank (80 → 31). Unexplained; re-measure with the same estimator at several checkpoints and with per-event (not per-token) accounting, since our "examples" are whole 32k-token events with strong intra-event correlation — the *event*-level noise scale is the right unit for DP scaling, which is what was measured.
6. **MAE/NLL objective vs. LM loss.** All transformer-pretraining empirics above are next-token LMs. Gaussian-NLL on heteroscedastic targets adds a loss-curvature dependence on predicted σ that none of these papers cover. Directionally the noise-scale framework still applies (it only needs gradient statistics, which we measured), but exponents fitted on LLMs (e.g., D^0.47 CBS) should not be assumed numerically.

---

## 4. Reading our three observations through the literature

- **(a) 30k sweep crowned 8e-3, worst at 150k.** Textbook horizon effect (Bjorck 2409.19913; Zhou 2601.05049: lr_opt ∝ D^(−0.35); Porian 2406.19146 on scale-dependent tuning). A 5× horizon change moving the optimum by ~(5)^0.35 ≈ 1.8× is enough to flip a sweep ranking. Consequence: only trust sweeps at (or extrapolated to) the deployment horizon.
- **(b) B=4 @ 3e-3 worse than B=2 @ 1.6e-3 at matched steps, convergence back-loaded into the anneal.** At matched steps B=4 sees 2× the data and lower gradient noise; with a correctly-set LR it should dominate. That it lost indicts the LR, and 3e-3 is 1.4× above what the Adam-specific rules prescribe (sqrt: 2.26e-3; surge with B_noise=31: 2.13e-3). Back-loading into the anneal is the too-hot-stable-LR signature (Hägele 2405.18392). This run is an in-house falsification of near-linear scaling and is *quantitatively consistent* with sqrt/surge.
- **(c) High peak LR degraded a downstream probe on the smaller dataset.** Consistent with the stability literature (Wortsman 2309.14322): near/above the loss-tolerable LR edge, representation quality degrades before the loss diverges; smaller data = longer effective epochs = hotter effective temperature per epoch (also a wd-timescale effect, §1.5). Another argument for the conservative end.

---

## 5. Recommendation

**Rule adopted: surge/sqrt scaling anchored at the validated (B=2, lr=1.6e-3, 150k-step) point, with B_noise=31 events.** `lr(B) = 1.6e-3 · f(B)/f(2)`, `f(B) = [½(√(31/B) + √(B/31))]^(−1)`.

| B (events) | linear / B/(B+B_noise) | pure √B | **surge (recommended)** | safe floor |
|---|---|---|---|---|
| 2 | 1.6e-3 (anchor) | 1.6e-3 | **1.6e-3** | — |
| 4 | 3.0e-3 — **do not use** (falsified in-house) | 2.3e-3 | **2.1–2.3e-3 → use 2.2e-3** | 1.6e-3 |
| 8 | 5.4e-3 — do not use | 3.2e-3 (ceiling) | **2.6–2.8e-3 → use 2.7e-3** | 1.6e-3 |

**Confidence.**
- **B=4 at 2.2e-3: medium-high (~75%)** that it beats both 1.6e-3 and 3e-3 at the 150k horizon. Supported by Malladi (assumptions verifiably met: B ≪ B_noise), Surge (quantitative, uses our measured B_noise), LAMB-lineage empirics, and consistent with the failure of 3e-3. Downside risk is small: 2.2e-3 is only 1.4× the proven-safe 1.6e-3.
- **B=8 at 2.7e-3: medium (~60%)**, range 2.4–3.2e-3. B=8 is ~26% of B_noise (or 10% if B_noise=80), so the sub-sqrt bend of the surge curve is starting to matter and pure sqrt (3.2e-3) should be treated as a ceiling, not a target. Clip fraction will be substantially lower at B=8 (raw norms shrink ~2× vs B=2), removing spike protection exactly as LR rises — prefer the lower end if clip fraction drops below ~5–10%.
- **Keeping 1.6e-3 at any B: safe (>90% no regression vs B=2 at matched *samples*)** — the Wortsman/TP5/batch-invariant-Adam evidence says the basin moves slowly; the cost is only a modest per-step slowdown (you give up ≤1.4–2× of the theoretical step-count saving, and at matched *tokens* you likely lose little). Use this if no budget exists for even one confirmation run.

**Secondary knobs (do these with, not instead of, the LR change):**
1. **β2:** two independent lines say β2=0.95 is likely mis-set in our small-B, high-noise regime — Porian et al. (needed 0.99–0.999 at small batch because squared-gradient noise corrupts the denominator) and Malladi's rule (β2 should move *with* B; equivalently our B=2 baseline sits at an effectively too-small 1/(1−β2)=20-step horizon for its noise level). Cheap A/B worth running: **β2=0.99 at B=2–4, LR unchanged.** If it helps at B=2, re-anchor before scaling B.
2. **Weight decay:** to keep the Wang–Aitchison/Power-Lines timescale `τ ∝ B/(ηλ)` constant in data units when B and η rise: **λ: 0.05 → ~0.07 (B=4) → ~0.10 (B=8)** (λ ∝ B/η ≈ √B). Second-order; skip if runs are short relative to τ anyway.
3. **Warmup:** hold warmup fixed in *samples*, not steps, when raising B (Porian identified warmup mis-scaling as a scaling-law-distorting factor; Merrill's rising-CBS result implies early training tolerates less batch/LR — a slightly *longer* sample-denominated warmup at B=8 is prudent).
4. **ε hygiene:** clipped gradients entering v are 3–9× smaller; with η·√κ scaling Malladi also prescribes ε′=ε/√κ. At ε=1e-8 this is almost surely irrelevant at 50M params, but if per-coordinate `√v` telemetry approaches 1e-6, switch to ε=1e-9 or Adam-atan2 (Everett 2407.05872).

**Validation protocol (horizon-mirage-proof):** compare B=4@2.2e-3 vs B=2@1.6e-3 at **matched samples** (75k steps vs 150k) *including the full anneal in both*, not matched steps mid-schedule — Hägele 2405.18392 shows mid-run rankings before the decay are unreliable. Track: clip fraction (expect ~2× fewer clipped steps at B=4; collapse ⇒ back off LR), the downstream probe (observation (c) makes it the canary), and re-measure B_noise at 2–3 checkpoints.

---

## 6. Source list

1. McCandlish, Kaplan, Amodei, et al. "An Empirical Model of Large-Batch Training." 2018. arXiv:1812.06162.
2. Goyal, Dollár, Girshick, et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." 2017. arXiv:1706.02677.
3. Krizhevsky. "One weird trick for parallelizing convolutional neural networks." 2014. arXiv:1404.5997.
4. Hoffer, Hubara, Soudry. "Train longer, generalize better." NeurIPS 2017. arXiv:1705.08741.
5. Smith, Kindermans, Ying, Le. "Don't Decay the Learning Rate, Increase the Batch Size." ICLR 2018. arXiv:1711.00489.
6. Zhang, Li, Nado, Martens, Sachdeva, Dahl, Shallue, Grosse. "Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model." NeurIPS 2019. arXiv:1907.04164.
7. Malladi, Lyu, Panigrahi, Arora. "On the SDEs and Scaling Rules for Adaptive Gradient Algorithms." NeurIPS 2022. arXiv:2205.10287.
8. Li, et al. "Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling." NeurIPS 2024. arXiv:2405.14578.
9. Wang, Aitchison. "Batch size invariant Adam." 2024. arXiv:2402.18824.
10. "Why Adam Works Better with β1=β2: The Missing Gradient Scale Invariance Principle." 2026. arXiv:2601.21739.
11. Zhang, He, Sra, Jadbabaie. "Why Gradient Clipping Accelerates Training: A Theoretical Justification for Adaptivity." ICLR 2020. arXiv:1905.11881.
12. Zhang, Karimireddy, Veit, Kim, Reddi, Kumar, Sra. "Why are Adaptive Methods Good for Attention Models?" NeurIPS 2020. arXiv:1912.03194.
13. Wortsman, Liu, Xiao, et al. "Small-scale proxies for large-scale Transformer training instabilities." ICLR 2024. arXiv:2309.14322.
14. Yang, Hu, Babuschkin, et al. "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." NeurIPS 2021. arXiv:2203.03466.
15. Lingle. "A Large-Scale Exploration of µ-Transfer." 2024. arXiv:2404.05728.
16. Everett, Xiao, Wortsman, et al. "Scaling Exponents Across Parameterizations and Optimizers." ICML 2024. arXiv:2407.05872.
17. Wang, Aitchison. "How to set AdamW's weight decay as you scale model and dataset size." ICML 2025. arXiv:2405.13698.
18. Bergsma, Dey, Gosal, Gray, Soboleva, Hestness. "Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training." NeurIPS 2025. arXiv:2505.13738.
19. Shallue, Lee, Antognini, Sohl-Dickstein, Frostig, Dahl. "Measuring the Effects of Data Parallelism on Neural Network Training." JMLR 2019. arXiv:1811.03600.
20. You, Li, Reddi, et al. "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes." ICLR 2020. arXiv:1904.00962.
21. Zhang, Morwani, Vyas, Wu, Zou, Ghai, Foster, Kakade. "How Does Critical Batch Size Scale in Pre-training?" ICLR 2025. arXiv:2410.21676.
22. Merrill, Arora, Groeneveld, Hajishirzi. "Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training." NeurIPS 2025. arXiv:2505.23971.
23. Kaplan, McCandlish, et al. "Scaling Laws for Neural Language Models." 2020. arXiv:2001.08361.
24. Bjorck, et al. "Scaling Optimal LR Across Token Horizons." ICLR 2025. arXiv:2409.19913.
25. Porian, Wortsman, Jitsev, Schmidt, Carmon. "Resolving Discrepancies in Compute-Optimal Scaling of Language Models." NeurIPS 2024. arXiv:2406.19146.
26. Hägele, Bakouch, Kosson, Ben Allal, von Werra, Jaggi. "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations." NeurIPS 2024. arXiv:2405.18392.
27. Zhou, Huang, Xing, Zhang, Peng, Guo, Qiu. "How to Set the Batch Size for Large-Scale Pre-training?" 2026. arXiv:2601.05034.
28. Zhou, et al. "How to Set the Learning Rate for Large-Scale Pre-training?" 2026. arXiv:2601.05049.
