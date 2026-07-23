# LR schedules for unknown training horizons: theory, empirics, and a protocol for the coeff-FM

*Literature review, 2026-07. Context: 50M-param transformer, AdamW + µP, MAE-style SSL pretraining on
TPC wavelet-coefficient data. Observed failures: (i) cosine horizon guessed too short at 300k / 600k /
1.2M steps — val metrics (triangulation recon, 3D probe) still improving when the cosine hit its floor;
(ii) most metric improvement happens in the anneal, not the high-LR plateau; (iii) final checkpoint
(long time at the LR floor) shows a downstream 3D-probe DROP vs ~200k steps earlier (0.783 → 0.726);
(iv) optimal peak LR appears horizon-dependent (short sweeps crown LRs that lose at long horizons).*

---

## 1. TL;DR

Every one of the four observations above is a **documented, expected property of cosine schedules**, not
a quirk of our setup. The literature's answer is unambiguous:

1. **Replace cosine with warmup–stable–decay (WSD, a.k.a. trapezoidal / constant + cooldown).**
   A constant LR with a branched cooldown of ~10–20% of elapsed steps matches or beats a cosine tuned
   to the same horizon, at every horizon, from one run — Hägele et al. 2024 (arXiv:2405.18392), MiniCPM
   (arXiv:2404.06395), DeepSeek-LLM (arXiv:2401.02954), Zhai et al. ViT (arXiv:2106.04560).
2. **The anneal-phase "sudden improvement" is real optimization progress being *revealed*, not created
   at the end** — the river-valley analysis of Wen et al. 2024 (arXiv:2410.05192). This is exactly our
   observation (b), and it means high-LR-phase metrics *understate* the model; monitor an EMA or
   branched cooldowns instead.
3. **Optimal peak LR genuinely shrinks with horizon** — LR\*(D) ∝ D^(−0.32) empirically (Bjorck et al.,
   arXiv:2409.19913), ∝ T^(−1/2) in the matching convex theory (Schaipp et al., arXiv:2501.18965) —
   **and µP does NOT fix this** (explicitly shown by Bjorck et al.). Our observation (d) is this law.
4. **Long dwell at a tiny LR floor is known to be useless-to-harmful.** Decay should go (essentially)
   to zero and then training should STOP (Bergsma et al. "Straight to Zero", arXiv:2502.15938);
   continued low-temperature training overfits the pretext and degrades transfer ("catastrophic
   overtraining", Springer et al., arXiv:2503.19206). Our observation (c) — the 0.783 → 0.726 probe
   drop — is the expected signature.
5. **EMA/weight averaging is a free *monitor* and a partial substitute, not a full replacement** for a
   cooldown for AdamW (Hägele et al. 2024; Song et al. "Through the River", arXiv:2507.09846).
   Schedule-Free AdamW is the one averaging method shown to genuinely remove the schedule, but it is
   momentum-sensitive and changes the optimizer (Defazio et al., arXiv:2405.15682; arXiv:2507.09846).

The concrete protocol is in §8.

---

## 2. Why cosine fails for open-ended horizons

Cosine's defining pathology: the schedule at *every* step depends on the total horizon T. Hoffmann et
al.'s Chinchilla paper (arXiv:2203.15556, App. A) already showed the cosine cycle length must match the
actual training duration — overestimating the cycle by ≳25% measurably degrades loss, and
underestimating it (our case: training wants to continue past T) leaves you at the floor with no good
continuation. Extending a finished cosine requires **re-warming**, and Ibrahim et al.
(arXiv:2403.08763; also Gupta et al., arXiv:2308.04014) show re-warming the LR causes an immediate loss
spike (≈+0.1 val loss at peak for a 3e-4 re-warm, *even on the identical data distribution*) that takes
significant compute to re-absorb. Cyclic cosine / SGDR-style restarts (Loshchilov & Hutter,
arXiv:1608.03983) institutionalize this spike; Wen et al. (arXiv:2410.05192) show cyclic-cosine is
dominated by WSD-S (decay-and-resume) for producing multiple checkpoints along one run.

A second, subtler failure: with cosine, a peak-LR sweep at horizon T₁ is *also* a schedule sweep — you
cannot separate "this peak LR is better" from "this decay trajectory is better for this horizon."
Hägele et al. (arXiv:2405.18392) argue this conflation contaminated scaling-law research (Chinchilla
needed a fresh cosine run per horizon; with WSD, one constant-LR run + branched cooldowns reproduces
the same frontier for **< half the FLOPs** — 5.59×10²³ → 2.36×10²³ for the Chinchilla suite). Porian et
al. (arXiv:2406.19146) trace much of the Kaplan-vs-Chinchilla exponent discrepancy to exactly these
optimization-hyperparameter/horizon interactions (warmup length, β₂, scale-dependent LR tuning), and
find, notably, that careful decay is *not* essential for the scaling law itself — but tuned LR is.

## 3. WSD / trapezoidal schedules: the empirical record

**MiniCPM (Hu et al. 2024, arXiv:2404.06395)** introduced the name WSD: warmup → long constant
("stable") phase → short decay. Purpose-built for "continuous training with an un-predefined
pre-training token number": any stable-phase checkpoint can be branched into a decay to obtain a
deployable model, and the stable branch continues indefinitely (data can also be re-weighted at the
decay start, e.g. their annealing-with-SFT-data recipe). They report the loss **drops sharply during
the decay stage** and that a decay of ~10% of the total tokens suffices; with WSD they could do
data-scaling studies from a single stable run.

**Hägele et al. 2024 (arXiv:2405.18392, NeurIPS spotlight)** is the most careful head-to-head. Key
quantitative findings (124M–1B models):

- Constant LR + cooldown **matches cosine's loss-vs-compute frontier at every horizon** from one run.
- **Cooldown length:** the cooldown surpasses a tuned cosine at ~10–20% of total steps; benefits
  plateau at ~20%. For long runs the *fraction* shrinks: in a 200k-step run, a 10k-step cooldown (5%)
  "almost perfectly matches cosine."
- **Decay shape:** `1 − sqrt(x)` (concave, fast-then-flat) consistently beats linear, cosine, mirror-
  cosine, square — the gap grows with run length. Decay target is 0 (or ~0).
- **Peak LR:** the optimal *constant* LR sits at about **half the optimal cosine peak LR** (cosine
  spends most of its time below peak; a constant schedule at cosine's peak is too hot).
- **SWA:** stochastic weight averaging on the constant-LR trajectory "improves performance along the
  trajectory" for free, but **does not close the full gap to an explicit cooldown**.

**DeepSeek-LLM (arXiv:2401.02954)** used a discrete version (multi-step: constant to 80% of tokens,
drop to 31.6%, then 10% of peak for the last 10%): final loss "essentially consistent" with cosine,
chosen specifically because stage-1 checkpoints are reusable for continual training.

**Zhai et al., Scaling Vision Transformers (arXiv:2106.04560)** — the "infinite schedule" ancestor:
reciprocal-sqrt (or constant) middle section plus a **linear cooldown branched at multiple points**, so
one long run yields models at many durations. Their rsqrt middle section is a mild built-in decay;
modern LLM practice simplified it to constant. The same idea underlies "infinite LR schedules" for
continual pretraining (Beyond Cosine Decay, arXiv:2503.02844).

Adoption signal: SmolLM2 (arXiv:2502.02737), Apertus (arXiv:2509.14233), DeepSeek, MiniCPM, Hugging
Face scaling work all train with WSD-family schedules now; this is no longer exotic.

## 4. Theory: why the decay produces the sudden drop

**River-valley picture (Wen et al. 2024, arXiv:2410.05192).** Pretraining loss looks like a deep,
narrow valley with a gently sloping river at the bottom. At high constant LR the iterate bounces
between valley walls ("mountain" directions, sharp curvature) while drifting fast along the river
(flat direction). Measured loss is dominated by the bouncing, so it plateaus **even while genuine
progress accumulates along the river**. Decaying the LR suppresses the oscillation and lets the iterate
settle to the river bed, *revealing* the accumulated progress as a sharp loss drop. Corollaries, all of
which match our observations:

- Metric improvement concentrates in the anneal (our observation (b)) — but the *cause* is the entire
  stable phase; a longer stable phase yields a lower post-anneal loss.
- A short decay (~10%) recovers most of the drop; the decay's job is transverse (mountain-direction),
  which is fast, not longitudinal (river), which is slow.
- WSD-S: you can decay, evaluate, and *resume from the decayed checkpoint* (not the pre-decay one),
  keeping one main branch — outperforms cyclic-cosine for getting checkpoints at many budgets.
- Follow-up (Dremov et al., "Training Dynamics of the Cooldown Stage", arXiv:2508.01483, TMLR 2025):
  loss-landscape visualizations support the river picture; cooldown *shape* trades off
  exploration/exploitation (bias–variance); shapes that keep some exploration early in the cooldown
  win (consistent with 1−sqrt); and **raising AdamW β₂ during the cooldown consistently helps** —
  cooldown-phase hyperparameters deserve real tuning.

**SGD-as-annealing / temperature picture.** Classical view: constant-LR SGD samples a stationary
distribution whose "temperature" scales with η (Mandt et al., SGD as Approximate Bayesian Inference,
arXiv:1704.04289; Smith & Le, arXiv:1710.06451; Smith et al., "Don't decay the learning rate, increase
the batch size", arXiv:1711.00489). High LR = high temperature = broad exploration + implicit
regularization toward flat/simple solutions (Li, Wei & Ma, arXiv:1907.04595 show an initial large-LR
phase is itself a regularizer that improves generalization); annealing = cooling into the basin.
This view also predicts the *harm* of long tiny-LR tails: at near-zero temperature the sampler settles
into ever-sharper, sample-specific structure (memorization) — see §7.

**AdamW-as-EMA (Bergsma et al., "Straight to Zero", arXiv:2502.15938).** AdamW with weight decay is an
exponential moving average of recent weight *updates* with timescale 1/(ηλ). Early training needs a
short timescale (escape init), late training needs a long one (average over more gradient noise).
Linear **decay-to-zero (D2Z)** optimally interpolates the two, and empirically beats both cosine and
"10× decay" (i.e., decay stopping at a 10%-of-peak floor) across model sizes/batches/datasets, with the
margin *growing* with tokens-per-parameter (their 610M model at 80 TPP with D2Z beats the same model at
200 TPP with 10×-decay — a 60% compute saving). Message for us: **the floor of the decay should be ~0,
and the "value" of late training is noise-averaging, which a floor-dwell does badly and averaging does
well.** Companion "Power Lines" paper (arXiv:2505.13738) shows the optimal (η, λ) pair is governed by
the EMA timescale being a fixed fraction of the horizon — another mechanism making optimal LR
horizon-dependent at fixed λ.

**Convex theory quantitatively matches (Schaipp et al. 2025, arXiv:2501.18965).** The last-iterate
bound for constant-then-linear-cooldown schedules in non-smooth stochastic convex optimization
reproduces, term for term, the empirical phenomenology: the cooldown kills the log(T) term in the bound
(the theoretical image of the "sudden drop"), the bound-optimal cooldown fraction is ~O(20%), and the
bound-optimal base LR scales as **T^(−1/2)** — the theoretical anchor for horizon-dependent LR. They
use the bound to *transfer* the optimal LR when extending a run (continued training with
√-rescaled LR), with real gains on 124M/210M Llama-style models.

**Loss-curve laws that formalize "area under the schedule".** Tissue et al. (arXiv:2408.11029) fit
L(s) = L₀ + A·S₁^(−α) − C·S₂ where S₁ = cumulative LR area ("forward progress" ≈ river) and S₂ =
cumulative *annealing* area ("revealed progress" ≈ descent to river bed); one or two runs suffice to
predict the loss of any schedule, and the law's optimum is WSD-shaped. Luo et al.'s multi-power law
(arXiv:2503.12811, ICLR 2025) generalizes this, predicts unseen schedules accurately, and its
*optimized* schedule is again constant + concave decay, beating both cosine and hand-tuned WSD.
These two papers are the practical tools if we ever want to *derive* our own decay shape from our own
loss curves.

## 5. Horizon-dependent optimal peak LR (our observation (d))

- **Bjorck et al., "Scaling Optimal LR Across Token Horizons" (arXiv:2409.19913, ICLR 2025):** large-
  scale sweeps show optimal LR follows **LR\*(D) = B·D^(−β), β ≈ 0.32** (joint law LR\* ∝ N^(−0.23)·D^(−0.32)).
  Rule of thumb: LR\*(D₂) ≈ LR\*(D₁)·(D₂/D₁)^(−0.32). Two findings directly on-point for us:
  (1) **µP does not transfer LR across token horizons** — µTransfer handles width, not data/steps; a µP
  sweep at 300k steps is still miscalibrated for 1.2M steps. (2) Post-hoc case study: Llama-1's LR was
  ~2.5× too high for its 1T-token horizon by their law — "short sweeps crown too-hot LRs" is common
  even at frontier labs.
- **Porian et al. (arXiv:2406.19146):** independent evidence that optimal LR (and batch) shift with
  scale/horizon, and that β₂ tuning matters at small batch; hyperparameter-horizon coupling is strong
  enough to flip scaling-law exponents.
- **Schaipp et al. (arXiv:2501.18965):** theory says η\* ∝ T^(−1/2) for the last iterate of WSD-type
  schedules (empirical β ≈ 0.32 is milder, plausibly because curvature/adaptivity effects break the
  non-smooth worst case).
- **Hägele et al.:** at *matched* horizon, optimal constant LR ≈ **0.5× optimal cosine peak**. So
  translating a cosine-tuned LR to WSD requires halving *and* horizon-correcting.
- Practical synthesis: with WSD the problem mostly dissolves — the stable LR is a *mild* preference
  rather than a horizon-committed choice, because the anneal (which you branch whenever you like) does
  the horizon adaptation. Wen et al. additionally show a too-hot stable LR still makes river progress;
  it's recoverable by the decay, whereas cosine bakes the error into every step. Under-shooting LR is
  safer than over-shooting for the *stable* phase since averaging/decay can't undo instability.

## 6. Averaging: EMA, SWA, LAWA, Schedule-Free

What averaging **does**: it cancels the mountain-direction oscillation *without* touching the LR, so an
average over recent iterates sits near the river bed while the raw iterate keeps exploring hot. It is
the natural *monitor* of "what would I get if I annealed right now."

- **SWA on the stable phase** (Hägele et al., arXiv:2405.18392): free improvement along the trajectory
  at every scale, but a persistent gap to a true cooldown remains.
- **EWA during stable phase** (Song et al., "Through the River", arXiv:2507.09846, NeurIPS 2025): for
  standard AdamW, exponential weight averaging gives only modest gains vs the sharp decay-phase drop —
  averaging alone did **not** reproduce the cooldown in their setting. (Consistent with Hägele: partial,
  not full, substitution. The river interpretation: averaging handles the transverse component, but the
  decay phase also makes genuine extra river progress at intermediate LRs.)
- **LAWA** (Kaddour, arXiv:2209.14981): uniform average of the k latest epoch-spaced checkpoints gives
  dozens-of-epochs speedups (ResNet-50/ImageNet, RoBERTa/WikiText). **Sanyal et al.**
  (arXiv:2306.03241, COLM 2024): averaging checkpoints sampled with *large spacing* while training at
  *high LR* beats both EMA and SWA baselines for LLM pretraining (nanoGPT, Pythia 1B–12B) — gains are
  largest exactly in the high-LR regime WSD keeps you in.
- **EMA of weights** (Morales-Brotons et al., arXiv:2411.18704, TMLR 2024): systematic study; EMA
  models generalize better, are better calibrated and better teachers, and — key phrase — **"EMA
  requires less learning rate decay"** because averaging performs part of the noise reduction that
  decay otherwise must. **Busbridge et al., "How to Scale Your EMA"** (arXiv:2307.13813, NeurIPS 2023)
  gives the momentum scaling rule (keep EMA timescale fixed in *data* units when batch size changes).
- **Schedule-Free AdamW** (Defazio et al., "The Road Less Scheduled", arXiv:2405.15682): replaces the
  schedule with an interpolation of Polyak–Ruppert averaging and primal averaging (gradient evaluated
  at an interpolated point); no horizon input at all; won the MLCommons AlgoPerf 2024 self-tuning
  track. "Through the River" (arXiv:2507.09846) explains *why* it works — SF implicitly averages over
  momentum iterates with a window that widens with T, so the x-iterate rides the river bed
  continuously; decays applied on top of SF give almost no further drop, and tuned SF matches or beats
  WSD final loss. Caveats: sensitive to β₁ (0.95 vs 0.9 matters; bad β₁ falls off the river), needed
  their refinement (decoupling parameter C) to match cosine at 2M-token batches, and it changes the
  optimizer state/µP story. Powerful, but a bigger change than adopting WSD.

**Bottom line:** run an EMA (or LAWA) *on top of* the WSD stable phase as a free proxy-annealed model
for evaluation and as insurance, but keep the explicit branched cooldown as the mechanism that
produces release checkpoints. Averaging complements annealing; for AdamW it does not replace it.

## 7. Is lingering at the LR floor harmful? (our observation (c))

Three independent lines say yes, or at best "pure waste":

1. **Decay-target evidence.** Bergsma et al. (arXiv:2502.15938): decaying to a 10%-of-peak floor is
   strictly worse than decaying to zero, increasingly so at high tokens-per-parameter; the useful role
   of the late phase is *averaging*, and a long constant dwell at small-but-nonzero LR is an
   inefficient, drifting average. Hägele et al. and MiniCPM likewise decay to ~0 and **stop** — no
   published WSD recipe includes a floor-dwell.
2. **River-valley view.** Once the decay has collapsed the transverse oscillation, extra steps at
   near-zero LR make negligible river progress (progress speed ∝ LR) — there is nothing left for those
   steps to do *except* fit fine-grained, sample-specific structure.
3. **Low-temperature overfitting / transfer damage.** The SGD-temperature literature (Mandt
   arXiv:1704.04289; Smith & Le arXiv:1710.06451; Li–Wei–Ma arXiv:1907.04595) predicts that at tiny LR
   the implicit regularization of gradient noise is gone and the model descends into sharper, more
   memorizing solutions. Empirically, on the transfer side: Springer et al., "Overtrained Language
   Models Are Harder to Fine-Tune" (arXiv:2503.19206) document **catastrophic overtraining** — models'
   sensitivity grows with continued pretraining and downstream/adapted performance can *degrade* with
   more pretraining past a point; and improvements in pretraining loss are known to not always yield
   downstream gains (negative transfer). In SSL specifically, probe metrics are routinely
   non-monotonic in late training while the pretext loss still creeps down.

**Explaining our 0.783 → 0.726 probe drop.** The final checkpoint sat ~200k steps at the cosine floor.
During that time: (i) essentially zero river progress (LR ≈ 0), so no upside; (ii) the pretext (masked
MSE on wavelet coeffs, finite cached event set, many epochs) keeps being optimized at near-zero
temperature, so the encoder redistributes capacity toward dataset-/noise-specific reconstruction detail
— exactly the features a 3D-triangulation probe does *not* use — i.e., pretext overfitting +
low-temperature sharpening; (iii) our own MSE-ceiling finding compounds this: under a calibrated
posterior, late MSE gains come from mean-regression sharpening, which *shrinks* feature variance along
uncertain directions that the geometric probe reads. Note this drop is **not** evidence against
annealing — the anneal itself (the ~10–20% cooldown) is where the gains appear; the damage came from
*continuing to train after the anneal was finished*. The fix is structural: decay to ~0, evaluate the
downstream metric on checkpoints *throughout the decay*, take the best, and stop — never park at a
floor. (Also: keep an EMA; the EMA checkpoint is far less exposed to terminal drift.)

## 8. Concrete protocol for the coeff-FM

**Schedule.** Warmup (keep current warmup) → **constant LR** → branched **1−sqrt cooldown to 0**.
No terminal floor-dwell, ever. Implement cooldown as a *branch* from a stable-phase checkpoint; the
stable branch keeps running (MiniCPM / WSD-S pattern). If we want to fold the annealed progress back
in, resume the stable branch *from the decayed checkpoint* (WSD-S, arXiv:2410.05192) rather than
re-warming a finished cosine (avoids the arXiv:2403.08763 re-warm spike).

**Stable LR value.** Start from the best cosine peak LR found at the *longest* completed horizon
(1.2M steps), then apply two corrections:
- ×0.5 for constant-vs-cosine (Hägele et al.);
- ×(D_target/D_sweep)^(−0.32) if extrapolating to a longer intended horizon (Bjorck et al.).
Given the µP finding (our runs under-trained, effective LR too low; suggested lr ≈ 1.6e-3) and that µP
does **not** license horizon transfer (arXiv:2409.19913), do one cheap 3-point stable-LR check
{0.5×, 1×, 2×} around the corrected value using ~50k-step runs *each ended with a short 10k cooldown*
— compare *post-cooldown* metrics, never plateau metrics (river-valley: plateau loss misranks LRs).
Prefer the lower LR on a tie (stable-phase LR errors on the hot side are only partially recoverable).

**EMA (monitoring + insurance).** Maintain one EMA of weights during the stable phase with half-life
≈ 5–10k steps at our batch size (β_EMA ≈ 1 − ln2/7000 ≈ 0.9999); rescale the momentum if batch size
changes (Busbridge et al., arXiv:2307.13813). Evaluate triangulation-recon and the 3D probe on the
**EMA weights** (and optionally a LAWA average of the last ~5 checkpoints spaced 2–5k steps, per
Sanyal arXiv:2306.03241) — this is the low-noise proxy for "what a cooldown would give now."
Expect the EMA to close much but not all of the gap to a real cooldown (arXiv:2405.18392,
arXiv:2507.09846).

**When to anneal (trigger).** Two-tier rule:
1. *Periodic probe cooldowns:* every ~150–250k stable steps, branch a short cooldown (length
   max(20k, 5% of elapsed steps)) and evaluate the downstream metrics at its end. Cost is a few % of
   compute; gives the honest learning curve that cosine never shows, and each one is a usable release
   candidate. (This is exactly Hägele et al.'s Chinchilla-at-half-compute recipe and Zhai et al.'s
   multi-cooldown ViT recipe.)
2. *Stop/anneal criterion:* fit the improvement of the post-cooldown (or EMA) downstream metric per
   200k steps; trigger the final cooldown when projected improvement over the next window drops below
   its eval noise (or the compute deadline arrives — with WSD any time is a valid time). Do NOT trigger
   off the raw high-LR training curve.

**Final cooldown.** Length ≈ 10–20% of *total elapsed* steps (20% if we can afford it — benefits
plateau there, arXiv:2405.18392; 10% is MiniCPM's default and nearly matches), shape 1−sqrt, decay to
0. Consider raising β₂ (e.g. 0.95 → 0.99) during the cooldown (Dremov et al., arXiv:2508.01483).
Checkpoint every few k steps during the cooldown, evaluate the 3D probe + triangulation on each
(raw and EMA weights), and **select the best, which will likely not be the last** — then stop.
No post-cooldown training at the floor.

**Optional, later:** (a) fit Tissue et al.'s annealing law (arXiv:2408.11029) or the multi-power law
(arXiv:2503.12811) to our existing 300k/600k/1.2M cosine curves — two runs suffice — to predict loss
for candidate schedules before spending GPU; (b) trial Schedule-Free AdamW (arXiv:2405.15682,
arXiv:2507.09846) on a 100k-step side run with β₁ = 0.95 as a no-horizon-input alternative; adopt only
if it matches the WSD branch post-cooldown downstream metrics.

---

## 9. Citations

- Hägele, Bakouch, Kosson, Ben Allal, von Werra, Jaggi (2024). *Scaling Laws and Compute-Optimal
  Training Beyond Fixed Training Durations.* NeurIPS 2024. arXiv:2405.18392.
- Hu et al. (2024). *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training
  Strategies.* arXiv:2404.06395.
- DeepSeek-AI (Bi et al.) (2024). *DeepSeek LLM: Scaling Open-Source Language Models with
  Longtermism.* arXiv:2401.02954.
- Zhai, Kolesnikov, Houlsby, Beyer (2021/2022). *Scaling Vision Transformers.* CVPR 2022.
  arXiv:2106.04560.
- Wen, Li, Wang, Hall, Liang, Ma (2024). *Understanding Warmup-Stable-Decay Learning Rates: A River
  Valley Loss Landscape Perspective.* arXiv:2410.05192.
- Dremov, Hägele, Kosson, Jaggi (2025). *Training Dynamics of the Cooldown Stage in Warmup-Stable-
  Decay Learning Rate Scheduler.* TMLR 2025. arXiv:2508.01483.
- Bjorck et al. (2024). *Scaling Optimal LR Across Token Horizons.* ICLR 2025. arXiv:2409.19913.
- Porian, Wortsman, Jitsev, Schmidt, Carmon (2024). *Resolving Discrepancies in Compute-Optimal
  Scaling of Language Models.* NeurIPS 2024. arXiv:2406.19146.
- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla).
  arXiv:2203.15556.
- Bergsma, Dey et al. (Cerebras) (2025). *Straight to Zero: Why Linearly Decaying the Learning Rate to
  Zero Works Best for LLMs.* arXiv:2502.15938. Companion: *Power Lines: Scaling Laws for Weight Decay
  and Batch Size in LLM Pre-training.* arXiv:2505.13738.
- Schaipp, Hägele, Taylor, Simsekli, Bach (2025). *The Surprising Agreement Between Convex
  Optimization Theory and Learning-Rate Scheduling for Large Model Training.* arXiv:2501.18965.
- Tissue, Wang, Wang (2024). *Scaling Law with Learning Rate Annealing.* arXiv:2408.11029.
- Luo et al. (2025). *A Multi-Power Law for Loss Curve Prediction Across Learning Rate Schedules.*
  ICLR 2025. arXiv:2503.12811.
- Defazio et al. (2024). *The Road Less Scheduled* (Schedule-Free). arXiv:2405.15682.
- Song, Baek, Ahn, Yun (2025). *Through the River: Understanding the Benefit of Schedule-Free Methods
  for Language Model Training.* NeurIPS 2025. arXiv:2507.09846.
- Kaddour (2022). *Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight
  Averaging* (LAWA). NeurIPS HITY 2022. arXiv:2209.14981.
- Sanyal, Neerkaje, Kaddour, Kumar, Sanghavi (2023). *Early Weight Averaging meets High Learning Rates
  for LLM Pre-training.* COLM 2024. arXiv:2306.03241.
- Izmailov et al. (2018). *Averaging Weights Leads to Wider Optima and Better Generalization* (SWA).
  UAI 2018. arXiv:1803.05407.
- Morales-Brotons, Vogels, Hendrikx (2024). *Exponential Moving Average of Weights in Deep Learning:
  Dynamics and Benefits.* TMLR 2024. arXiv:2411.18704.
- Busbridge, Ramapuram, Ablin et al. (2023). *How to Scale Your EMA.* NeurIPS 2023. arXiv:2307.13813.
- Loshchilov, Hutter (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017.
  arXiv:1608.03983.
- Ibrahim, Thérien et al. (2024). *Simple and Scalable Strategies to Continually Pre-train Large
  Language Models.* arXiv:2403.08763. Also Gupta et al. (2023), *How to (re)warm your model?*
  arXiv:2308.04014.
- Springer, Goyal et al. (2025). *Overtrained Language Models Are Harder to Fine-Tune.*
  arXiv:2503.19206.
- Mandt, Hoffman, Blei (2017). *Stochastic Gradient Descent as Approximate Bayesian Inference.*
  JMLR. arXiv:1704.04289.
- Smith, Le (2018). *A Bayesian Perspective on Generalization and Stochastic Gradient Descent.*
  ICLR 2018. arXiv:1710.06451. Also Smith et al., *Don't Decay the Learning Rate, Increase the Batch
  Size.* ICLR 2018. arXiv:1711.00489.
- Li, Wei, Ma (2019). *Towards Explaining the Regularization Effect of Initial Large Learning Rate.*
  NeurIPS 2019. arXiv:1907.04595.
- (Continual-pretraining schedules:) *Beyond Cosine Decay: On the Effectiveness of Infinite Learning
  Rate Schedules for Continual Pre-training.* arXiv:2503.02844.
