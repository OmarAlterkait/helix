# Optimization Choices vs. Representation Quality (as Distinct from Pretraining Loss)

**Literature review + recommendations for the coeff-FM runs** (50M CrossMAE-style Gaussian-NLL
pretraining on wavelet-coefficient tokens; model selection on a 3D-position linear/MLP probe
[fisher-r] and a whole-plane-masking "triangulation" reconstruction subtask).

Phenomena to explain:

- **(a)** small-data run (20k events, ~110 epochs): probe metric **dips at peak LR** (3e-3),
  recovers during anneal; 4x data → no dip (data acted as regularizer).
- **(b)** the hard triangulation subtask improves mostly **during LR decay** ("crystallizes late").
- **(c)** train loss is healthy throughout — **loss and representation quality decouple**.
- **(d)** final checkpoint after a long dwell at floor LR is **worse on the probe** than earlier
  checkpoints.

Verdict up front: **all four observations are instances of documented phenomena**, and the
literature converges on a consistent recipe: moderate-large peak LR (just above the stability
threshold, scaled down when data is scarce), WSD-style schedule with real decay but *no long dwell
at the floor*, EMA/weight averaging as a free "virtual anneal" and as the object you probe, and
model selection by probes on (EMA'd) snapshots rather than by pretraining loss.

---

## 1. Probe–loss decoupling: pretraining loss is not a sufficient statistic

The strongest direct result is **Liu, Xie, Li, Ma, "Same Pre-training Loss, Better Downstream:
Implicit Bias Matters for Language Models," ICML 2023, [arXiv:2210.14199](https://arxiv.org/abs/2210.14199)**.
They construct language models that reach the *same* (near-minimal) pretraining loss yet differ
substantially downstream; among same-loss models, **flatness of the found minimum correlates with
downstream quality** while loss does not, and SGD noise implicitly biases toward the flatter
(better-transferring) minima. Directly: which minimum the optimizer selects — a function of LR,
batch, noise — controls representation quality at fixed loss. This is the cleanest published
justification for observation (c) and for *not* selecting checkpoints on pretraining loss.

Complementary theory: **Saunshi, Ash, Goel, Misra, Zhang, Arora, Kakade, Krishnamurthy,
"Understanding Contrastive Learning Requires Incorporating Inductive Biases," ICML 2022,
[arXiv:2202.14037](https://arxiv.org/abs/2202.14037)** proves that analyses relating SSL objective
value to downstream performance while ignoring the function class and *training algorithm* can be
vacuous: two representations with identical SSL loss can have arbitrarily different downstream
performance. The mapping loss→probe is fundamentally mediated by optimization-dependent inductive
bias.

Empirical SSL-evaluation corollaries:

- **Garrido, Balestriero, Najman, LeCun, "RankMe," ICML 2023,
  [arXiv:2210.02885](https://arxiv.org/abs/2210.02885)**: across 110 joint-embedding SSL models,
  a label-free *geometric* property of the embedding (effective rank / spectral entropy) predicts
  downstream performance where the training loss does not. Representation-quality metrics live on
  a different axis than the objective.
- **Islam, Chen, Panda, Karlinsky, Radke, Feris, "A Broad Study on the Transferability of Visual
  Representations with Contrastive Learning," ICCV 2021,
  [arXiv:2103.13517](https://arxiv.org/abs/2103.13517)**: transferable representation is mostly
  learned in the *early* phase of pretraining; late epochs keep improving the pretraining objective
  while adding mostly source-specific structure (transfer gains ~saturate after ~80 epochs).
  I.e., late-training loss improvements need not show up in — and can trade against — probes.
- **Chen, Shwartz-Ziv, Cho, Leavitt, Saphra, "Sudden Drops in the Loss: Syntax Acquisition, Phase
  Transitions, and Simplicity Bias in MLMs," ICLR 2024,
  [arXiv:2309.07311](https://arxiv.org/abs/2309.07311)**: internal-structure metrics (syntactic
  attention structure, probe-style capability measures) evolve through phase transitions that are
  only loosely coupled to the smooth loss curve; interpretable capabilities can even *compete* with
  each other during training. Tracking structure probes over training — exactly what we do — is the
  advocated methodology.

**Takeaway:** the decoupling in (c) is expected, well-documented in both theory and SSL practice,
and selecting on probes-not-loss is the literature-endorsed procedure.

---

## 2. Large LR: when it builds features and when it damages them

### 2.1 The "large LR is a feature-quality regularizer" line

- **Li, Wei, Ma, "Towards Explaining the Regularization Effect of Initial Large Learning Rate in
  Training Neural Networks," NeurIPS 2019, [arXiv:1907.04595](https://arxiv.org/abs/1907.04595)**:
  provable two-layer setting where large-initial-LR + anneal generalizes better than small LR
  throughout. Mechanism: a small LR lets the network lock onto easy-to-fit, low-noise "clean"
  patterns *first* and memorize the rest; a large LR forces it to first learn noise-robust,
  harder-to-fit patterns and only pick up the fine, easily-memorized structure *after annealing*.
  Note the corollary: **the fine-grained patterns are, by design, learned during/after the anneal**
  — the theoretical seed of "crystallizes late" (observation b).
- **Lewkowycz, Bahri, Dyer, Sohl-Dickstein, Gur-Ari, "The large learning rate phase of deep
  learning: the catapult mechanism," 2020,
  [arXiv:2003.02218](https://arxiv.org/abs/2003.02218)**: three LR regimes — lazy / catapult /
  divergent. In the catapult (large-LR) phase, loss spikes then drops into *flatter* minima with
  better generalization; the flatness benefit comes from the LR itself, not SGD noise. Large LR is
  what pushes training out of the lazy/NTK regime into feature learning (cf. **Chizat, Oyallon,
  Bach, "On Lazy Training in Differentiable Programming," NeurIPS 2019,
  [arXiv:1812.07956](https://arxiv.org/abs/1812.07956)**; **Yang & Hu, "Feature Learning in
  Infinite-Width Neural Networks," ICML 2021, [arXiv:2011.14522](https://arxiv.org/abs/2011.14522)**).
- **Cohen, Kaur, Li, Kolter, Talwalkar, "Gradient Descent on Neural Networks Typically Occurs at
  the Edge of Stability," ICLR 2021, [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)**:
  sharpness rises ("progressive sharpening") until it saturates at ≈2/η; the LR therefore acts as a
  *ceiling on the sharpness of what can be learned*. Big η ⇒ forced flatness while at peak; tiny η
  ⇒ the ceiling is lifted and training can enter very sharp regions (relevant to §5).
- **Andriushchenko, Varre, Pillaud-Vivien, Flammarion, "SGD with Large Step Sizes Learns Sparse
  Features," ICML 2023, [arXiv:2210.05337](https://arxiv.org/abs/2210.05337)**: large steps make
  the iterates bounce across the valley; the resulting loss-stabilization phase hides a slow drift
  that implicitly selects *sparse, task-relevant* features, with no explicit regularizer. Longer
  time at large LR = stronger feature-sparsifying bias.

### 2.2 …but only in a narrow band; too large damages representations

- **Sadrtdinov, Kodryan, Pokonechny, Lobacheva, Vetrov, "Where Do Large Learning Rates Lead Us?,"
  NeurIPS 2024, [arXiv:2410.22113](https://arxiv.org/abs/2410.22113)** — the most operationally
  useful result: only a **narrow range of initial LRs slightly above the convergence threshold**
  is optimal once followed by decay or weight averaging. In that band, training (i) settles into a
  basin containing only high-quality minima and (ii) learns a sparse, relevant feature set.
  *Smaller* LRs try to learn all features at once and generalize worse; *larger* LRs "fail to
  detect a basin with good solutions and extract meaningful patterns." So peak LR has an interior
  optimum for representation quality, and the failure mode on the high side is precisely
  *feature/representation* damage that need not appear in train loss.
- The **stability gap** literature documents the transient version of this: under a sudden increase
  in effective plasticity, performance on established capabilities *drops sharply then recovers* —
  visible only with per-iteration evaluation, invisible in end-of-phase metrics
  (**De Lange, van de Ven, Tuytelaars, "Continual evaluation for lifelong learning: identifying
  the stability gap," ICLR 2023, [arXiv:2205.13452](https://arxiv.org/abs/2205.13452)**;
  **Harun & Kanan, "Overcoming the Stability Gap in Continual Learning," 2023,
  [arXiv:2306.01904](https://arxiv.org/abs/2306.01904)**, who find lowering peak LR and modified
  schedules shrink the gap). Our mid-training probe dip at peak LR is the same signature: an
  emergent capability (cross-plane 3D structure) is partially *churned away* while the optimizer is
  at maximum temperature, and re-forms as the temperature drops.

### 2.3 Interaction with data volume — data as regularizer

Why does 4x data remove the dip at the same peak LR?

- In the Li–Wei–Ma mechanism the harm of any LR regime is expressed through **memorization of the
  small set**: with little data, the easily-memorized component is large relative to the
  generalizing component, so the representation is more fragile at high temperature; more data
  shrinks the memorizable fraction and stabilizes the feature-learning path
  ([arXiv:1907.04595](https://arxiv.org/abs/1907.04595)).
- **Hernandez, Brown, …, Kaplan, McCandlish, "Scaling Laws and Interpretability of Learning from
  Repeated Data," 2022, [arXiv:2205.10487](https://arxiv.org/abs/2205.10487)**: repeating data
  produces a strong double-descent-like degradation, and — key for us — the damage falls
  **disproportionately on generalization-carrying internal structure (e.g., induction heads)**
  while test loss moves comparatively little: a mechanistic demonstration that repetition-driven
  memorization destroys representation quality *before* it is visible in loss. At ~110 epochs on
  20k events we are deep in this regime; at 4x data, ~4x fewer effective repetitions.
- **Muennighoff, Rush, Barak, et al., "Scaling Data-Constrained Language Models," NeurIPS 2023,
  [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)**: ≤4 epochs of repetition ≈ fresh data;
  returns decay and hit ~zero near ~16 epochs. 110 epochs is far beyond the useful-repetition
  frontier for the *objective*, meaning most late optimization pressure on the small set is
  memorization pressure.
- **Liu, Michaud, Tegmark, "Omnigrok: Grokking Beyond Algorithmic Data," ICLR 2023,
  [arXiv:2210.01117](https://arxiv.org/abs/2210.01117)** make the train/test landscape mismatch
  (their "LU mechanism") explicitly *data-size dependent*: small train sets create a large
  mismatch between the train-loss-minimizing region and the generalizing region; growing the
  dataset shrinks it. Same geometry: with 20k events the high-LR trajectory can wander into
  train-good/probe-bad territory; with 80k it can't stray as far.
- In vision-MAE specifically, small-data pretraining is known to overfit through excess decoder
  capacity ("Masked autoencoders are effective solution to transformer data-hungry," 2022,
  [arXiv:2212.05677](https://arxiv.org/abs/2212.05677)) — decoder shrinkage is the standard
  small-data MAE mitigation.

**Takeaway for (a):** the dip is a stability-gap/representation-churn transient at maximum
plasticity, amplified by heavy data repetition; both lowering peak LR slightly and adding data are
literature-supported cures, and the cure we observed (data) is the one predicted to be strictly
better (it also raises the ceiling, not just stability).

---

## 3. Why hard/rare structure crystallizes during the anneal

- **Wen, Li, Wang, Hall, Liang, Ma, "Understanding Warmup-Stable-Decay Learning Rates: A River
  Valley Loss Landscape Perspective," 2024, [arXiv:2410.05192](https://arxiv.org/abs/2410.05192)**:
  pretraining loss is a river valley; at high stable LR the iterates bounce between steep walls
  making progress only along the river (coarse, deterministic structure), while the **decay phase
  descends to the valley floor, which is where the fine/stochastic structure is fit** — producing
  the characteristic sharp loss drop and late capability gains during cooldown. Whole-plane
  triangulation — an inference that requires precise integration of cross-plane detail — is
  exactly "fine structure along sharp directions," expected to materialize during decay.
- **Hägele, Bakouch, Kosson, Ben Allal, von Werra, Jaggi, "Scaling Laws and Compute-Optimal
  Training Beyond Fixed Training Durations," NeurIPS 2024,
  [arXiv:2405.18392](https://arxiv.org/abs/2405.18392)**: constant-LR + short cooldown matches
  cosine; most of the end-quality appears during the cooldown; and **EMA/SWA along the trajectory
  recovers much of the cooldown gain without decaying at all** (a "free" measurement of
  post-anneal quality from any point of the run).
- **Li–Wei–Ma** (above): fine-grained/low-margin patterns are provably learned only after
  annealing ([arXiv:1907.04595](https://arxiv.org/abs/1907.04595)).
- **Grokking**: capabilities emerging long after train loss saturates, gated by implicit/explicit
  regularization — **Power, Burda, Edwards, Babuschkin, Misra, "Grokking: Generalization Beyond
  Overfitting on Small Algorithmic Datasets," 2022,
  [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)**; Omnigrok ties the delay to weight-norm
  dynamics and data size ([arXiv:2210.01117](https://arxiv.org/abs/2210.01117)). Small-data +
  hard-subtask late emergence is squarely in this family.
- **Chen et al. 2024** ([arXiv:2309.07311](https://arxiv.org/abs/2309.07311)): capabilities arrive
  as abrupt phase transitions at specific points of training, not smoothly with loss.

**Takeaway for (b):** late crystallization of the triangulation metric during LR decay matches the
river-valley/WSD picture quantitatively (sharp late gains during cooldown) and Li–Wei–Ma
qualitatively. Practical consequence: **the anneal is not optional and its placement/length is a
first-class hyperparameter; and you can sample "post-anneal quality" mid-run cheaply via EMA or
branched mini-cooldowns.**

---

## 4. EMA and weight averaging: effect on transfer and probes

- **Izmailov, Podoprikhin, Garipov, Vetrov, Wilson, "Averaging Weights Leads to Wider Optima and
  Better Generalization" (SWA), UAI 2018, [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)**:
  averaging along the trajectory finds flatter solutions and better generalization than the last
  iterate — and per §1 (Liu et al.), flatness is the quantity that tracks downstream quality at
  fixed loss.
- **Morales-Brotons, Vogels, Hendrikx, "Exponential Moving Average of Weights in Deep Learning:
  Dynamics and Benefits," TMLR 2024, [arXiv:2411.18704](https://arxiv.org/abs/2411.18704)**: EMA
  weights generalize better and specifically improve **transfer learning**, calibration,
  prediction consistency, and noisy-label robustness; EMA *reduces the amount of LR decay needed*
  (averaging substitutes for annealing); EMA is strong early — which is why it works as a teacher.
- **Sanyal et al., "Early Weight Averaging meets High Learning Rates for LLM Pre-training," COLM
  2024, [arXiv:2306.03241](https://arxiv.org/abs/2306.03241)** (LAWA): averaging a few
  well-spaced checkpoints acts as a surrogate LR decay; **gains are largest precisely when
  training with high LR**, and averaged models beat both EMA-with-small-window and SWA baselines
  on zero-shot downstream metrics. High-LR + averaging is a *combination* strategy: keep the
  feature-learning benefits of the high temperature, remove its noise via averaging.
- **Wortsman et al., "Model soups," ICML 2022,
  [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)**: averaging multiple fine-tuned/late
  checkpoints improves accuracy and OOD robustness without inference cost — applicable to our
  final-model construction (soup of top-k probe-selected snapshots from the anneal).
- **EMA teachers in SSL**: BYOL (**Grill et al., NeurIPS 2020,
  [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)**) and DINO (**Caron et al., ICCV 2021,
  [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)**) — in DINO the EMA teacher
  **consistently outperforms the student** throughout training and is the model you keep.
  **Pham et al., "On the Pros and Cons of Momentum Encoder in Self-Supervised Visual
  Representation Learning," 2022, [arXiv:2208.05744](https://arxiv.org/abs/2208.05744)** show the
  momentum encoder is not needed to avoid collapse per se, but yields *stabler targets and
  higher-quality representations*. For MAE-family objectives EMA is not needed for training
  stability, but the DINO lesson — evaluate/ship the EMA — transfers.
- **Hägele et al.** ([arXiv:2405.18392](https://arxiv.org/abs/2405.18392)): EMA approximates the
  cooldown result from a constant-LR run, i.e., the EMA checkpoint at time t estimates "what would
  I get if I annealed now," exactly the counterfactual our model selection wants.

---

## 5. End-of-training degradation at the LR floor

Observation (d) — final checkpoint worse on the probe than earlier ones — is supported by several
independent lines:

1. **Sharpness ceiling is lifted.** At LR η the attainable sharpness is capped near 2/η (Cohen et
   al., [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)); dwelling at a tiny floor LR lets the
   optimizer descend into much sharper minima, and sharpness anti-correlates with downstream
   quality at fixed loss (Liu et al., [arXiv:2210.14199](https://arxiv.org/abs/2210.14199)). The
   large-LR implicit regularizers (catapult flatness, large-step sparse-feature bias
   [arXiv:2210.05337](https://arxiv.org/abs/2210.05337)) are all *switched off* at the floor.
2. **Repetition-driven memorization dominates late.** Beyond the useful-repetition frontier
   (~4–16 epochs; Muennighoff, [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)), continued
   optimization on a small set mostly memorizes, and memorization disproportionately damages
   generalization-carrying internal structure while loss looks fine (Hernandez et al.,
   [arXiv:2205.10487](https://arxiv.org/abs/2205.10487); classically, the long tail is memorized
   late — Feldman & Zhang, "What Neural Networks Memorize and Why," NeurIPS 2020,
   [arXiv:2008.03703](https://arxiv.org/abs/2008.03703)).
3. **Overtraining raises sensitivity/degrades adaptability.** **Springer, Goyal, Wen, Kumar, Yue,
   Malladi, Neubig, Raghunathan, "Overtrained Language Models Are Harder to Fine-Tune," 2025,
   [arXiv:2503.19206](https://arxiv.org/abs/2503.19206)**: extended pretraining monotonically
   increases parameter sensitivity and makes *later* checkpoints worse after adaptation than
   earlier ones ("catastrophic overtraining") — the LLM-scale analog of our final-drop, found by
   probing intermediate checkpoints (see also "Amuro & Char,"
   [arXiv:2408.06663](https://arxiv.org/abs/2408.06663), where intermediate checkpoints fine-tune
   better than the final one; and Islam et al., [arXiv:2103.13517](https://arxiv.org/abs/2103.13517),
   where late pretraining adds source-specific rather than transferable structure).
4. **Caveat — loss says the opposite.** For *pretraining loss*, decaying fully to ~zero is optimal
   (**Bergsma, Dey, Gosal, Gray, Soboleva, Hestness, "Straight to Zero," 2025,
   [arXiv:2502.15938](https://arxiv.org/abs/2502.15938)**). This is not a contradiction but
   another instance of decoupling: the floor-LR phase buys loss (fine-fitting + memorization) while
   selling representation generality. With abundant unique data the trade is benign; with 110
   epochs over 20k events it is not. Also note Bergsma's own framing: AdamW acts like an EMA of
   updates whose effective window the decay controls — decay-to-zero and long EMA are near-
   substitutes, so one can take the EMA benefit *without* paying the memorization cost of a long
   floor dwell.

---

## 6. Synthesis: mapping mechanisms to observations

| Observation | Mechanism | Key citations |
|---|---|---|
| (a) probe dip at peak LR, cured by 4x data | representation churn at max plasticity (stability gap) + repetition-memorization pressure on small data; landscape train/test mismatch shrinks with data | [2205.13452](https://arxiv.org/abs/2205.13452), [2306.01904](https://arxiv.org/abs/2306.01904), [1907.04595](https://arxiv.org/abs/1907.04595), [2205.10487](https://arxiv.org/abs/2205.10487), [2210.01117](https://arxiv.org/abs/2210.01117) |
| (b) triangulation crystallizes during anneal | river-valley: decay phase fits fine/sharp-direction structure; fine patterns provably learned post-anneal; capability phase transitions | [2410.05192](https://arxiv.org/abs/2410.05192), [2405.18392](https://arxiv.org/abs/2405.18392), [1907.04595](https://arxiv.org/abs/1907.04595), [2309.07311](https://arxiv.org/abs/2309.07311) |
| (c) loss/representation decoupling | same-loss minima differ in flatness ⇒ downstream; SSL loss→downstream mediated by optimizer inductive bias | [2210.14199](https://arxiv.org/abs/2210.14199), [2202.14037](https://arxiv.org/abs/2202.14037), [2210.02885](https://arxiv.org/abs/2210.02885) |
| (d) final checkpoint worse after floor dwell | sharpness ceiling lifted at tiny LR; late-stage memorization of repeated data damaging internal structure; catastrophic overtraining | [2103.00065](https://arxiv.org/abs/2103.00065), [2210.14199](https://arxiv.org/abs/2210.14199), [2205.10487](https://arxiv.org/abs/2205.10487), [2503.19206](https://arxiv.org/abs/2503.19206) |

---

## 7. Concrete recommendations for our runs

**R1 — Peak LR philosophy: "largest stable, minus a notch," and couple it to data volume.**
Keep a large peak LR — it is the source of the flat-minima / sparse-feature bias that the probe
ultimately benefits from ([1907.04595](https://arxiv.org/abs/1907.04595),
[2003.02218](https://arxiv.org/abs/2003.02218), [2210.05337](https://arxiv.org/abs/2210.05337)) —
but target the *narrow band just above the convergence threshold*
([2410.22113](https://arxiv.org/abs/2410.22113)). Operationally: the peak LR at which the probe
dip appears but fully recovers with margin is near-optimal for the 80k-event runs; for 20k-event
runs, either drop peak LR ~2x or shorten time-at-peak — the dip there signals genuine
representation churn amplified by repetition, and prolonged churn risks basin damage rather than a
benign transient. A transient dip with full recovery is *not* by itself a reason to lower LR
(stability-gap dips precede recovery to better-than-before levels), but on heavily repeated data
the same dip coincides with the memorization mechanism of
[2205.10487](https://arxiv.org/abs/2205.10487), so on small data treat it as a warning. (Note: per
our µP finding, first make sure the *effective* hidden LR is what we think it is — an
unintentionally 4x-small effective LR changes which side of the optimal band we are on.)

**R2 — Schedule: WSD with a real anneal, then STOP; do not dwell at the floor.**
Use warmup → stable (peak) → decay; put the decay late enough that total-compute is used, but end
the run essentially when the decay ends. The triangulation metric's gains live in the decay phase
([2410.05192](https://arxiv.org/abs/2410.05192), [2405.18392](https://arxiv.org/abs/2405.18392));
the probe's losses live in the floor dwell (§5). If a floor phase is unavoidable, checkpoint
densely through the anneal and expect the best probe/triangulation checkpoint near the *end of
decay, not the end of training*. Consider a slightly higher terminal LR (e.g., decay to peak/50
rather than ~0) for the small-data runs: decay-to-zero is loss-optimal
([2502.15938](https://arxiv.org/abs/2502.15938)) but our selection metric is not loss, and the
final descent into sharp/memorizing minima is exactly what hurt the probe. Bonus from
[2405.18392](https://arxiv.org/abs/2405.18392): from a single constant-LR trunk you can branch
short cooldowns at several points and probe each — a cheap way to locate the probe-optimal anneal
point without retraining.

**R3 — Add EMA now; probe the EMA weights; keep both EMA and raw snapshots as candidates.**
Maintain a weight EMA (start with decay ≈ 0.999–0.9999 at our step counts, or LAWA-style average
of k~5–10 checkpoints spaced hundreds of steps) — near-zero cost. Expected effects, all
literature-backed: (i) it acts as a virtual anneal, so EMA probes at time t estimate "quality if I
annealed now" and give a smooth, low-variance selection signal
([2405.18392](https://arxiv.org/abs/2405.18392), [2306.03241](https://arxiv.org/abs/2306.03241));
(ii) it should largely *fill in the mid-training probe dip* (the churn is high-frequency trajectory
noise that averaging cancels — the dip on EMA weights, vs raw, is a useful diagnostic of whether
the damage is transient churn or basin-level); (iii) EMA/averaged weights transfer and probe
better than last-iterates ([2411.18704](https://arxiv.org/abs/2411.18704),
[1803.05407](https://arxiv.org/abs/1803.05407)), and in EMA-teacher SSL the EMA is systematically
the better model ([2104.14294](https://arxiv.org/abs/2104.14294),
[2208.05744](https://arxiv.org/abs/2208.05744)). The high-LR runs are exactly where averaging
gains are largest ([2306.03241](https://arxiv.org/abs/2306.03241)).

**R4 — Model selection: keep probe-on-snapshots, extend it to probe-on-EMA + small soup.**
Selecting on the probe/triangulation rather than pretraining loss is the correct call
([2210.14199](https://arxiv.org/abs/2210.14199), [2202.14037](https://arxiv.org/abs/2202.14037))
— do not switch to loss-based selection. Improvements: (i) run the probe on EMA weights as the
primary selection signal (less snapshot-to-snapshot variance ⇒ less selection overfitting to probe
noise; we already learned the fisher-r/ridge protocol is noise-sensitive); (ii) as final model,
try a greedy soup of the top few anneal-phase checkpoints, accepted only if it improves held-out
probe + triangulation ([2203.05482](https://arxiv.org/abs/2203.05482)); (iii) optionally log a
label-free representation-health metric (effective rank of token/group embeddings, RankMe-style,
[2210.02885](https://arxiv.org/abs/2210.02885)) every eval — it is nearly free, catches
rank-collapse/representation damage between probe evaluations, and gives an early-warning version
of the dip.

**R5 — Small-data runs: treat epochs as the dangerous axis, not LR alone.**
110 epochs on 20k events is ~7x past the ~16-epoch useful-repetition frontier
([2305.16264](https://arxiv.org/abs/2305.16264)) and inside the regime where repetition damages
internal structure disproportionately ([2205.10487](https://arxiv.org/abs/2205.10487)). Prefer
(in order): more unique events (validated by our own 4x result) > stronger stochastic augmentation
of the pretext (fresh mask patterns per epoch, noise re-draws — anything that makes an epoch
non-identical) > fewer effective epochs with the anneal moved earlier > weight decay increase
(grokking-style regularization pressure toward generalizing solutions,
[2201.02177](https://arxiv.org/abs/2201.02177), [2210.01117](https://arxiv.org/abs/2210.01117)).
And for small data, checkpoint-select from mid/late-anneal, never from the end of a long floor
dwell ([2503.19206](https://arxiv.org/abs/2503.19206), [2408.06663](https://arxiv.org/abs/2408.06663)).

---

## 8. Full reference list

1. Liu, Xie, Li, Ma. *Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models.* ICML 2023. [arXiv:2210.14199](https://arxiv.org/abs/2210.14199)
2. Saunshi, Ash, Goel, Misra, Zhang, Arora, Kakade, Krishnamurthy. *Understanding Contrastive Learning Requires Incorporating Inductive Biases.* ICML 2022. [arXiv:2202.14037](https://arxiv.org/abs/2202.14037)
3. Garrido, Balestriero, Najman, LeCun. *RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank.* ICML 2023. [arXiv:2210.02885](https://arxiv.org/abs/2210.02885)
4. Islam, Chen, Panda, Karlinsky, Radke, Feris. *A Broad Study on the Transferability of Visual Representations with Contrastive Learning.* ICCV 2021. [arXiv:2103.13517](https://arxiv.org/abs/2103.13517)
5. Chen, Shwartz-Ziv, Cho, Leavitt, Saphra. *Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs.* ICLR 2024. [arXiv:2309.07311](https://arxiv.org/abs/2309.07311)
6. Li, Wei, Ma. *Towards Explaining the Regularization Effect of Initial Large Learning Rate in Training Neural Networks.* NeurIPS 2019. [arXiv:1907.04595](https://arxiv.org/abs/1907.04595)
7. Lewkowycz, Bahri, Dyer, Sohl-Dickstein, Gur-Ari. *The Large Learning Rate Phase of Deep Learning: the Catapult Mechanism.* 2020. [arXiv:2003.02218](https://arxiv.org/abs/2003.02218)
8. Cohen, Kaur, Li, Kolter, Talwalkar. *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* ICLR 2021. [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)
9. Andriushchenko, Varre, Pillaud-Vivien, Flammarion. *SGD with Large Step Sizes Learns Sparse Features.* ICML 2023. [arXiv:2210.05337](https://arxiv.org/abs/2210.05337)
10. Sadrtdinov, Kodryan, Pokonechny, Lobacheva, Vetrov. *Where Do Large Learning Rates Lead Us?* NeurIPS 2024. [arXiv:2410.22113](https://arxiv.org/abs/2410.22113)
11. Chizat, Oyallon, Bach. *On Lazy Training in Differentiable Programming.* NeurIPS 2019. [arXiv:1812.07956](https://arxiv.org/abs/1812.07956)
12. Yang, Hu. *Feature Learning in Infinite-Width Neural Networks.* ICML 2021. [arXiv:2011.14522](https://arxiv.org/abs/2011.14522)
13. De Lange, van de Ven, Tuytelaars. *Continual Evaluation for Lifelong Learning: Identifying the Stability Gap.* ICLR 2023. [arXiv:2205.13452](https://arxiv.org/abs/2205.13452)
14. Harun, Kanan. *Overcoming the Stability Gap in Continual Learning.* 2023. [arXiv:2306.01904](https://arxiv.org/abs/2306.01904)
15. Hernandez, Brown, Conerly, DasSarma, Drain, El-Showk, Elhage, Hatfield-Dodds, Henighan, Hume, Johnston, Mann, Olah, Olsson, Amodei, Joseph, Kaplan, McCandlish. *Scaling Laws and Interpretability of Learning from Repeated Data.* 2022. [arXiv:2205.10487](https://arxiv.org/abs/2205.10487)
16. Muennighoff, Rush, Barak, Le Scao, Piktus, Tazi, Pyysalo, Wolf, Raffel. *Scaling Data-Constrained Language Models.* NeurIPS 2023. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)
17. Liu, Michaud, Tegmark. *Omnigrok: Grokking Beyond Algorithmic Data.* ICLR 2023. [arXiv:2210.01117](https://arxiv.org/abs/2210.01117)
18. Power, Burda, Edwards, Babuschkin, Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* 2022. [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
19. Wen, Li, Wang, Hall, Liang, Ma. *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective.* 2024. [arXiv:2410.05192](https://arxiv.org/abs/2410.05192)
20. Hägele, Bakouch, Kosson, Ben Allal, von Werra, Jaggi. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* NeurIPS 2024. [arXiv:2405.18392](https://arxiv.org/abs/2405.18392)
21. Bergsma, Dey, Gosal, Gray, Soboleva, Hestness. *Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs.* 2025. [arXiv:2502.15938](https://arxiv.org/abs/2502.15938)
22. Springer, Goyal, Wen, Kumar, Yue, Malladi, Neubig, Raghunathan. *Overtrained Language Models Are Harder to Fine-Tune.* 2025. [arXiv:2503.19206](https://arxiv.org/abs/2503.19206)
23. *Amuro & Char: Analyzing the Relationship between Pre-Training and Fine-Tuning of Large Language Models.* 2024. [arXiv:2408.06663](https://arxiv.org/abs/2408.06663)
24. Feldman, Zhang. *What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation.* NeurIPS 2020. [arXiv:2008.03703](https://arxiv.org/abs/2008.03703)
25. Izmailov, Podoprikhin, Garipov, Vetrov, Wilson. *Averaging Weights Leads to Wider Optima and Better Generalization (SWA).* UAI 2018. [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)
26. Wortsman et al. *Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time.* ICML 2022. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)
27. Sanyal, Neerkaje, Kaddour, Kumar, Sanghavi. *Early Weight Averaging Meets High Learning Rates for LLM Pre-training.* COLM 2024. [arXiv:2306.03241](https://arxiv.org/abs/2306.03241)
28. Morales-Brotons, Vogels, Hendrikx. *Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits.* TMLR 2024. [arXiv:2411.18704](https://arxiv.org/abs/2411.18704)
29. Grill et al. *Bootstrap Your Own Latent (BYOL).* NeurIPS 2020. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
30. Caron, Touvron, Misra, Jégou, Mairal, Bojanowski, Joulin. *Emerging Properties in Self-Supervised Vision Transformers (DINO).* ICCV 2021. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
31. Pham et al. *On the Pros and Cons of Momentum Encoder in Self-Supervised Visual Representation Learning.* 2022. [arXiv:2208.05744](https://arxiv.org/abs/2208.05744)
32. *Masked Autoencoders Are Effective Solution to Transformer Data-Hungry.* 2022. [arXiv:2212.05677](https://arxiv.org/abs/2212.05677)
