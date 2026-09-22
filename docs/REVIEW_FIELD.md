# The design choices, against the literature

Four parallel literature reviews, 2026-09-22, covering the MAE objective and
loss head; the attention pattern and positional encoding; muP, the optimizer and
stability; and the input representation and scaling regime. Each was given the
concrete helix choices and asked for a citation, a verdict (standard /
non-standard but defensible / likely wrong), an effect size and the one
experiment that would settle it.

**How to read this.** Nothing here is a measurement on helix except where it
says so. Items marked **VERIFIED** were checked against the code in this
repository and the numbers reproduced; everything else is a literature claim
with a citation, and the citations have not all been checked against the source
PDFs — §9 lists the ones known to be shaky.

---

## 1. Defects verified in this repository

These five were reproduced from the code, independently of any literature claim.

### 1.1 `uniform_attn` and `grouped_cross` disagree about padding, in the same file

`serial.py:16` pads to `npad = ceil(T/g)*g` and fills with `x[order[-1]]`
duplicated. `serial.py:26-28` picks `nb` first, then `gg = ceil(T/nb)`, so it
pads at most `nb-1` rows. At the training token count (T = 7,625 visible):

| | `uniform_attn` (as shipped) | `grouped_cross`'s formula |
|---|---:|---:|
| gp = 1024 | npad 8,192, **567 pad rows** | nb 8, gg 954, **7 pad rows** |
| gd = 2048 | npad 8,192, **567 pad rows** | nb 4, gg 1,907, **3 pad rows** |

Those pad rows are attended unmasked, so at g=1024 the final block is 457 real
tokens and 567 duplicates of one token — and that token, in a drift-ordered
layer, is systematically the maximum-drift token. Applying the formula already
present eleven lines below costs two lines, removes a kernel-visible defect, and
uses slightly *fewer* FLOPs.

**`MULTI_EVENT_BATCHING.md` rejects masking at the wrong price.** Its "36M
booleans per event" is for a full `(nb, g, g)` mask, which is what multi-EVENT
separation needs. Pad-only masking needs key validity alone — `(nb, 1, 1, g)`,
which SDPA broadcasts — about **32,768 booleans, ~1,000x cheaper** than the
figure used to reject it. (A bool mask does push SDPA off the flash kernel; the
`ceil` fix avoids that question entirely.)

Published practice: Point Transformer V3 (Wu et al., CVPR 2024, arXiv:2312.10035)
pads serialized patches by *borrowing points from neighbouring patches*;
Reformer (Kitaev et al., ICLR 2020, arXiv:2001.04451) masks explicitly.

**Also note the operating point.** `MULTI_EVENT_BATCHING.md` quotes the pad
perturbation at T=30,976 (8% of tokens >1%). Training runs the encoder at
T~7,625, and the same table's T=6,000 row reads **56.3% of tokens >1%**.

### 1.2 The probe evaluates an attention topology the weights never saw

`helix/probe/features.py:137` calls `encode_layers`, which runs over **all**
tokens; training runs the encoder over the **visible 25%**. With `g` fixed, that
is ~7,625 tokens in 4 blocks at train time and ~32,500 in 16 blocks at probe
time — **the per-layer physical reach is ~4x narrower at probe time**. The
default probe layer, 12, is a rolled drift-order layer: the cross-plane one.

**This confounds the headline science result.** Plane masking hides only 21-28%
of tokens, so a plane-masked step runs the encoder on ~24k tokens — much closer
to the probe's 32.5k. `coolplane25` spends 25% of its steps in that regime;
`coolbase` spends none. The reported +0.160 and the 0.53 -> 0.85 probe jump are
therefore **partly attributable to block-granularity matching rather than to the
masking objective**. Untested either way.

The fix is an inference-only experiment: re-run the `coolbase` probe with
`gp`/`gd` scaled by N_probe/N_train ~ 4.3, plus the mirror control on
`coolplane25`.

### 1.3 `apply_rope` deletes half the positional bandwidth when an axis is off

`layers.py:36`: `xw = rot(x[..., h2:], ang_w) if ang_w is not None else
x[..., h2:]`. With wire RoPE off, dims 32-63 are left **unrotated**, not
reallocated to time. So the measured result the production setting rests on —
*"removing wire RoPE freezes reconstruction, var_expl ~2%"* — cannot distinguish
"wire RoPE is essential" from "halving the positional dimension is fatal". It is
not a valid test of the hypothesis it is cited for, and `rope_split=False` rests
on a confounded negative. ~5 lines to fix (`rope_angles` takes `dim//2`
frequencies when one axis is active); until then every wire-RoPE conclusion is
open.

### 1.4 The decoder's block partition is fixed across all four CrossBlocks

`serial.py:145-149`: `oq` and `okv` are computed once and reused by every
decoder block. The drift-time boundaries are identical at every depth, so a
masked token whose partner sits across a boundary never reaches it — at any
layer. `docs/PERFORMANCE.md` notes this as a *performance* observation; it is
also a connectivity defect, in the half of the model that does 36% of the linear
work and all of the reconstruction.

Measured price of a hard non-overlapping partition elsewhere: LongLoRA (Chen et
al., ICLR 2024, arXiv:2309.12307) Table 1, Llama2-7B PG19 at 32k — grouped with
no shift **9.47** ppl vs **8.08** with a half-group shift, against 8.04 full.
Swin (Liu et al., ICCV 2021) no-shift -> shifted: 80.2 -> 81.3 IN top-1,
47.7 -> 50.5 COCO box AP.

### 1.5 Plane-major blocks straddle plane boundaries

`o_pt = argsort(plane*1e9 + t)` with gp=1024 over ~7,625 visible tokens
(~1,271/plane) puts a plane boundary inside **5 of 8 blocks** (verified). The
cross-plane pairs those blocks create are (latest-drift of plane p,
earliest-drift of plane p+1) — physically unrelated. The "plane-major" layers
are not within-plane.

`tools/profile/p10_eventaware.py` already implements per-segment block plans for
the multi-event case; the same machinery applied per *plane* fixes 1.1, 1.5 and
the roll wraparound in one change.

---

## 2. The objective and the loss head

**Hard one-hot cross-entropy is the one target encoding the literature is
unanimous against.** `losses_cat` (`loss.py:130`) does
`F.cross_entropy(logits, binid)` — a hard index. Every line of work that trains
a scalar target through a categorical head smooths across neighbouring bins:
HL-Gauss (Farebrother et al., ICML 2024, arXiv:2403.03950; sigma = 0.75 x bin
width, optimum independent of bin count), the histogram loss (Imani & White,
ICML 2018, arXiv:1806.04613), C51's projection operator, DreamerV3's two-hot,
PixelCNN++'s discretized logistic. Two-hot — already softer than one-hot —
underperforms HL-Gauss in every setting Farebrother et al. test. PyTorch's
`cross_entropy` accepts probability targets, so this is ~10 lines. Band bin
widths are 0.111 / 0.101 / 0.094 / 0.081 asinh, so sigma is well-defined.

Do **not** use uniform label smoothing: it spreads mass to physically distant
bins. The smoothing has to be Gaussian in target space.

**But it will not break the 66-74% ceiling.** `var_expl` is measured against a
posterior-mean reconstruction, and the categorical read-back
`sum_k p_k E[coeff/sigma | bin k]` *is* a posterior-mean estimator, which is
MMSE-optimal by construction. If the calibrated-posterior diagnosis in
`docs/SCIENCE.md` is right, `var_expl` is already at its Bayes ceiling and no
loss can raise it. Blau & Michaeli (CVPR 2018, arXiv:1711.06077) prove the
perception-distortion tradeoff holds for any distortion measure; GenCast (Price
et al., *Nature* 2024, arXiv:2312.15796) is the science-domain instance — the fix
was a generative model scored by **CRPS**, not a better deterministic loss.

Action: add CRPS and per-slot NLL to `CoeffFMEvaluator`. The categorical head can
win decisively there while `var_expl` stays flat, and that is the correct
evidence that it is working.

Counterweight worth recording: Ordinal Entropy (Zhang et al., ICLR 2023,
arXiv:2301.08915) derives that cross-entropy learns **higher-entropy features**
than MSE, because MSE does not increase marginal feature entropy. That is the
strongest published defence of the binned head — with the same caveat, that
plain N-way binning discards the ordering.

**`vis_w = 0` is inherited from a regime helix is not in.** MAE's masked-only
rule was measured where the visible target *is the input*, so supervising it adds
an identity map; it is worth ~0.5% there (SimMIM ablates 82.8 masked vs 81.7
full). helix's target is clean and its input is noisy, so visible tokens carry
the entire supervised denoising problem. DMAE (arXiv:2210.06983, ICLR 2023) runs
the identical configuration and says it outright: *"the loss is calculated on all
patches as the model can also learn purification on the unmasked positions."*
The blind-spot literature quantifies what predicting a position only from its
context costs: ~2 dB (Laine et al., arXiv:1901.10277), up to 1.35 dB for hiding
**one** pixel (Noise2Void).

**Implementation trap, VERIFIED.** Under `dec_mode="cross"`, `fm.py:292` fills
visible positions with **raw encoder features** before `dec_norm` and the heads.
So `vis_w > 0` today does not supervise a decoded prediction — it pushes the
encoder toward reconstruction specialisation, which is what He et al. say the
decoder exists to prevent, and the 3D probe reads frozen encoder features. The
experiment needs a code change first.

**The occupancy/value balance changed silently when the head changed, VERIFIED.**
`fm.py:443` returns `bce + vloss` with no weight. Under MSE, `vloss` was
O(0.1-1); a 128-way CE starts at ln 128 = 4.85. The value term's gradient share
rose roughly an order of magnitude with no compensation to the occupancy term,
which gates which slots get a value at all. Cheapest item in this document:
sweep `lambda_occ` in {1, 3, 10}.

What is right and should not be touched: restricting the value CE to
`occ & valid & masked` is a correct hurdle / zero-inflated factorisation (Kong
et al., arXiv:2010.16040) and is the right call at 93% zeros; occupancy BCE
being masked-only is right because visible occupancy is observed.

**Bin count is measured not to be the bottleneck.** From the shipped table, bin
widths give a quantization RMS of 0.024-0.032 asinh against a metric computed in
asinh space — **<=0.1% of unexplained variance**. A 10-minute oracle read-back
(feed true bins one-hot through the centroid table on held-out data) should
return var_expl >= 0.999 and closes this permanently. Note the bins are
**uniform in asinh**, not quantiles (`bins.py:200-201`, and the docstring says so
explicitly) — mu-law-style companding, the physically motivated choice.

**Decoder width is unsupported at equal-width, and is a scaling landmine.**
MAE Table 1(b) against a 1024-d encoder: decoder 512 -> 84.9/73.5, 768 ->
84.4/73.1, 1024 -> 84.3/73.1 — wider is worse on both. MAE-ST, the closest
analogue (harder, higher-dimensional data), also optimum at 512 against a 1024
encoder, with an explicit warning not to go *below* 512. AudioMAE: 512 and 768
tie. CrossMAE hard-codes 512 for ViT-B/L/H and never ablates width. Since
`CrossBlock` takes a single `d` (VERIFIED, `layers.py:108`), scaling the encoder
to 1024 drags the decoder with it — ~27% of forward FLOPs for an expected gain of
zero. Decoupling `d_dec` is ~20 lines and should land before the next width
increase.

**Two CrossMAE levers inherited without.** Inter-block fusion — decoder block k
cross-attends a *learned mixture* of encoder feature maps rather than only the
last: 1 map 82.9, 6 maps **83.5**, for `n_enc` learned scalars per block, larger
than CrossMAE's entire decoder-depth range. And decoding only a subset of masked
tokens: 25% gives 83.2 vs 75%'s 83.3, for 8.41 -> 6.32 min/epoch and
57,987 -> 36,805 MB.

**Mask ratio 0.75: leave it.** MAE fine-tuning is flat over 40-80%. But **define
it over occupied cells and say so** — at 93% structural zeros, "75% of slots" and
"75% of occupied cells" differ by an order of magnitude in difficulty, and the
sparse cluster (Point-MAE 0.6, Voxel-MAE 0.7 of non-empty voxels, PoLAr-MAE 0.6
on LArTPC data) all define it over occupied units.

---

## 3. Attention and positional encoding

**Sorting by a physics key before blocking is good practice and now has a
paper.** The published versions learn the permutation — Sparse Sinkhorn Attention
(Tay et al., ICML 2020, arXiv:2002.11296), PBS-Attn (arXiv:2510.21270), Routing
Transformer (Roy et al., TACL 2021). Drift time is a known-good routing key given
by physics; not having to learn it is a genuine advantage and worth stating
explicitly in any write-up.

**But none of those papers confines a query to exactly one group.** Sinkhorn sums
own-block and sorted-block scores, and its best variant is a mixture with vanilla
attention. Routing Transformer gives half its heads to plain local attention, and
routing-only degrades 2.958 -> 3.291 bits/dim on CIFAR-10. Reformer attends
"each chunk and one chunk back", plus multi-round hashing, because *"there is
always a small probability that similar items nevertheless fall in different
buckets."*

**The iso-FLOP upgrade not being taken.** Block-diagonal attention costs `T*g`
per layer, so `g=2048` single-block and `g=1024` own+previous-block cost
**exactly the same**, and the second has no hard boundary. HaloNet (CVPR 2021,
arXiv:2103.12731) makes the point empirically: *"a block size of 4 and a halo of 1
results in better accuracy than using a block size of 8 with 0 halo, despite a
smaller neighborhood size."*

**Cycling the sort key across layers is standard** — Sparse Transformer's
interleaved attention types, BlockBERT's per-head permutations, LongNet's
per-head offsets, PTv3's Shift Order. Two free improvements with numbers:
*randomize* the schedule rather than fixing the cycle (PTv3 Shift Order 76.56 ->
Shuffle Order **77.36**, and PTv3 also shows shuffling buys nothing with a single
pattern — helix has four, so it is in the regime where it pays); and cycle across
**heads** as well as layers (LongLoRA Table 6: shifted across heads 8.12 vs
across layers 8.30, their own summary being *"shifting between layers is
acceptable but not the best"*).

**The rolled order's wraparound block is unmasked.** `serial.py:91`
`o_ts = roll(o_t, gd//2)` is Swin's cyclic shift including the efficiency trick,
but block 0 then holds the latest-drift and earliest-drift tokens attending each
other densely. Swin's own text: *"a masking mechanism is employed to limit
self-attention computation to within each sub-window."* Low magnitude (~6.6% of
tokens in 6 of 12 layers) and arguably BigBird-style random attention, but
uncontrolled. One line to drop or mask block 0.

**No global or coarse-summary route.** BigBird's (NeurIPS 2020) universal-
approximation and Turing-completeness proofs require the sparse graph to contain
the star graph; their Proposition 1 is the sharper statement that depth alone
cannot repair a fixed sparse partition in general. NSA (DeepSeek, ACL 2025,
arXiv:2502.11089) keeps a compressed coarse-token branch alongside selection and
sliding-window. The cheapest version here: mean-pool each drift-time block into
one summary token and let every token attend the ~16 summaries — cost ~T*nb,
negligible, and it is the structurally right home for "how much charge is at
drift time tau across all six planes", which is the triangulation prior.

**Wire RoPE across planes is more defensible than it looks — but the plane
embedding is not the defence.** For the SBND-like geometry (U/V at +-60 degrees,
Y vertical, uniform 3 mm pitch), the epipolar constraint is a **sum**
(w_U + w_V proportional to w_Y) while RoPE only sees **differences**. U<->V
differences are a genuine metric transverse coordinate; U<->Y and V<->Y mix
coordinates and carry an arbitrary per-plane index origin. Uniform pitch means
the scale is consistent and a constant origin offset is a fixed rotation the
model can absorb, so this is not noise injection — verdict **defensible, not
wrong**.

The "the plane embedding disambiguates it" argument is what the literature
refutes. PETR (ECCV 2022, arXiv:2203.05625) Table 3, nuScenes val: per-view 2D PE
0.208 NDS / 1.165 mATE; **plus a learned per-view identity 0.224 / 1.165** —
closes ~8% of the gap and does **nothing** for localisation error; coordinates
lifted to a shared 3D frame 0.356 / 0.835. Cross-view Transformers (CVPR 2022)
agree, with the authors' warning that a learned per-camera embedding *"simply
bakes in all geometric information"* — shown to be memorisation by EAFormer's
cross-setup transfer (CVT 2.70 mIoU vs geometry-based 12.17). The mechanism:
FiLM and `plane_emb` are additive on token *content*; RoPE is a multiplicative
rotation by an angle depending only on `delta_wire`, and an additive per-group
bias cannot condition it. Circle-RoPE (arXiv:2505.16416) is built on exactly that
recognition; HoPE (arXiv:2505.20444) proves no frequency allocation rescues a
relative encoding on an axis whose offset does not track semantics.

The closest structural twin is **POYO** (NeurIPS 2023, arXiv:2310.16046): tokens
with one shared physical axis (time) and one incomparable identity axis (which
neuron). Their design is RoPE on time only, a learned lookup on identity. That is
`rope_split=True` — which §1.3 shows has never been fairly tested.

**If it does cost, the principled fix is a shared-frame 2D coordinate.** Encode
each token at `(w_p * pitch * cos theta_p, w_p * pitch * sin theta_p)` in
detector millimetres plus a per-TPC offset, and apply 2D RoPE on that pair inside
the 32 wire dims. Within a plane this is a reparametrisation, so it cannot
reproduce the var_expl ~2% collapse; across planes the difference becomes a true
metric offset. Use **RoPE-Mixed**, not axial, for that pair — Heo et al. (ECCV
2024, arXiv:2403.13298): axial frequencies *"cannot handle diagonal directions"*,
and here the epipolar direction **is** the diagonal.

**Axial RoPE with a contiguous half-split is standard** (Su's original 2D
construction; FiT; RoPE-ViT; RoMAE, which is the closest precedent for continuous
physical positions in a scientific setting). RoPE-ViT measures axial vs mixed at
the training resolution as 80.9 vs 80.9 (ViT-S) and 83.6 vs 83.8 (ViT-B) — the
gap is purely extrapolation, and detector geometry is fixed. Giving each axis its
own **full** band is the side of the M-RoPE/VideoRoPE/MRoPE-I argument that won.

**The explicit wavelength band beats base=10000, with one exposure.** RoFormer
gives no justification for 10000 — it is inherited from Vaswani et al. verbatim.
"Scaling Laws of RoPE-based Extrapolation" (ICLR 2024, arXiv:2310.05209) states
outright that *"base 10000 is the worst base value"*; FoPE shows components below
one cycle over the training span become noise; Wu et al. (arXiv:2607.07678) prove
a frequency-matching principle with optimal theta ~ 1/W for a dependency width W,
which is exactly what a per-axis physical band instantiates. **The exposure is
lambda_t_max = 4,336 = exactly the drift span, with no margin** — Men et al.
(NeurIPS 2024, arXiv:2405.14591) give a lower bound on base for retrieval, which
translates to lambda_max ~37x the context, and below it perplexity stays fine
while retrieval dies ("superficial long-context capability"). That is the one
parameter worth changing. lambda_min per axis is, as far as the search found,
**entirely untested in the literature** — no paper varies it at all.

---

## 4. muP, the optimizer and the schedule

**The muP implementation is, rule for rule, the reference one.** Hidden init
variance /m, hidden LR /m, readout forward x1/m, input group unscaled. The
`weight_decay * m` compensation on the hidden group — which looks wrong — is
literally `microsoft/mup`'s `MuAdam` default for PyTorch's lr-coupled AdamW, and
it is what the u-muP and Kosson et al. (arXiv:2510.19093) line calls independent
weight decay and recommends. **Do not let a reviewer "fix" it.**

**But muP has never been given a base-width sweep.** `d_base=128` is a formal
divisor in `fm.py:112`, not a model anyone trained. The proposed patch
`base_lr -> 1.6e-3` does not undo the diagnosed 4x: restoring the pre-muP
*hidden* LR needs 4.4e-3, and that also quadruples the embedding/FiLM/readout LR,
which the pre-muP run never had. **No scalar undoes a parametrization change.**
The sanctioned procedure is a sweep at the base width and transfer (Tensor
Programs V; the EleutherAI/Cerebras practitioner's guide prescribes a random
search over four coupled HPs). Do not drop muP before running it — it is the only
affordable path to d=2048.

**The attention-scale reasoning is theoretically right and empirically
contested.** With `head_dim` fixed at 64 and only head count growing, 1/sqrt(64)
vs 1/64 is a width-independent constant, and a constant cannot break transfer
asymptotically — that argument is correct. Lingle (arXiv:2404.05728) nevertheless
ran *exactly* this configuration — head width fixed, model width swept 128 ->
8192 — and found 1/sqrt(D) "quite suboptimal" and that it **prevented LR
transfer**. Both can be true: a constant multiplier changes softmax saturation,
which interacts with the optimal LR. The right frame is an untuned attention
multiplier fixed at 8x the muP value. Fold both scales into the sweep as extra
arms at zero marginal cost.

**Two hyperparameters inherited from a 512x larger batch.** MAE's recipe is
AdamW (0.9, 0.95), wd 0.05, **batch 4096**; helix runs **batch 4** (VERIFIED:
150,239 x 3 / 112,679 = world 4). The published rule holds Adam's second-moment
half-life fixed in data, beta2* = beta2^(B*/B) — 0.9999 by samples, 0.983 by
tokens (Marek et al., arXiv:2507.07101, who rescale to 0.9999 at batch 1). The
config comment at `coeff_fm_train.py:251-256` argues beta2=0.999 is too long a
window; **at batch 4 that reasoning is backwards and torch's default is closer to
correct than the value deliberately set.** For weight decay, report
tau = 1/(eta*lambda) rather than lambda: helix's is 18,182 steps, ~1.8% of a
1.01M-step run, against Llama-2's ~6.7%. Note the coupling — raising base_lr
without touching lambda makes the decay 4x more aggressive as a side effect.

**Warmup: the number is standard, the encoding is the bug.** 4,000 absolute steps
is more than Llama-2 (2,000) or DeepSeek-LLM (2,000). But
`WARMUP = max(100, round(0.0040 * STEPS))` gives **238 steps** on the production
config (VERIFIED) and 29 on a short run, on a cold 12-block transformer at
1.1e-3. Floor it at ~2,000 absolute. **A naive reviewer will flag "0.4% warmup is
too short" and be wrong about why.**

**For d=2048, QK-norm is the one that matters.** ViT-22B (ICML 2023) traced
divergence to attention logits growing until attention weights were near-one-hot;
Wortsman et al. (ICLR 2024) show the instability **reproduces in small models at
high LR** — reachable as soon as base_lr is raised. It also reduces LR
sensitivity, which makes the transfer more robust. **z-loss and logit
soft-capping are cargo cult here**: both target large-vocabulary softmax
pathology and these heads are 128-way, with no published ablation against
vocabulary size. **RMSNorm is cosmetic** — Pre-LN and Pre-RMSNorm are *provably*
equivalent (arXiv:2305.14858), a 1-10% throughput choice; and Lingle found
*trainable* norm gains can break muP transfer, which is a reason not to churn the
norm layer while establishing transfer. **Depth-muP does not apply** — helix
scales width at fixed depth.

**The unflagged trap at d=2048.** Adam `eps` is an absolute constant while muP
shrinks hidden updates as 1/m. At m=16 in bf16, eps=1e-8 is 16x larger relative
to hidden updates than at d=128, and a coordinate check at small width cannot see
it (Everett et al., ICML 2024, arXiv:2407.05872, call eps "an overlooked aspect
of parameterization"). Log per-group `mean(sqrt(v))` for the first 2k steps at
d=512 and d=1024.

**No LR-vs-batch rule, and the global batch IS the GPU count.** Moving 4 -> 8
ranks silently changes the LR regime. Square-root scaling (Malladi et al.,
NeurIPS 2022) gives `base_lr ~ sqrt(world/4)`. Run the base-width sweep at the
same global batch as production, or transfer an LR tuned in the wrong noise
regime.

---

## 5. Tokenization and the scaling regime

**Correction first.** An earlier draft of this review used the single-run corpus
(19,999 events). The production run used the **eight-run** corpus:
150,239 train events x 32,267 mean tokens = **4.85e9 tokens**, 112,679 steps at
batch 4 (all VERIFIED). That is **81.9 unique tokens per parameter**.

**Every ruler now says 59.2M is under-parameterised for this corpus:**

| ruler | implied optimal params at 4.85e9 tokens |
|---|---:|
| Chinchilla 20:1 | **242M** — essentially exactly d=1024 |
| MAE ViT-B ratio (2.92 unique tok/param) | 1.66B |
| MAE ViT-L ratio (0.83) | 5.84B |

219M lands at 22.1 tokens/param, still above Chinchilla's 20:1. **The step count
is not anomalous either** — 112,679 is 0.23x PoLAr-MAE's 500k and 0.54x Panda's
208k.

**The caveat that is not resolvable from the literature.** 150,239 events is
**1.1%** as many independent samples as ImageNet-22K's 14.2M images, and tokens
inside one event share physics, noise realisation and detector conditions. If the
governing unit is *events* rather than tokens, the small-data warnings return.
**No measurement anywhere establishes which unit governs MIM scaling on
correlated within-sample tokens.** The diagnostic that settles it costs no
training: **Jaccard overlap of the (plane, band, wire-block, tick-block) cell
sets between random event pairs.** Above ~0.7 means the tokenizer is largely
encoding a fixed geometric template and the effective independent-token count is
far below 4.85e9.

**The batch is the binding constraint.** Per step, helix pushes **2.5x more
tokens than PoLAr-MAE through 32x fewer independent samples** (4 events vs 128).
Gradient noise scale (McCandlish et al., arXiv:1812.06162) is variance across
*independent samples*; tokens within one event contribute almost nothing to it.
Far below critical batch, the usable LR scales with batch — so batch 4 forces a
small LR, **compounding the muP hidden-LR deficit**. That is a coherent
under-convergence mechanism requiring no claim about step count.

**Which reframes the tokenizer from a throughput question to the binding one.**
32,000 tokens/event against PoLAr-MAE's ~400 on the same detector family. A
density-adaptive grouping (fixed count of real coefficients per token, as
Point-MAE and PoLAr-MAE do) gives 4-15x fewer tokens, hence batch 16-64 at the
same memory. **Cheapest version is a one-config change: enlarge the cell to
32x16 (`n_slot=512`)** — same code path, 4x fewer tokens, occupancy unchanged,
batch 16 at the same memory. That isolates "token count" from "occupancy".

**The 93%-zero INPUT vector is fine and should not be rewritten.** Value plus a
binary missingness indicator is the standard encoding (Lipton et al. 2016; GRU-D,
Che et al. 2018), it costs 0.13M params, and the dense-slot layout preserves
intra-cell geometry for free — a set encoder would have to re-inject
(delta-wire, delta-tick) explicitly. Note helix's own Perceiver experiment
already killed the latent-bottleneck alternative.

**The strongest single argument for attacking the tokenizer.** Vigl et al.
(2026, arXiv:2602.15781) fit `L = L_inf + A/N^alpha + B/D^beta` for jet tagging
and find the input representation sets the **irreducible loss L_inf** (0.74 for
3 features down to 0.32 for 21), not the scaling rate — *"the choice of input
features primarily affects the irreducible loss, rather than the scaling rate."*
That is the literature form of this repository's own "the corpus is compression;
the tokenizer is interpretation", and it means no amount of compute or parameters
buys back a floor the tokenizer sets.

**Where helix sits in its own field, and it is uncomfortable.** PoLAr-MAE (Young,
Jwa, Terao 2025, arXiv:2502.02558) is a masked autoencoder on LArTPC point
clouds — ViT-S, 60% mask, ~400 tokens/event, 500k steps on 1.03M events; fine-
tuning on 100 labelled events matches a supervised baseline trained on >100,000.
Its stated limitation is that fine-grained semantics are not captured (Michel
F1 **0.440**, Delta **0.518** against Track 0.994), attributed by its own authors
to fixed-resolution tokenization. Panda (Young & Terao 2025, arXiv:2512.01324)
took that advice: sparse hierarchical encoder, DINO/iBOT-style **self-
distillation rather than masked reconstruction**, 98.8% mean F1 vs 95.6%
supervised, matching prior SOTA with 1,000x fewer labels — and it explicitly
attributes its advantage to avoiding coarse fixed-radius tokenization. Panda V2
(arXiv:2609.00611) is ~53M parameters: **the domain SOTA is at helix's current
scale and wins on data and batch, not width.**

So the current LArTPC self-supervision SOTA is neither a masked autoencoder nor
fixed-cell tokenized. helix's bet — masked reconstruction in the
wire/coefficient domain, pre-3D-reconstruction, which neither paper touches — is
a real novelty claim and defensible, but it is swimming against the current in
its own field and should cite and position against both.

**Three independent HEP results point away from a discretised masked-
reconstruction objective** and toward conditional-generative reconstruction or
self-distillation: Leigh et al. (arXiv:2409.12589, MLST 2025) find
non-discretized conditional generative reconstruction beats the tokenized
objective once the decoder is strong enough; Panda's result above; and
PoLAr-MAE's own recommendation. That converges with this repository's measured
MSE ceiling and its own conclusion that the fix is a generative pretext.

**Two free levers.** Raise the mask ratio with model size — Wettig et al. (EACL
2023, arXiv:2202.08005) find the optimum rises with scale (51M: 15%, 124M: 20%,
354M: 40%). And **probe intermediate layers**: MIM-Refiner (ICLR 2025,
arXiv:2402.10093) finds *"strong representations within MIM models generally
reside in intermediate layers"*, later blocks degrading as they take on the
decoder's job. Running the existing 3D ridge probe on every encoder block of an
existing checkpoint costs one evaluation pass, and if block 8 beats block 12,
several recorded negative results — the dead angle probes in particular —
deserve re-running before any conclusion about what the FM does not encode.

Also worth a five-minute measurement: the eigenspectrum of the feature
covariance. Dimensional collapse (U-MAE, NeurIPS 2022, arXiv:2210.08344) is an
alternative explanation for "reconstruction plateaus while representation keeps
improving".

---

## 6. Where independent reviews converged

Four reviews ran without sight of each other. Three agreements are worth more
than any single finding:

1. **The truncated cooldown is the leading explanation for the plateau.** Under
   WSD the stable phase is *supposed* to look flat; "the last 1.2 epochs bought
   +0.0037" is a measurement taken during the phase where nothing is meant to
   happen. The cooldown that was supposed to cash it in ran at **12.7%** of the
   stable phase against a recommended 10-20% (Hägele et al., NeurIPS 2024; Dremov
   et al., arXiv:2508.01483) and was **still rising when it ended**. The
   `1 - sqrt(p)` shape is the one Dremov et al. find best — keep it, lengthen it.
   If a 30% cooldown is still improving, the plateau was never real and every
   conclusion drawn from it needs re-checking against a properly annealed
   checkpoint.
2. **The binned head is defensible but its target encoding is not.** One review
   reached HL-Gauss from the RL/regression literature, another from Ordinal
   Entropy's feature-entropy argument. Both land on: keep the classification
   head, soften the target.
3. **The measured "8x data bought representation, not reconstruction" is the
   expected signature of a loss on its irreducible floor**, not evidence about
   data. Reconstruction loss is a bad scaling signal at fixed corpus and model;
   the probe suite is the right metric and already exists.

---

## 7. What a reviewer will "correct" and be wrong about

1. **`weight_decay * m` breaks muP.** It is `mup.MuAdam`'s default behaviour.
2. **0.4% warmup is far too short.** 4,000 absolute steps is standard; the defect
   is the fractional encoding, which yields 238.
3. **muP requires a 1/head_dim attention scale.** That rule is derived for
   head_dim growing with width; here it is fixed. The counter-argument is
   empirical, not a parametrization error.
4. **Add z-loss / logit soft-capping before scaling.** Both target 50k-way
   vocabulary softmax pathology; these heads are 128-way.
5. **Switch LayerNorm to RMSNorm for stability.** Provably equivalent under
   pre-norm; a throughput choice.
6. **Use depth-muP.** helix scales width at fixed depth.
7. **The cooldown should be linear or cosine.** `1 - sqrt(p)` is the shape the
   evidence favours.
8. **beta2=0.999 is far too long a window.** True at batch 4096, backwards at
   batch 4.
9. **Drop muP.** It is correctly implemented; it has merely never been swept.
10. **Plane masking is the easy task.** It hides 21-28% of tokens and is
    *harder* than 75% random — this repository's own measurements
    (`mask.py`): random 0.75 -> var_expl 0.703; plane n=1, 0.333 -> 0.506.

---

## 8. Open questions with no literature to answer them

- Whether MIM scaling on **correlated within-sample tokens** is governed by
  tokens or by independent samples. Decided for helix by the cell-overlap
  diagnostic, not by reading.
- The fraction of high-attention query-key pairs lost purely to block boundaries,
  as a function of block size. No paper measures it; the co-block recall
  experiment would be novel.
- Training block size != inference block size at fixed retrieval budget. No
  ablation anywhere.
- An identity embedding alone, with a shared positional encoding across
  incomparable frames. PETR Table 3 and CVT Table 4 are the closest and both say
  it is insufficient, but neither isolates it.
- Varying `lambda_min` per axis in RoPE. No paper varies `lambda_min` at all.
- Fixed-geometric-cell vs density-adaptive tokenization on the same sparse
  scientific data. The compute argument is solid; the representation-quality
  argument is not.
- Whether a decoder wider than 512 ever helps a *denoising* target. Nobody has
  ablated decoder capacity against a sparse-coefficient or dense-regression
  target.
- No compute-optimal (isoFLOP) scaling law for MAE/MIM exists anywhere. Any
  "N tokens/param" figure quoted for an MAE, including those in §5, is
  extrapolation.
- No LArTPC scaling law with model size, data or compute exists; Panda V2 says so
  itself. An N x D sweep on this corpus would be the first.

---

## 9. Citation reliability

Everything attributed to **this repository** was verified against the code or the
shipped artifacts. Literature claims were not uniformly checked against source
PDFs. Flagged as needing verification before publication: MAE Table 1 numbers and
the "~0.5% all-pixels" sentence (read from ar5iv, the CVPR PDF 403s); AudioMAE's
structured-masking figures (read off a plot); ViTok rFID figures (read off
plots); BigBird Table 1 (no R+W+G row exists — the global-token delta is
inferred, not measured); LongLoRA Table 6 column alignment; Swin Table 5 FPS;
Sparse Transformer's enwik8 bpc; and arXiv:2606.02680 (single-author 2026
preprint, numbers PDF-extracted). SeerAttention figures were internally
inconsistent on extraction and are **not** cited. The physics-FM addendum in §5
(Vigl, FM4NPP, Panda V2, Ordinal Entropy, Occupancy-MAE, RS3L) was cross-checked
by a second agent but not by the one that reported it.

---

## 9b. Measured verdicts

Four of the free diagnostics in §10 have been run. Three changed a verdict above
and one found something no review asked about. Scripts: `tools/profile/i1`-`i4`.

### I1 — the corpus is not a fixed template. §5's caveat closes.

300 events spread across the run, Jaccard on the `(plane, band, wire-block,
tick-block)` cell sets:

| | |
|---|---:|
| Jaccard between unrelated event pairs | **0.238** (p5 0.224, p95 0.251) |
| cells present in ALL 100 events | **0** |
| union over 100 events | 161,707 vs 32,064 mean per event |
| per-band token count, CV | 0.055 - 0.126 |

No cell is universal and no band is an event-independent floor. The effective
independent-token count is not collapsed, so §5's "if the governing unit is
events, the small-data warnings return" resolves in favour of the token count,
and 219M is supported. Events do share ~40 % of the smaller one's cells, which
is the detector's active volume, not a template.

### I2 — the boundary cluster is retired, and §1.5 is INVERTED.

Candidate cross-plane partners (same volume, different view, coincident in
TOFF-corrected drift time), co-block fraction per layer and as a union over the
12-layer cycle:

| regime | L1 plane | L2 time | L3 plane+wire | L4 time-rolled | union |
|---|---:|---:|---:|---:|---:|
| train (visible 25 %) | 0.049 | 0.998 | 0.063 | 0.992 | **1.000** |
| probe (all tokens) | 0.000 | 0.986 | 0.010 | 0.968 | **1.000** |

Union recall is 1.000 in every regime and at every block size tested (0.999 at
`gp/gd x 0.5`), against the 0.95 threshold the review set for itself. **Cross-
plane pairs meet only in the drift-ordered layers, and essentially always
there** — the schedule does the job it was designed for. So Reformer's
own+neighbour (§3), the decoder's fixed partition (§1.4) and the
plane-straddling blocks (§1.5) buy nothing on this measure.

**§1.5's recommendation is a regression, not a cleanup.** The 5-of-8 straddling
blocks are the ONLY cross-plane contact the plane-major layers have; making them
plane-pure would take 0.049/0.063 to exactly zero.

Note §1.2's predicted train-vs-probe gap is visible here in the right direction:
the plane-major layers do 0.049/0.063 of cross-plane mixing at training and
0.000/0.010 at probe time.

### I3 — the padding defect is worse than documented; the proposed fix is not free.

Trained cooldown weights, at the real training token count (T ~ 7,630, pad 562),
each variant against its OWN pad-masked reference:

| | median relative deviation | tokens > 10 % |
|---|---:|---:|
| padding effect, shipped geometry | **0.112** | **53.4 %** |
| padding effect, ceil geometry | 0.022 | 6.0 % |
| **partition change** (ceil vs shipped, both pad-masked) | **0.988** | 99.9 % |

The attended padding moves the median token's encoder output by **11 % of the
mean feature magnitude** — far worse than `MULTI_EVENT_BATCHING.md`'s "8 % of
tokens > 1 %", which §1.1 correctly identified as quoted at the wrong operating
point.

But **the `ceil` fix changes the model about nine times more than the defect it
removes** (0.99 vs 0.11 median). It re-partitions the token set, so no existing
checkpoint is interpretable under it. §1.1's "two lines, free, slightly fewer
FLOPs" is wrong about "free": it is a new-run change exactly like the defect.

The change that IS cheap in model terms is pad-MASKING the shipped geometry:
identical partition, removes the 11 %, costs the flash kernel. That, not the
`ceil` fix, is what a next run should adopt. `helix/model/fastpath.py`
preserves the shipped contract deliberately, and this is why.

### I4 — the encoder has massive activations, and no register token to put them in.

I3's sanity check showed mean |feature| 1.47 against max 954 in the same tensor.
Followed up on the cooldown checkpoint, per encoder layer:

| layer | median \|f\| | max \|f\| | max/median | tokens > 100x median | channels |
|---:|---:|---:|---:|---:|---:|
| 1-6 | 0.18 - 0.33 | 12 - 18 | 47 - 70 | **0** | 0 |
| 7 | 0.379 | 64 | 170 | 133 | 4.8 |
| 9 | 0.528 | 82 | 155 | 134 | 4.0 |
| 10 | 0.712 | 497 | **703** | **4,695** | 8.5 |
| 12 | 1.107 | 932 | **847** | **5,421** | 7.5 |

**Layers 1-6 are clean. The pollution appears at layer 7 and explodes at layer
10.** By layer 12 it is **17.7 % of tokens** and ~8 channels.

The carriers are physically identified, and the identification is unambiguous.
The top 16 tokens at layer 12 are **all band 3** (the finest kept band) and carry
**1-2 active slots against an event mean of 7.30**, spread across all six planes
and the whole drift window:

| \|f\|max | plane | band | active slots |
|---:|---:|---:|---:|
| 926.0 | 4 | 3 | 1 |
| 914.7 | 4 | 3 | 1 |
| 906.4 | 0 | 3 | 1 |
| 898.4 | 5 | 3 | 1 |

This is the Darcet, Oquab, Mairal, Bojanowski signature exactly (ICLR 2024,
arXiv:2309.16588): a ViT with no CLS or register token repurposes its **lowest-
information patch tokens** as high-norm scratch space, and the damage lands on
dense/spatial readout. helix has no CLS, no BOS and no register token, and its
downstream is a dense 3D localisation probe on frozen features.

**Consequence for every probe number in `docs/SCIENCE.md`.** The probe reads
layer 12 by default (`scripts/run_probe.py`), where ~18 % of rows are dominated
by ~8 scratch channels. §5's MIM-Refiner citation said intermediate layers hold
the better representation; this supplies the mechanism, and it says *which*
layers: 6 is the last clean one, and the cliff is between 9 and 10. Probing
every layer of an existing checkpoint costs one evaluation pass and may
retroactively change the recorded negative results — the dead angle probes in
particular.

This was not on any review's list. It came out of a sanity check.

## 10. Ranked plan

**Free, no training.**
1. ~~Cell-set Jaccard overlap~~ — **done, §9b I1.** Corpus is not a template.
2. ~~Co-block recall~~ — **done, §9b I2.** Union 1.000; the boundary cluster is
   retired and §1.5 is inverted.
2b. **Probe every encoder layer of the cooldown checkpoint** — promoted to the
   top by §9b I4. Layers 1-6 are free of massive activations, layer 12 (what the
   probe reads) is 17.7 % polluted. One evaluation pass.
3. Oracle bin read-back (§2) — closes the bin-count question permanently.
4. Run the existing 3D probe on every encoder block of an existing checkpoint
   (§5) — may retroactively change recorded negative results.
5. Probe `coolbase` at `g` scaled by N_probe/N_train, plus the mirror control on
   `coolplane25` (§1.2) — may partly re-explain the headline plane-masking
   result.
6. Add CRPS and per-slot NLL to `CoeffFMEvaluator` (§2).

**Two lines to ~20 lines.**
7. `ceil`-block padding in `uniform_attn` (§1.1).
8. Full-width time RoPE when one axis is off (§1.3) — unblocks every wire-RoPE
   conclusion.
9. HL-Gauss target smoothing at sigma = 0.75 x bin width (§2).
10. Floor `WARMUP` at ~2,000 absolute steps (§4).
11. Decouple `d_dec` from `d_enc` before the next width increase (§2).

**Runs.**
12. Cooldown at 12.7% / 20% / 30% from the existing stable-phase checkpoint (§6).
13. muP base-width LR sweep at d in {128, 256}, with the attention-scale and
    query-zero-init arms folded in (§4).
14. beta2 in {0.95, 0.99, 0.999} and lambda in {0, 0.01, 0.05}, before locking a
    base LR (§4).
15. Larger cells (`n_slot=512`) at batch 16 (§5).
16. `lambda_occ` in {1, 3, 10} (§2) and `plane_frac` in {0.1, 0.25, 0.4} (§2).
17. d=1024 at 12 encoder blocks, decoder held at 512 — only after 12-15.
18. d=2048 only inside an isoFLOP triple, and only if the cell-overlap diagnostic
    says the tokens are not largely redundant.
