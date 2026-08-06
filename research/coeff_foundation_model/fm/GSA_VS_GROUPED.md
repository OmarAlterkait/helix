# Random-Sparse Attention vs. our Grouped-Serial scheme

> **Note (research thinning).** The scripts named below (prof_gsa.py, prof_scale.py, slurm/gsa.sh) were removed from this
> branch; the measurements they produced stand. Recover any of them with
> `git show main:research/coeff_foundation_model/fm/<name>` — `main` keeps the
> full research tree.


Profiling head-to-head of **"Google's random sparse attention"** against our PTv3/FlatFormer-style
grouped-serial scheme, on our LArTPC wavelet-coeff token setup, at scale (N = 16k → 256k).

Repo artifacts: `prof_gsa.py` (harness), `slurm/gsa.sh` (job), this file.
Run: job `31925136` on A100-40GB (39.4 GB usable), torch 2.5.0+cu124, bf16 autocast, fwd+bwd,
d=512 / 12 blocks / 8 heads (identical to `prof_scale.py`).

---

## (a) What the method actually is — and the disambiguation

**There is no separately-named, genuinely-new "Google random sparse attention" from 2025-2026
that surfaced in web search.** Searches for a recent Google random/scalable-sparse method return
*other* labs' work — DeepSeek Sparse Attention (DSA), MiniMax Sparse Attention (MSA), Native Sparse
Attention (NSA, DeepSeek), NOSA, etc. — none of which are Google's, and none of which use a *random*
subset as their mechanism (they use learned/indexed/block-top-k selection). Google's own recent
efficiency work (e.g. *Sequential Attention* for feature selection, *Performers*/FAVOR+) is not
random-subset attention either.

The canonical **Google** method whose *defining novel ingredient is random attention* — each query
attends to a **random** subset of keys — is **BigBird** (Zaheer et al., *"Big Bird: Transformers for
Longer Sequences"*, NeurIPS 2020, Google Research). BigBird combines three attention types:

1. **Random attention** — each query attends to `r` **randomly chosen** keys, `A(i,·)=1` for `r`
   random keys. Justified as a **graph-sparsification / Erdős–Rényi random graph** that behaves as an
   **expander** (approximates the complete graph's spectral properties), giving the theoretical
   backbone for approximating full attention.
2. **Sliding-window** — `w/2` local neighbors each side.
3. **Global tokens** — `g` tokens that attend to / are attended by everything.

Combined cost is **O(N)** (linear), enabling ~8× longer sequences on the same hardware. On GPUs the
random component is realized at **block** granularity (random *blocks* of keys, not individual keys)
precisely because token-level random gather is a memory wall (confirmed below).

Because no specific *new* Google method exists by this name, **this profile targets the GENERIC
random-subset mechanism** (each query attends `k` randomly-chosen keys, O(N·k)), realized in BigBird's
hardware-efficient **block-random** form. It is labeled as the generic mechanism throughout — no
specific unreleased paper's numbers are claimed.

Sources:
- BigBird paper: https://arxiv.org/abs/2007.14062
- BigBird random-attention / Erdős–Rényi expander summary: https://sh-tsang.medium.com/brief-review-big-bird-transformers-for-longer-sequences-12ccd3430e3b
- 2025-2026 sparse-attention landscape (DSA/MSA/NSA/NOSA — none are Google random-subset):
  https://arxiv.org/abs/2508.18224 , https://huggingface.co/papers/2606.13392 ,
  https://www.emergentmind.com/topics/sparse-attention-algorithms
- Google efficiency work (not random-subset): https://research.google/blog/sequential-attention-making-ai-models-leaner-and-faster-without-sacrificing-accuracy/ ,
  https://research.google/blog/rethinking-attention-with-performers/

### How we made it apples-to-apples

- Same `SBlock` recipe for the grouped path; `RBlock` is byte-identical except the attention pattern
  (pre-LN → qkv → axial RoPE → attention → proj → MLP).
- Same synthetic tokens at our measured **~8.7 tokens/tick** density; same `timed()` helper, bf16
  autocast, fwd+bwd, 5 iters.
- **Budget matched**: our serial cycle spends 2/4 layers at `g_plane=1024` and 2/4 at `g_drift=2048`,
  mean group size **1536**. Random uses `k ≈ 1536` (block-random: `BLK=128`, `nrand=11` random blocks
  + own block = 12·128 = **1536** keys/query). Per-token attention FLOPs therefore match.
- **Realization** (`random_attn`): block-random gather + batched flash-SDPA, wrapped in **gradient
  checkpointing** so backward recomputes each tile's gather → peak memory bounded by one tile, not the
  whole loop. This is the honest efficient form; see the memory caveat in (c).

---

## (b) Profiling results (A100-40GB, fwd+bwd, bf16)

```
      N |   FULL ms    GB |  SERIAL ms    GB |  RANDOM ms    GB | full/ser  full/rnd  ser/rnd
  16000 |   195.3  4.87 |     99.9  4.66 |    333.8  5.23 |    1.95x    0.58x    0.30x
  32000 |   619.6  9.32 |    184.6  8.92 |    782.5  9.74 |    3.36x    0.79x    0.24x
  64000 |  2243.8 17.85 |    352.1 17.12 |   2023.8 18.76 |    6.37x    1.11x    0.17x
 128000 |  8410.8 35.01 |    645.1 33.47 |   6067.4 36.82 |   13.04x    1.39x    0.11x
 256000 |     OOM        |     OOM        |     OOM        |     —        —        —
```

- `full/ser`, `full/rnd` = speedup of SERIAL / RANDOM vs FULL (higher = faster than full).
- `ser/rnd` = RANDOM time ÷ SERIAL time inverse → **RANDOM ÷ SERIAL speed**; 0.30 means RANDOM runs at
  30% of SERIAL's speed (i.e. SERIAL is 3.3× faster). At 128k SERIAL is **9.4× faster** than RANDOM.
- **256k OOMs for ALL THREE** on 40 GB (the N·d activations + checkpoint/params exceed 39.4 GB even for
  the cheapest scheme). Caught gracefully. A larger GPU (80 GB) or activation offload is needed there;
  the *shape* of the curves below already decides the comparison.

### Time scaling (ms, fwd+bwd)
- FULL is O(N²): 195 → 8411 ms over 16k→128k (≈43×, ≈ the 8× N² factor).
- SERIAL is ~linear: 100 → 645 ms (≈6.5× for 8× N) — sub-quadratic, the win grows with N (6.4×→13×
  vs full).
- RANDOM is ~linear in FLOPs but with a large constant: 334 → 6067 ms. It only overtakes FULL at
  **N ≥ 64k** (1.11×) and reaches just **1.39×** vs full at 128k — where SERIAL is already **13×**.

### Memory (peak GB)
- SERIAL is **lowest at every N** (sorted contiguous groups, flash within group, no gather buffer).
- RANDOM is **highest at every N** (5.23 vs 4.66; 36.82 vs 33.47) despite checkpointing — the gathered
  K/V tile and the recompute save cost more than a contiguous group read.

---

## (c) Head-to-head — does random-sparse scale better? Qualitative tradeoffs for OUR tokens

**Time: SERIAL wins decisively.** Both are ~linear in FLOPs, but RANDOM carries a 3–9× constant-factor
penalty from **scattered memory access**. Our grouped scheme sorts tokens by a space-filling `order`,
so a group's `g` queries share **one contiguous g-key block** read once (coalesced, flash-friendly).
Random-subset attention gathers `k` **scattered** keys per query with **no cross-query reuse** — the
gather buffer has no locality, and on GPUs that is the dominant cost. This is not an implementation
artifact we can tune away: it is intrinsic to a structure-blind random pattern. (We also confirmed the
**naive token-level random gather OOMs even at 16k under backward** — a 4096-query tile alone is ~12 GB;
BigBird's block-random form + checkpointing is what makes it run at all, and it is *still* slower.)

**Memory: SERIAL wins.** Even with gradient checkpointing, RANDOM's per-tile gather + recompute keep
peak memory above SERIAL's contiguous-group flash at every N.

**Structure — the property we actually care about.** Our tokens are LArTPC wavelet-coeff patches over
6 wire-planes, with **drift-time the one shared cross-plane coordinate**. The FM's job is **cross-plane
triangulation** (memory: *"FM encodes cross-plane 3D position"*, B1y/z probes): a point in 3D projects
to a **drift-time-aligned epipolar locus** across planes. Our grouped scheme is built around exactly
this — the `O_t` (time-only) drift orders put tokens that are close in drift-time (hence
cross-plane-corresponding) into the **same equal-count group**, so a plane-U token attends the plane-V
and plane-Y tokens at its own drift-time *within one group*. The 3-order cycle (plane-time, drift-time,
plane-wire-time) then mixes information across groups over depth. **This is epipolar locality made
cheap.**

Random attention is **structure-blind**: a query's `k` keys are drawn uniformly over all N tokens,
ignoring drift-time and plane entirely. For triangulation this is doubly bad:
- The **cross-plane, same-drift-time** partners a token needs occupy a vanishingly small fraction of N
  (density ~8.7/tick; the epipolar-relevant set per query is O(planes) tokens, ~tens, out of N).
  Uniform random sampling of `k≈1536` keys hits those specific partners only with probability
  ~`k/N` per candidate — at N=128k that is ~1.2%, so the exact cross-plane correspondence a query
  needs is **almost never in its random set** in any single layer.
- BigBird's *theory* says random+window+global recovers full-attention expressivity **in the limit of
  depth** (expander mixing). But that guarantee is about eventual global mixing, **not** about placing
  the *specific* epipolar partner in-context cheaply. For our task the useful attention is sparse *and
  structured*; a structured sort captures it in O(N·g), random needs many layers of diffusion to route
  the same signal — more depth, more compute, weaker inductive bias.

So random-sparse is not just slower and heavier in this profile; it **discards the one piece of
structure (drift-time alignment) that our grouped scheme converts directly into cross-plane locality.**

---

## (d) Recommendation

**Do not adopt random-sparse attention as a replacement, and there is no compelling case for it as a
complement either.**

- **As a replacement:** no. It is 3–9× slower and uses more memory than our grouped-serial scheme at
  every N we tested, and it is structure-blind to drift-time — the exact property our cross-plane
  triangulation depends on. BigBird's linear-complexity, expander-mixing argument is real but buys us
  nothing our sorted equal-count groups don't already buy, at far higher constant cost.
- **As a complement:** the *idea* BigBird adds on top of local attention is a few **global tokens** +
  a little **long-range random mixing** to guarantee the graph stays connected. In our scheme, the
  3-order cycle (plane-time ↔ drift-time ↔ plane-wire-time) **already** provides the cross-group,
  long-range mixing that random edges are meant to provide — and it does so along physically meaningful
  axes. If we ever find the cyclic orders leave some pair of planes under-connected at shallow depth,
  the cheaper fix is the VGGT-style **dense-global tier** already profiled in `prof_scale.py` (a few
  full-attention layers), or a handful of **learned global tokens** — both structure-aware — not
  uniform random edges.
- **The one honest caveat:** at very large N where even our serial scheme's O(N·g) group *size* would
  have to grow, random attention's O(N·k) with fixed small `k` is asymptotically appealing. But our
  design already fixes `g` and grows *number* of groups, so we stay linear too — and the profile shows
  we stay faster and lighter through 128k. Revisit only if a future regime forces `g` to scale with N.

**Bottom line:** grouped-serial dominates random-sparse on time, memory, and inductive-fit for
cross-plane triangulation. Keep grouped-serial; reach for structure-aware global tiers, not random
edges, if more long-range mixing is ever needed.
