# Optimal Architecture — wavelet-coefficient foundation model (fusion-ready)

> **REVISED by 4-expert review (see EXPERT_REVIEW_SYNTHESIS.md, 2026-06-13).**
> Net changes: build flat/shallow trunk first (pooling hierarchy premature —
> measured regime is optimization-bound, not cost-bound); collapse to 2 tiers;
> SSL head must be DISTRIBUTIONAL/NLL (L2 destroys fluctuations) with one
> masked-prediction head (policy = masking geometry); MAKE PAIRED SIM events
> (unpaired-fusion is wishful); cross-view/fusion needs GEOMETRY-biased
> attention (wire=line integral, not point; epipolar/transport bias) not naive
> coord attention. Stages 5-10 below are the original design; treat them as
> superseded by the synthesis where they differ.

End-to-end spec for both modalities, designed so optical and TPC tokens can
enter a SHARED global fusion layer with no redesign. Grounded in the measured
facts (D-01..D-17). Provenance: stages 0-4 are measured/built; 5-10 are the
designed trunk (part two), reasoned from measurements + transformer practice.

## Design drivers (all measured)
- Tokenizer is provably lossless at trunk width (d >> max-active-per-patch:
  63-247 << 384-1024) and sub-classical denoising on both modalities.
- Attention crossover N ~ 6d (~6k tok @ d=1024): optical (~6k tok) is at/below
  -> global affordable; TPC (~25k tok) is 4x above -> hierarchy mandatory.
- FFN is linear in N and runs on every token every block; global attention is
  quadratic. => two levers: (a) GROUP attention (quadratic->linear), (b) POOL
  tokens up the hierarchy (cuts the linear FFN term). Both needed.
- Axis ordering inverts by modality: optical time>scale; TPC wire>scale>time.
- Cross-scale coupling is value-level, modest (tree op -15% TPC, +4% optical).
- Cross-plane U/V/Y correspondence = the 3D-reconstruction core (trunk's job).
- Cross-volume weak (cathode, direct same-tick), summary-only.

## The one idea that makes fusion trivial: the UNIFIED TOKEN
Every token from either modality is a pair `(z in R^d, coord)` with a shared
coordinate:
```
coord = [ x, y, z (detector frame, metres),     # PMT xyz; wire->transverse + plane angle
          t (ns, drift/photon time),            # one DAQ clock, shared
          ell = log2(physical scale of band),   # wavelet level in physical units
          view  (onehot: U,V,Y, PMT-E, PMT-W),
          modality (TPC | optical) ]
```
- `ell` + `t` tile a CONTINUUM across modalities (optical 2ns-1us, TPC 1-8us,
  meeting ~1us) -> one scale axis, not two.
- xyz in the common detector frame is what physically links a TPC charge
  deposit to the PMT light it produced -> the substrate of charge-light fusion.
- PE = sinusoidal-per-axis -> MLP(coord) (or relative bias = Fourier(coord_i -
  coord_j) for the global tier). Same module both modalities.
Consequence: the global fusion layer is PLAIN attention over the union of
token sets, keyed by coord. No special-casing. Towers are modality-specific;
the top is shared.

## The parts (end to end)

### 0. Data / forward  [BUILT]
pimm extract -> densify -> on-the-fly GPU noise (TPC coherent+intrinsic;
optical white sigma=2.6) -> digitize. Clean truth retained as target.

### 1. Wavelet + threshold  [BUILT]
coif3 DWT (TPC L4 time; optical L10) -> bands. TPC: coeff-space smart removal
(kgate=4). Per-band sigma hard threshold. A kept, D1 dropped.
=> NOISY support + values; CLEAN values at same coords (denoise target).

### 2. Normalize  [BUILT]
`asinh(c / sigma_band)` -- signed, compressive; per-(plane,band) sigma.

### 3. Patchify  [BUILT, sizes finalized here]
Per-band patches in each band's native grid. Patch size = largest s.t.
(a) max-active < d_model (lossless margin) and (b) extent <= local
correlation length (reach audits). Recommended:
- optical: hybrid -- column cells A10-D4 (anchor 1024) + per-band P=64 for D3/D2.
- TPC: per-band 2D, 16 wires x {8 @ViT-S | 16 @ViT-B/L} band-ticks; occupied only.
Token counts: optical ~5.8k, TPC ~25k (16x16) / ~33-41k (16x8). Lossless by
construction (stage-3 reshape) + stage-4 full-rank embed.

### 4. Linear embed -> unified token space  [BUILT (linear); +coord here]
`Linear([values, occupancy bits]) -> R^d` + MLP(coord) PE + view emb +
modality emb. 0.2M params; reaches the PCA floor; NO deep encoder. Output =
unified tokens. (TPC: append dead-wire bits + wire-kill augmentation.)
Optional, only if a trunk pilot shows missing local context: ONE batched
SubMConv2d (TPC) / depthwise conv (optical) BEFORE patchify -- the named hedge.

### 5. LOCAL tier  (modality-specific, windowed)  [DESIGN]
Windowed self-attention over a neighbourhood in (space x scale): a window =
a few adjacent spatial patches across ALL bands at that location (the cone),
Swin-style on the regular patch grid. Captures track-segment structure + its
cross-scale cone in one operator -> NO separate "within-level then across-level"
split needed; scale is in `coord`, attention decides.
- cost: grouped => LINEAR in N. Cheap. Spend most depth here (e.g. 6-8 blocks).
- TPC window long in WIRE (the dominant axis, r=0.685), short in time+scale.
- optical window along time + scale.
- shifted windows alternate (Swin) so windows communicate.

### 6. POOL #1  (learned merge, keep U-Net skip)  [DESIGN]
Merge 2x spatial (or 2x2) neighbouring patches per (view,band) -> halve/quarter
N. THIS is the FFN lever. Skip connection saved for dense decode.
TPC 25k -> ~6-8k; optical 5.8k -> ~3k (or skip pooling, already sub-crossover).

### 7. VIEW tier  (within plane / sensor, global-in-view)  [DESIGN]
Full self-attention over all (pooled) tokens of ONE view: full track extent
within a plane / full waveform within a sensor. ~1-3k tokens/view -> cheap.
2-4 blocks, interleaved with a light cross-view exchange (NOT deferred).

### 8. POOL #2 -> view summary tokens  [DESIGN]
Pool each view to ~K summary tokens (K~256-512). TPC: 6 views -> ~1.5-3k;
optical: 2 sides -> ~0.5-1k. Skips saved.

### 9. GLOBAL FUSION tier  (SHARED weights, both modalities)  [DESIGN — the ask]
Plain global self-attention over the UNION of summary tokens from all views
AND both modalities (~3-4k total -> below crossover, affordable). Coord-keyed
=> a TPC token and the PMT token of the same xyz/t attend directly. VGGT-style:
alternate [within-modality-global] and [cross-modality-global] blocks, 4-8 deep.
- Cross-volume (cathode) and charge-light fusion both happen HERE, by geometry.
- Shared weights across modalities = the unification payoff; the per-modality
  specificity already lives in stems + local/view tiers.
FUSION-READY WITHOUT PAIRED DATA: train each tower + its local/view tiers on
unpaired corpora; the global tier trains on whatever is available (single-
modality now; paired events when they exist) -- it is the same module either
way. Turning fusion on later = concatenate token sets + enable the cross-
modality blocks. No redesign.

### 10. Heads  [DESIGN]
- SSL: masked-token denoising at stage 4/5 (mask patches, predict clean slots);
  cross-view prediction (hold out a plane/sensor, predict from the rest) at the
  global tier; cross-MODALITY prediction (predict optical summaries from TPC
  and vice versa) = the fusion pretext.
- Dense decode (segmentation): U-Net up-path through the pool skips (6,8) back
  to per-coefficient output. Skips are why losslessness at stage 4 matters.

## Cost flow (TPC, d=1024, illustrative)
| tier | tokens | blocks | per-block cost | note |
|---|--:|--:|---|---|
| local (windowed) | 25k | 6-8 | linear (grouped) | cheap, most depth |
| view (pooled) | ~6-8k | 2-4 | ~crossover | medium |
| global+fusion | ~3-4k | 4-8 | global, but small N | affordable |
Pooling cuts the FFN (linear-in-N) term geometrically; grouping keeps the
local tier off the quadratic. Net: below the flat ViT-L MAE estimate
(208 ms/event); fusion tier adds little (small N). Fits 10-50k GPU-h budget.

## Where this differs from the "many-within -> one-across -> one-across-plane"
proposal (and why)
1. INTERLEAVE, don't defer: cross-scale and cross-view exchanges happen
   throughout (Swin shifts, light cross-view in the view tier), not as single
   late layers -- a lone fusion layer bottlenecks propagation (VGGT result).
2. POOL between tiers: grouping alone is ~2.4x; the rest of the win is reducing
   N upward (FFN is linear-in-N and otherwise runs full-width every block).
3. WINDOWED (space x scale) replaces "within-level only" -- scale rides in
   `coord`, so one windowed operator does spatial + cone; no separate level
   tier needed.
4. ADD the shared cross-modality global tier (#9) with the unified coordinate
   -- the explicit goal: optical+TPC tokens in one global fusion layer.

## Open / to measure (part two)
- cross-plane token-level MI (the trunk's analog of the M2 audit; only the
  cathode entry is measured) -> sets view/global depth.
- window size in (space x scale); within:across:pool ratio (cost arithmetic ->
  small-model sweep).
- pool factor vs dense-decode fidelity (skip sufficiency).
- whether shared global weights transfer across modalities (the unification
  test) -- now a config flip, not a redesign.
