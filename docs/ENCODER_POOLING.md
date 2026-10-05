# Pooling inside the encoder

Where an encoder hierarchy can save tokens in this data, what it must not break,
and the arms that decide it. Companion to `docs/SCIENCE.md` §10 (patch size and
resolution) and the literature reviews behind it (`$HELIX_SCRATCH/helix_work/litreview/`).

## 1. What a pool can buy here: measured, not assumed

Token counts after each candidate pooling, from pw8 x pt8 per-band tokens (15
events of run 0027575715, `pool_ratios.py`):

| pooling | tokens left |
|---|---|
| none (pw8) | 1.00 (40.0k / event) |
| wire x2 (pw16) | 0.78 |
| time x2 (pt16) | 0.80 |
| wire x2 and time x2 | 0.59 |
| wire x4 (pw32) | 0.59 |
| **every band of a (plane, 8 wires, 128 ticks) location** | **0.46** |
| every band of a (plane, 16 wires, 128 ticks) location | 0.29 (0.37x today's pw16) |

Two facts follow.

**A spatial 2x pool removes ~20%, not 50%.** Charge is sparse and track-like:
only 28% of pw16 units hold both pw8 halves. Hierarchies earn 4x per stage on
dense images; here a wire or time stage earns ~1.3x.

**Under random token masking, pooling the visible set does almost nothing.** With
25% of tokens visible, pooling the visible pw8 tokens to pw16 removes 6% of them
(1.06x); the neighbours that would merge are mostly masked. Pooling only shrinks
an MAE encoder when masking is done in whole pooling units (Hiera's mask units),
and then it removes the full unit ratio (1.28x for wire x2, 2.7x for bands at pw16).

## 2. Why the band axis

The four bands (A4, D4, D3, D2) are the same waveform at four scales, laid at the
same place. A location of 16 wires x 128 ticks holds one A4 and one D4 pw16 token
and up to two D3 and four D2 tokens. Pooling them:

* is the largest reduction available (0.37x at pw16; 0.46x of pw8 at pw8 -- pw8
  wire resolution at 58% of today's trunk tokens);
* coarsens nothing spatially: A4's time extent and position are unchanged (an A4
  pw16 token already spans 128 ticks), and every coefficient stays in the pooled
  token through a typed merge;
* leaves alone the axis that measurably carries the floor: time (pt16 cut floor
  efficiency 0.39-0.48 -> 0.13) is the coordinate shared across planes, which is
  what cross-plane confirmation of a faint deposit uses.

It also exposes a property of today's task: under random masking a masked A4 token
has a visible co-located D4/D3/D2 sibling ~87% of the time (1 - 0.75^7), so part of
the reconstruction is cross-scale interpolation inside one location.

## 3. The open question the arms must answer

pt16 embeds every coefficient (the patch embedding is lossless) and still lost the
floor, so "nothing is discarded" does not mean "nothing is lost". Three mechanisms
fit pt16, and they predict different things for band pooling:

| | mechanism | predicts for band pooling |
|---|---|---|
| H1 | cross-plane matching runs on drift time; pt16 coarsened it | harmless: A4 timing unchanged |
| H2 | dilution: a faint deposit is a smaller part of a bigger token | harmful: the A4 token now carries D2/D3 noise |
| H3 | mask granularity changes the task | location masking alone moves the floor |

## 4. Arms (d768, kernel_ab recipe, 29,200 steps, equal steps)

| arm | what | isolates |
|---|---|---|
| res_loc | pw16 band tokens, `mask_mode=location` | H3: the mask unit alone, and the cross-band leak |
| res_bp0 | `band_pool=0, pool_skip=False`: locations from the embedding on, decoder keys are locations | H2: one token per location, no fine path (a band-merged tokenizer) |
| res_bp2 | `band_pool=2`, skip on: 2 blocks over band tokens, 10 over locations, per-token features = stage 1 + up(trunk) | whether a fine path recovers what bp0 loses |
| res_pw8bp2 | the same at pw8, `mask_cell=(8,128)` | the candidate: pw8 localisation at below-pw16 trunk cost |
| res_pw8loc | pw8 band tokens, the same location masking, no pooling | pooling at pw8 against its own masking task |

References: ab_varlen (pw16, varlen, seed 0), res_pw8. All arms are varlen +
fused_qk, which the pooled encoder requires. Scored by `eval_resolution.py`:
floor efficiency and AUC at 0.1-0.2 MeV, wire localisation, map_r, recon var_expl;
step time from the same runs. Seed spread in floor efficiency is ~0.08 (§10), so an
arm within that of its reference is not distinguished without a second seed.

Decision rules:

* res_loc within noise of ab_varlen -> location masking is safe and the pooled
  arms are interpretable. If it LOSES the floor, the cross-band leak was helping,
  and band pooling must be judged against res_loc, not ab_varlen.
* res_bp0 holds the floor -> H2 is wrong and the simplest design (one token per
  location) is available at 0.37x tokens.
* res_bp0 loses it, res_bp2 holds it -> the fine path is necessary: per-band
  features must reach the decoder and the probe.
* res_pw8bp2 localises like res_pw8 and holds the floor at <= pw16 step time ->
  the production candidate; combine with the cheapest decoder from the decoder arms.

## 5. Implementation (`helix/model/serial.py`, varlen path)

`band_pool=k` runs `enc[:k]` over band tokens, groups rows into locations
(`_locations`, the same unit as `mask_mode="location"`, extent `mask_cell`), lays
each location's tokens into `sum(pool_sub) = 8` typed sub-slots (A4 | D4 | D3 x2 |
D2 x4 by band and sub-window), concatenates and projects them (`pool_norm`,
`pool_proj`), runs `enc[k:]` over locations at the location centre, and returns
per-token features `x_k + pool_up(trunk)[location]` (`pool_up` zero-initialised, so
training starts from the stage-1 features). `pool_skip=False` returns the trunk
alone and gives the decoder the locations as keys. The decoder's queries are always
band tokens, so the objective is unchanged. Tests: `tests/test_band_pool.py`.

## 6. Results so far (2026-10-05; d768, 29,200 steps, `eval_resolution.py` v6)

**Variance first.** The same weights scored through the padded and the varlen
code paths agree (floor AUC 0.924 / 0.929 for ab_ref, 0.890 / 0.890 for
ab_varlen), so inference is not a source. Training is: across six pw16-like runs
floor AUC at 0-0.1 MeV is 0.840-0.855 with one draw at 0.790 (ab_varlen; its
sibling ab_pkg on the same code path scores 0.855). Floor efficiency at 1% FPR
moves by up to 0.09 on FIXED weights through probe numerics alone. Floor AUC is
the stable floor metric, and a single run can still be off by ~0.03 in it: floor
claims need two seeds.

**The cross-band leak, measured** (`eval_mask_modes.py`, 40 events, explained
variance of masked reconstruction):

| model | random masks | location masks | A4: random / location |
|---|---|---|---|
| ab_ref (pw16, trained random) | 0.739 | 0.393 | 0.844 / 0.544 |
| res_pw8 (trained random) | 0.799 | 0.378 | 0.883 / 0.512 |
| res_pw8bp2 (trained location, pooled) | 0.418 | **0.423** | 0.549 / **0.591** |

About half of what a random-mask model reconstructs comes from the co-located
tokens of the other bands. On the leak-free task the pooled, location-trained pw8
model beats the random-trained one.

**First pooled arm** (one seed; res_pw8loc and a second seed pending):

| | pw16 (4 runs) | res_pw8 | res_pw8bp2 |
|---|---|---|---|
| wire loc. 0.2-0.5 MeV | 1.95-2.11 | 1.44 | 1.52 |
| wire loc. 0.5-1 MeV | 1.29-1.47 | 0.81 | 0.85 |
| map_r | 0.936-0.939 | 0.967 | 0.954 |
| floor AUC 0.1-0.2 MeV | 0.890-0.927 | 0.925 | 0.918 |
| floor AUC 0-0.1 MeV | 0.790-0.855 | 0.831 | 0.821 |
| region-masked recon var_expl | 0.31-0.32 | 0.300 | 0.353 |
| step time, 1 node B=4 (s) | 0.141 | 0.168 | 0.145 |

~90% of pw8's localisation gain at pw16's step time; floor within the spread.

**Decoder arms** (same recipe, pw16): uniform `dec_frac=1/3` holds the floor
(AUC 0.927) at map_r 0.933 and recon 0.288; `d_dec=384` costs map_r 0.938 ->
0.922, floor AUC -0.01, recon 0.277, for ~3% step time at d768 -- not worth it at
this width. Step-time table for the designs (1 node, B=4, s/step): pw16 0.141,
location mask 0.140, bp0 0.112, bp2 0.117, bp2 + d_dec 384 0.102, pw8 0.168,
pw8bp2 0.145, pw8bp2 + d_dec 384 0.114.
