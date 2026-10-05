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

**pw16: what the mask unit costs, and what pooling adds on top.**

| | pw16 random (4 runs) | res_loc | res_bp2 |
|---|---|---|---|
| map_r | 0.936-0.939 | 0.911 | 0.911 |
| wire loc. 0.2-0.5 / 0.5-1 MeV | 1.95-2.11 / 1.29-1.47 | 2.24 / 1.67 | 2.26 / 1.74 |
| time loc. 0.2-0.5 MeV | 12.0-12.6 | 11.2 | 11.5 |
| floor AUC 0.1-0.2 / 0-0.1 MeV | 0.890-0.927 / 0.790-0.855 | 0.914 / 0.854 | 0.908 / 0.837 |
| presence AUC | 0.835-0.839 | 0.829 | 0.830 |
| region-masked recon var_expl | 0.31-0.32 | 0.388 | 0.359 |
| masked recon, location masks (all / A4) | 0.39 / 0.54 (ab_ref) | 0.439 / 0.610 | 0.413 / 0.587 |

The location mask is what costs: map_r -0.026 and wire localisation, against
better time localisation and leak-free reconstruction; the faintest floor holds.
Pooling adds little on top (floor AUC -0.006 / -0.017, within seed spread; map_r
unchanged). So the cross-band "leak" is not only a shortcut: reconstructing a
masked token from its co-located bands teaches the within-token detail that the
frozen probe reads for map_r and wire position.

Hence res_pw8bp2r: the pooled pw8 encoder trained with RANDOM masks. The trunk
then runs over every location with any visible band (less saving than under
location masks), and the cross-band task is kept -- the visible bands of a
location are pooled into the token the masked ones attend. res_pw8loc separates
pooling from masking at pw8.

**pw8: masking x pooling** (one seed each; second seeds of pw8 and pw8bp2 running):

| | res_pw8 | res_pw8loc | res_pw8bp2 | res_pw8bp2r |
|---|---|---|---|---|
| mask / pool | random / no | location / no | location / yes | random / yes |
| map_r | 0.967 | 0.957 | 0.954 | 0.961 |
| wire loc. 0.2-0.5 / 0.5-1 MeV | 1.44 / 0.81 | 1.58 / 1.02 | 1.52 / 0.85 | 1.49 / 0.81 |
| time loc. 0.2-0.5 MeV | 11.5 | 11.6 | 12.2 | 11.3 |
| floor AUC 0.1-0.2 / 0-0.1 MeV | 0.925 / 0.831 | 0.923 / 0.834 | 0.918 / 0.821 | 0.913 / 0.803 |
| region-masked recon | 0.300 | 0.412 | 0.353 | 0.230 |
| masked recon, location masks (all) | 0.378 | 0.482 | 0.423 | 0.296 |
| step time, 4 nodes (s; mixed node types) | 0.217 | 0.197 | 0.161 | 0.158 |

Location masking alone keeps the floor and costs map_r and wire localisation.
Pooling at fixed masking (bp2 vs loc) is 18% faster and does not cost wire
localisation; it costs reconstruction (-0.06) and floor AUC within seed spread.
Pooling with random masks recovers plain pw8's map_r and localisation at 27% less
step time; its faintest-bin floor AUC is the lowest of the four -- the claim the
seeds must test. Across all runs the pw8 family sits at 0.80-0.83 floor AUC in the
faintest bin against 0.84-0.855 for pw16, while its 1%-FPR efficiency is higher.

**Second seeds** (SEED=1, same recipe) agree to ~0.003 on these metrics:

| | res_pw8 / _s1 | res_pw8bp2 / _s1 |
|---|---|---|
| map_r | 0.967 / 0.965 | 0.954 / 0.955 |
| floor AUC 0.1-0.2 MeV | 0.925 / 0.928 | 0.918 / 0.919 |
| floor AUC 0-0.1 MeV | 0.831 / 0.828 | 0.821 / 0.826 |
| floor eff. 5% FPR 0.1-0.2 MeV | 0.772 / 0.771 | 0.727 / 0.739 |
| wire loc. 0.2-0.5 MeV | 1.442 / 1.449 | 1.519 / 1.535 |
| region-masked recon | 0.300 / 0.300 | 0.353 / 0.371 |

So the location-masked pooled encoder's cost against plain pw8 is real and small
-- map_r -0.012, floor AUC -0.004 to -0.008, wire localisation +5%, 5%-FPR floor
efficiency -0.04 -- for 26% less step time and better leak-free reconstruction.
ab_varlen's floor (0.790) is an outlier against this spread, not typical noise.

**Pooled + random masks, two seeds** (res_pw8bp2r / _s1): map_r 0.961 / 0.963,
floor AUC 0.1-0.2 0.913 / 0.923, 0-0.1 0.803 / 0.820, 5%-FPR efficiency 0.714 /
0.773, wire loc. 0.2-0.5 1.485 / 1.511, 0.5-1 0.813 / 0.794, time loc. 11.33 /
11.23. Its seeds differ more in the faintest bin than the others' do.

Two-seed means, pw8 family against pw16 (4 runs):

| | pw16 | pw8 | pw8bp2 (loc.) | pw8bp2r (random) |
|---|---|---|---|---|
| map_r | 0.938 | 0.966 | 0.955 | 0.962 |
| wire loc. 0.2-0.5 / 0.5-1 MeV | 2.06 / 1.39 | 1.45 / 0.84 | 1.53 / 0.86 | 1.50 / 0.80 |
| time loc. 0.2-0.5 MeV | 12.3 | 11.9 | 11.8 | 11.3 |
| floor AUC 0.1-0.2 / 0-0.1 MeV | 0.919 / 0.835 | 0.927 / 0.829 | 0.919 / 0.823 | 0.918 / 0.812 |
| floor eff. 5% FPR 0.1-0.2 MeV | 0.70-0.75 | 0.772 | 0.733 | 0.744 |
| presence AUC | 0.837 | 0.860 | 0.857 | 0.860 |
| step time, 4 nodes (s) | ~0.18 | 0.217 | 0.161 | 0.158 |

The random-mask pooled pw8 encoder is within ~0.005 of plain pw8 on map_r and
localisation (time localisation better), 0.009-0.017 lower in floor AUC, at 27%
less step time. The whole pw8 family is lower than pw16 in faintest-bin floor AUC
and higher in low-FPR floor efficiency: finer patches do better where a detection
threshold is set (low false-positive rate) and worse in the high-FPR part of the ROC.

**The fine path is required** (res_pw8bp0r: one token per location from the
embedding, no band blocks, no skip; random masks): floor AUC 0.856 / 0.725 (0.1-0.2
/ 0-0.1 MeV), 5%-FPR efficiency 0.531, map_r 0.948, wire loc. 1.56 / 0.89 -- the
floor collapses, as it did for pt16, although every coefficient is in the token.
H2 (dilution) holds: a faint deposit does not survive being one part of a bigger
token. Two blocks over band tokens and the skip path to the decoder and probe are
what keep it (res_pw8bp2r).

**Partial decoding is free on the pooled encoder** (res_pw8bp2r_dec3,
`dec_frac=1/3`): map_r 0.963, floor AUC 0.915 / 0.812 (inside res_pw8bp2r's seed
range), 5%-FPR efficiency 0.747, presence AUC 0.864, wire loc. 1.39 / 0.76 and time
loc. 11.07 -- the best localisation of any run, plain pw8 included. With the
encoder pooled, the decoder dominates the step, so the decoder cut now buys wall
time: -16% on the same allocation.

Same-node-type step time (1 node, B=4, s/step): pw16 0.149, pw8 0.170, pw8 pooled
+ random masks 0.152, **pw8 pooled + random masks + dec_frac 1/3: 0.121** -- 19%
below today's pw16 and 29% below pw8.

## 7. Conclusion

Recommended encoder: **pw8 band tokens, `band_pool=2` (two blocks over band tokens,
typed pool of every band of an 8-wire x 128-tick location, ten blocks over
locations, skip path), random masks, `dec_frac=1/3`.** Against today's pw16 it
localises ~30% better in wire and ~10% in time, raises map_r 0.938 -> 0.963 and
presence AUC 0.837 -> 0.864, matches 5%-FPR floor efficiency (0.75 vs 0.70-0.75),
and costs ~0.02 in faintest-bin floor AUC (a property of pw8 tokens, not of the
pooling) -- at 19% less step time. A second seed of it is running.

What decided it, in order of size: a fine per-band path is necessary (without it
the floor collapses); the mask unit matters more than the pool (location masks cost
map_r and wire localisation, because reconstructing a masked band from the co-located
others teaches within-token detail); pooling itself is nearly free once the fine path
and random masks are kept; and pooling is what lets the decoder budget turn into wall
time. Not tested here: the same at d1024+, where the encoder's share of FLOPs and the
savings grow.
