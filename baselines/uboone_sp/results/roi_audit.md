# Audit: example uboone ROI finding vs Wire-Cell OmnibusSigProc (MicroBooNE)

Draft, 2026-10-08. Read-only audit; nothing was run except fetching sources.

## Sources and labels

| short name | file |
|---|---|
| `OSP.h` | `reference/wirecell/sigproc_inc_WireCellSigProc_OmnibusSigProc.h` (WCT master @ b7bd2a1a) |
| `OSP.cxx` | `reference/wirecell/sigproc_src_OmnibusSigProc.cxx` |
| `RF.cxx` | `reference/wirecell/sigproc_src_ROI_formation.cxx` |
| `RR.cxx` | `reference/wirecell/sigproc_src_ROI_refinement.cxx` |
| `SR.cxx` | `reference/wirecell/sigproc_src_SignalROI.cxx` |
| `filt.jsonnet` | `reference/wirecell/cfg_pgrapher_experiment_uboone_sp-filters.jsonnet` |
| `sp.jsonnet` | `reference/wirecell/cfg_pgrapher_experiment_uboone_sp.jsonnet` |
| `LfFilter.cxx`, `HfFilter.cxx`, `Response.cxx` | fetched with curl to stdout from raw.githubusercontent.com, WCT master (`sigproc/src/{Lf,Hf}Filter.cxx`, `sigproc/inc/WireCellSigProc/{Lf,Hf}Filter.h`, `util/src/Response.cxx`) |
| `ex/roi.py`, `ex/filters.py`, `ex/deconv.py`, `ex/pipeline.py` | `reference/uboone_deconv/uboone_pipeline/*.py` |
| `wrap` | `sp/pipeline.py` (our wrapper) |
| paper | arXiv:1802.08709. Text from ar5iv HTML; the LaTeX source (`extraction.tex`) and the Fig. 15b/16b/17b PDFs came from the arXiv e-print tarball. I read the figure values off the plots by eye. Where noted, I quote draft text that is commented out in `extraction.tex`. That text is NOT in the published paper. |

Labels: **VERIFIED** means I read the code or paper line that is cited. **ASSUMED/UNVERIFIED** means it is inferred and I did not read it directly.

`sp.jsonnet` overrides only `anode, dft, field_response, elecresponse, postgain:1, per_chan_resp/shaping, fft_flag:0` (sp.jsonnet:72-104). The `r_fake_signal_*` overrides there are commented out (lines 102-103). So every ROI parameter below takes its `OSP.h` default. **VERIFIED**

---

## 0. Parameter table (same units: MHz for time filters, cycles/wire for wire filters, electrons/tick for charge)

Wire-Cell formulas (**VERIFIED**, `Response.cxx:435-444`):
`hf_filter(f,σ,p,flag) = (flag && f==0) ? 0 : exp(-0.5*(f/σ)^p)` and `lf_filter(f,τ) = 1 - exp(-(f/τ)^2)`.
`{Lf,Hf}Filter::filter_waveform` uses `freq = i/N * 2 * max_freq`, wrapped to negative values above `max_freq`, then `|freq|` (`LfFilter.cxx:41-50`, `HfFilter.cxx:47-56`; `use_negative_freqs` defaults to true in the headers). The time filters use `max_freq = 1 MHz` (`filt.jsonnet:13,21`), so `freq` is real frequency in MHz with Nyquist at 1 MHz, which is the same as the example's `fftfreq(nt, 0.5 µs)`. The wire filters use `max_freq = 1` (dimensionless, `filt.jsonnet:32`), so `freq_WC = 2 × (cycles/wire)`. The example equivalent is therefore `a_w = σ/2`. Time filters have `flag: true` (DC zeroed, `filt.jsonnet:24`) and wire filters have `flag: false` (`filt.jsonnet:34`).

| quantity | Wire-Cell (uboone) | example `PLANE_PARAMS` (`ex/filters.py:85-113`) | paper figure (by eye) | differ? |
|---|---|---|---|---|
| Wiener_tight U (ROI finding): σ, power → F=0.5 at | 57.5416/400 = **0.14385 MHz**, p=4.104 → 0.156 MHz (`filt.jsonnet:47-48`) | `wiener_a=0.28, b=4` → 0.304 MHz | Fig 15b: dashed curves reach 0.5 at about 0.155 MHz | **yes, about 2× wider** |
| Wiener_tight V | 59.9306/400 = **0.14983**, p=4.208 → 0.162 | 0.28, 4 → 0.304 | same | **yes** |
| Wiener_tight W | 58.8802/400 = **0.14720**, p=4.175 → 0.159 | `0.35, 4` → 0.380 | same | **yes, 2.4×** |
| Wiener_wide U/V/W (final "wiener" output only) | 0.178695 / 0.184666 / 0.183044 MHz, p=5.331 / 5.605 / 5.449 → 0.190-0.196 (`filt.jsonnet:53-58`) | none | not shown | n/a for ROI |
| Gaus_wide (charge), all planes | σ = **0.111408 MHz**, p=2 → 0.131 MHz; time-domain σ_t = 1.43 µs (`filt.jsonnet:46`) | `gauss_a` = 0.24 (U/V), 0.30 (Y) → 0.283 / 0.353 MHz; σ_t = 0.66 / 0.53 µs | Fig 15b: solid curve is 0.5 at about 0.13 and about 0.2 at 0.2 MHz | **yes, 2.2-2.7× wider** |
| Wire_ind (U,V) | σ_WC = 1.4/√π = 0.790 → **a_w = 0.395 cyc/wire**; F(0.5 cpw) = 0.449; σ ≈ 0.40 wire (`filt.jsonnet:60`) | `wire_a=0.11` → F(0.5) = 3e-5; σ ≈ 1.45 wire | Fig 16b: red curve = 0.45 at 0.5 | **yes, 3.6× narrower (heavy cross-wire smoothing)** |
| Wire_col (W) | σ_WC = 3/√π = 1.693 → **a_w = 0.846**; F(0.5) = 0.84; σ ≈ 0.19 wire (`filt.jsonnet:61`) | `wire_a=0.30` → F(0.5) = 0.25; σ ≈ 0.53 wire | Fig 16b: blue curve = 0.84 at 0.5 | **yes, 2.8× narrower** |
| ROI_tight_lf (τ) | **0.02 MHz** → F=0.5 at 0.0167 (`filt.jsonnet:41`) | `lf_tight_a=0.035` → 0.029 | Fig 17b: red curve 0.5 at about 0.017 | **yes, 1.75×** |
| ROI_tighter_lf | **0.1 MHz** → F=0.5 at 0.083 (`filt.jsonnet:42`) | none | not in paper | **missing** |
| ROI_loose_lf | **0.0025 MHz** → F=0.5 at 0.0021 (`filt.jsonnet:43`) | `lf_loose_a=0.010` → 0.0083 | Fig 17b: black curve 0.5 at about 0.002 | **yes, 4×** |
| tight th factor, induction | `m_th_factor_ind{3}` (`OSP.h:135`) | `roi_th_factor=3.5` | text: 3.5 (§3.2.3) | code ≠ paper; example follows the paper |
| tight th factor, collection | `m_th_factor_col{5}` (`OSP.h:136`) | 5.0 | 5 | same |
| tight pad | `m_pad{5}` (`OSP.h:142`) | `roi_pad` 10 (U/V), 5 (Y) | draft only: "5 time ticks" | U/V differ |
| connectivity asymmetry | `m_asy{0.1}` (`OSP.h:143`) | none | draft only: 0.1 | missing |
| loose rebin | `m_rebin{6}` (`OSP.h:144`) | none | draft only: 6 bins | missing |
| loose th factor | `m_l_factor{3.5}` (`OSP.h:145`) | uses 3.5 on unbinned data | draft only: 3.5 | different meaning |
| loose max th | `m_l_max_th{10000}` (`OSP.h:146`) | none | draft only: 10k e | missing |
| loose th factor1 | `m_l_factor1{0.7}` (`OSP.h:147`) | none | draft only: 70% | missing |
| loose short length | `m_l_short_length{3}` (`OSP.h:148`) | none | draft only: ±3 bins | missing |
| loose jump_one_bin | `m_l_jump_one_bin{0}` (off) (`OSP.h:149`) | none | - | - |
| refinement th factor | `m_r_th_factor{3.0}` (`OSP.h:152`) | (3.5 reused) | draft only: "3σ" | differ |
| fake-signal low / high | `500` / `1000` e (`OSP.h:153-154`); ind factors 1.0 / 1.0 (`OSP.h:155-156`) | none | draft only: 1200 / 1500 e (older values) | **missing** |
| r_pad | `5` (`OSP.h:157`) | (roi_pad reused: 10/5) | draft only: 5 | differ |
| r_break_roi_loop | `2` (`OSP.h:158`) | 1 pass | "iterative" | differ |
| r_th_peak / r_sep_peak | 3.0 / 6.0 (`OSP.h:172-173`) | peak > 3.5·rms; split if valley < 0.3·min(peak) | draft: 3σ peaks, "twice the 3σ" separation | differ |
| r_low_peak_sep_threshold_pre | 1200 e (`OSP.h:174`) | none | - | missing |
| r_max_npeaks / r_sigma / r_th_percent | 200 / 2.0 / 0.1 (`OSP.h:175-177`) | none (simple local max) | - | missing |
| multi-plane protection | `m_use_multi_plane_protection{false}` (`OSP.h:273`); mp_th1 = 1000, mp_th2 = 500, tick res 4 (`OSP.h:277-279`) | none | - | off in both |
| roi_abs_threshold | none (`+1` e floor only, `RF.cxx:464,468,472`) | 350 / 500 / 300 (noiseless mode only) | the paper's text quotes ~350/500/300 e as *typical* 3.5σ/5σ values | example-only |

The paper prints Eq. 3.14 as `1 - e^{(ω/a)^2}` (no minus sign). The LaTeX source line `extraction.tex:490` has the same typo. The code and Fig. 17b agree on `1 - exp(-(f/τ)^2)`, and so does the example (`ex/filters.py:63`). **VERIFIED**

The example's comments state that it read its numbers from Figs. 15-17 (`ex/filters.py:73-83`). Every one of those readings disagrees with both the figures and the code. For example, it says the Wiener filter "cross[es] 0.5 at ~0.30 MHz", but the figure shows 0.155. For Fig. 16b it probably misread the y-axis, which starts at 0.45 and not at 0. **VERIFIED** (figures read by eye; the y-axis cause is ASSUMED)

---

## 1. Which deconvolution outputs feed the ROI finder

All outputs share `decon_2D_init`: 2D FFT, division by the averaged response, the wire filter `m_Wire_filters = {Wire_ind, Wire_ind, Wire_col}` (`OSP.h:188`, applied at `OSP.cxx:1310-1333`), inverse wire FFT, then a wire shift and a time shift of `(m_coarse_time_offset + m_intrinsic_time_offset)/period` with `m_coarse_time_offset{-8 µs}` (`OSP.h:115`, `OSP.cxx:1354-1362`). After that, each output multiplies the time spectrum by its own filter, takes the inverse FFT, and calls `restore_baseline` (median subtraction, explained below). **VERIFIED**

| purpose | Wire-Cell output (filter product) | where | example |
|---|---|---|---|
| tight ROI, induction (a) | Wiener_tight × ROI_tight_lf (0.02) | `decon_2D_tightROI`, `OSP.cxx:1544-1563` | `C_tight` = C_wiener × LF(0.035) (`ex/deconv.py:200-233`) |
| tight ROI, induction (b) | Wiener_tight × ROI_tighter_lf (0.1). Thresholded with the threshold from (a); either output can open an ROI | `decon_2D_tighterROI` `OSP.cxx:1596-1615`; used `OSP.cxx:2017-2021`, `RF.cxx:488,492` | **absent** |
| tight ROI, collection | Wiener_tight_W only (no LF) | `OSP.cxx:1564-1568, 2025-2026` | C_wiener (Wiener a=0.35 + wire filter) |
| loose ROI, induction | Wiener_tight × ROI_loose_lf (0.0025). Channels within ±2 (U) / ±1 (V) of a `bad` channel, or ±2 of an `lf_noisy` channel, get Wiener_tight × **tight** LF instead | `decon_2D_looseROI` `OSP.cxx:1653-1725` | `C_loose` = C_wiener × LF(0.010); no channel masks |
| ROI refinement contents (all ROI checks after formation) | induction: Wiener_tight, **no LF** (`decon_2D_ROI_refine`, `OSP.cxx:1509-1537`, called at 2050). Collection: Wiener_tight_W (the tightROI output). Then `roi_refine.load_data(iplane, m_r_data)` at `OSP.cxx:2065` | | break_rois uses \|C_wiener\| |
| final "wiener" frame | Wiener_wide, ROI-applied | `decon_2D_hits` `OSP.cxx:1825-1865, 2159-2161` | none |
| final charge ("gauss") | Gaus_wide, ROI-applied | `decon_2D_charge` `OSP.cxx:1867-1904, 2179-2184` | C_gauss (Gaussian a=0.24/0.30) |

**DC handling.** Every Wire-Cell time filter has `flag:true`, which makes F(0)=0 (`Response.cxx:437-439`). In addition, every ROI-finding output gets `restore_baseline` (`OSP.cxx:1536,1587,1639,1737`). That function removes a per-row **median** of nonzero samples, then removes the median again using only samples with `|x - median| < 500` (`OSP.cxx:1032-1087`). On the final outputs only collection gets `restore_baseline` (`OSP.cxx:1862-1864, 1901-1903`). The example zeroes DC exactly when `not noiseless` (`ex/deconv.py:178-179`), which matches Wire-Cell in noisy mode, and it never subtracts a median. **VERIFIED**

**Pre-decon rebase.** At master, `m_rebase_planes{0,1,2}` (`OSP.h:282`) and `RB_SIGMASK` (`OSP.h:305`) mean `load_data` subtracts a straight line per channel, anchored on robust 200-tick windows at each end (`OSP.cxx:467-473, 1100-1199`). This is a master default. Whether MicroBooNE production had it is **UNVERIFIED**; I suspect it is a later addition. The example has nothing like it. Matters little for baseline-subtracted, noise-filtered input.

The paper's §3.2 bullet says ROI finding uses "the deconvolved charge distribution after the Wiener-inspired filter", combined with tight and loose LF filters (§3.2.3). It does not mention the tighter (0.1 MHz) output. **VERIFIED**

---

## 2. Tight ROIs (`ROI_formation::find_ROI_by_decon_itself`, `RF.cxx:407-549`)

| aspect | Wire-Cell | paper | example | matters with noise? |
|---|---|---|---|---|
| RMS data | per wire, the **tight-LF** output for induction and the Wiener_tight output for collection, over the full readout. Ticks in `bad` ranges are excluded (`RF.cxx:427-461`). This RMS is stored as `uplane_rms` / `vplane_rms` / `wplane_rms` (`RF.cxx:465,469,473`) and **reused for every later refinement threshold** | "deconvolved charge distribution with the tight LF filter is employed to calculate the RMS", full window (~9600 ticks), per channel (§3.2.3) | per wire, computed separately on each array that gets thresholded: tight on C_tight, loose on C_loose (`ex/roi.py:117-118`), collection on C_wiener (`:93`), break on C_wiener (`:227`) | yes, see §3 and §4 |
| RMS estimator | `cal_RMS` (`RF.cxx:364-405`), with `roi_mad_rms` false by default (`OSP.h:141`). Pass 1: `rms0 = sqrt(((p84-p50)^2+(p50-p16)^2)/2)`, using `percentile(signal, 0.16 / 0.5 / 0.84)` (`RF.cxx:385-388`). Pass 2: `rms = sqrt(mean(x^2))` over samples with **`|x| < 5·rms0`** (`RF.cxx:393-400`). The cut is on \|x\| about zero, not about the median; the median is ~0 after `restore_baseline`. Exactly one refinement pass, not iterated | "68% quantile range relative to the ... mean value. ... insensitive to the true signals" (§3.2.3) | `compute_rms`: pass 1 only, using 15.87 / 50 / 84.13 percentiles, floored at 1e-6 (`ex/roi.py:21-25`). **No 5σ-clipped second pass** | moderate. Equal for pure Gaussian noise; differs when the noise is non-Gaussian or occupancy is high. The internals of `Waveform::percentile` are UNVERIFIED |
| threshold | `threshold = th_factor·rms + 1`; th_factor = 3 (ind) / 5 (col) (`RF.cxx:464,468,472`). The +1 e floor is negligible | 3.5 (ind) / 5 (col) × RMS | `roi_th_factor·rms`: 3.5 / 5 (`ex/roi.py:79`) | ind: 3 vs 3.5 changes the noise ROI rate a lot |
| polarity | **signed**: `content > threshold \|\| content_tight > threshold` (`RF.cxx:488,492`) | "signal above noise" | **\|signal\| > threshold** (`ex/roi.py:33`) | **yes**: about twice the noise false positives, and it catches the negative undershoots that LF filters produce next to real signal |
| ROI shape | contiguous run above threshold. Runs that touch are merged (`RF.cxx:507-517`). No minimum length | - | contiguous run, no minimum length (`ex/roi.py:28-43`) | - |
| padding | `extend_ROI_self`: ±`pad` = 5 ticks, clipped, overlapping runs merged (`RF.cxx:167-250`) | "extended in time to cover the signal tails" | ±10 (U/V) / ±5 (Y) (`ex/roi.py:123, 95`) | U/V: wider ROIs pick up more noise charge |
| connectivity | `create_ROI_connect_info`, **all planes including collection**. For wires i and i+2: if `|L2-L1| < (L1+L2)·asy` with asy=0.1, add the ROI `((s1+s2)/2, (e1+e2)/2)` to wire i+1, unless it overlaps an existing ROI there (`RF.cxx:252-362`) | "additional ROIs are created by examining the connectivity of the existing ones". The draft (commented out) has the same rule, asymmetry < 0.1 | **absent**. The example instead copies tight ROIs to w±1, see §4 | yes (different) |
| re-check after formation | In `ROI_refinement::load_data` each tight ROI is rebuilt from the refinement contents with a linear baseline between its end bins (`SR.cxx:14-22`). It is **dropped unless some bin > r_th_factor·rms = 3·rms** (`RR.cxx:319-327`). Loose ROIs get the same check (`RR.cxx:494-501`) | draft: same idea, a 3σ check after linear baseline | absent | yes, it removes noise ROIs whose baseline-corrected content is small |

---

## 3. Loose ROIs (induction only; `ROI_formation::find_ROI_loose`, `RF.cxx:705-957`)

Wire-Cell (all **VERIFIED**, `RF.cxx` lines in brackets):

1. Data: the loose-LF output, per wire, with `bad` ranges zeroed. The unbinned copy without bad ticks is used for the RMS [719-755].
2. **Rebin**: `signal2[i] = Σ_{j<6} signal1[6i+j]`. This is a sum, not a mean, with `int(nbins/6)` bins. At 4321 ticks the final tick is dropped [722, 760-766].
3. **Threshold**: `th = cal_RMS(unbinned loose-LF) × rebin × l_factor = 6 × 3.5 × rms = 21·rms` in rebinned-sum units, capped at `l_max_th = 10000` [771-773]. The RMS is **not** computed on the rebinned data. The factor 6 assumes fully correlated ticks; for white noise the rebinned σ would be √6·rms, so the cut is effectively stricter than 3.5σ of the rebinned waveform (ASSUMED interpretation).
4. Scan j = 1 … n-2 [785-870]:
   - If `signal2[j] > th`: walk out with `find_ROI_begin/end(signal2, j, 0.7·th)`. Each walk first moves while `content > 0.7·th`, then keeps moving while `local_ave(next) < local_ave(cur) + 25`. `local_ave` is a 3-bin mean with cyclic wrap [619-640]. So the edge goes down to a local minimum, with a 25 e tolerance in rebinned units [642-703]. The ROI is accepted [793-805].
   - Else, if j is a local maximum: walk out using `prev_content` / `next_content` as the walk threshold, then the same local-minimum walk. Accept if **either** `2·s[max] - s[begin] - s[end] > 2·th` (mean prominence > th) **or** both `s[max] - s[max∓3]` > 0.7·th, with ±3 = `l_short_length` clipped to the ROI [806-835].
   - An accepted ROI merges with the previous one if `begin <= prev.end`, then j jumps to `end` [848-868].
5. **Gap merge**, iterated until stable: neighbouring ROIs k and k+1 merge if every bin of the gap between them, minus the straight line from `s[k.begin]` to `s[k+1.end]`, is ≥ 0.7·th [880-926].
6. Scale back: `[6·b, 6·e+5]` [930-939].
7. `extend_ROI_loose`: if a loose start or end lies strictly inside a tight ROI on the same wire, move it to the tight boundary, then merge overlaps [556-617]. A tight ROI that no loose ROI covers is promoted later by `generate_merge_ROIs` (§4).
8. In `load_data` each loose ROI is kept only if some bin of its refinement contents (Wiener_tight without LF, linear baseline subtracted) is > 3·rms, where rms is the tight-LF RMS [`RR.cxx:494-501`].

`max_npeaks` is not a loose-ROI parameter. `r_max_npeaks` belongs to BreakROI (§4). **VERIFIED**

Paper (published text): loose ROIs come from the loose-LF output, aim for high efficiency and low purity, and are then reconciled with tight ROIs (§3.2.3). The rebinning and the local-maximum rules appear **only** in draft text commented out in the arXiv source (`extraction.tex`, "Every six time bins ... rebinned into one", 3.5×RMS capped at 10k, local minima from a ±1-bin average, ±3-bin rule at 70%). That draft matches the code. **VERIFIED** (source read; not published)

Example: loose ROIs are a plain per-wire \|C_loose\| > 3.5·rms(C_loose) on unbinned data, padded ±10 (`ex/roi.py:116-124`). There is no rebinning, no 0.7·th edge extension, no local-minimum walk, no local-maximum/prominence rule, no gap merge, and no 10k cap. **Matters a lot with noise.** Rebinning by 6 is what lets Wire-Cell see long, low-amplitude induction signals, such as tracks at large angle, above loose-LF noise. The prominence rule and the local-minimum edges set the ROI extent. The example's unbinned \|x\| > 3.5σ cut on a loose-LF waveform, which keeps a lot of low-frequency noise, behaves very differently.

Example reconciliation (`ex/roi.py:131-149`): a tight ROI that is not covered merges with any overlapping loose ROIs, or is added as a new loose ROI. This is close in intent to `extend_ROI_loose` plus `generate_merge_ROIs`, but it runs before cleaning. In Wire-Cell the promotion runs after the first `CleanUpROIs`.

---

## 4. Refinement (order, connectivity, cleaning, break, shrink, extend)

### 4a. Call order in `OmnibusSigProc::operator()` (**VERIFIED**)

Pass 1, per plane (`OSP.cxx:1974-2076`): `load_data` → `decon_2D_init` →
induction: `decon_2D_tighterROI` (copy) → `decon_2D_tightROI` → `find_ROI_by_decon_itself(tight, tighter)` → `decon_2D_looseROI` → `find_ROI_loose` → `decon_2D_ROI_refine` → `roi_refine.load_data`.
collection: `decon_2D_tightROI` → `find_ROI_by_decon_itself` → `roi_refine.load_data`.

Pass 2, per plane (`OSP.cxx:2079-2109`): `CleanUpROIs` → `generate_merge_ROIs` → (MP3ROI / MP2ROI only if `use_multi_plane_protection`, which is off).

Pass 3, per plane (`OSP.cxx:2111-2197`): `r_break_roi_loop` = 2 × { `BreakROIs` → `CheckROIs` → `CleanUpROIs` } → `ShrinkROIs` → `CheckROIs` → `CleanUpROIs` → (`CleanUpCollectionROIs` if plane 2, else `CleanUpInductionROIs`) → `ExtendROIs` → `decon_2D_hits` + `apply_roi` ("wiener") → `decon_2D_charge` + `apply_roi` ("gauss").

`CleanUpROIs`, `generate_merge_ROIs`, `CheckROIs`, `BreakROIs` and `ShrinkROIs` act only on planes 0 and 1. Their plane-2 branches are empty or absent (`RR.cxx:704-876, 878-1162, 1164-1287, 1855-1876, 2462-2490`). Collection gets only the load_data check, `CleanUpCollectionROIs` and `ExtendROIs`. **VERIFIED**

Example order (`ex/roi.py:342-362`): formation+reconcile → `connectivity_propagation` → `clean_rois` → `break_rois` (one pass) → `shrink_rois`. Collection has **no** refinement (`ex/roi.py:337-339`).

### 4b. Step by step

| step | Wire-Cell | example | matters? |
|---|---|---|---|
| connectivity graph | `front_rois` / `back_rois` link ROIs on **adjacent wires** that overlap in time (`SR.cxx:83-94`, strict `min_end > max_start`). Induction links only **loose** ROIs (`RR.cxx:505-528, 603-627`); collection links tight ROIs (`RR.cxx:429-457`). `contained_rois[loose]` lists the tight ROIs on the same wire that *overlap* it (`RR.cxx:583-598, 683-696`) | none (ad-hoc overlap checks) | - |
| neighbour propagation | none in Wire-Cell. Creating ROIs is limited to the i/i+2 interpolation (§2) | `connectivity_propagation` copies **every tight ROI onto w±1** when no loose ROI there overlaps it (`ex/roi.py:156-178`) | **yes**: adds ROIs, and so noise charge, on wires with no evidence |
| CleanUpROIs (`RR.cxx:704-876`) | flood fill over front/back links starting from every loose ROI that has `contained_rois` (overlaps a tight ROI). **Removes every loose ROI not in such a cluster**. The cluster has unlimited size | `clean_rois`: per wire, keep a loose ROI if it overlaps a tight ROI on the same wire or on w±1 (one hop, `ex/roi.py:183-210`) | moderate; the paper's "cluster removed entirely" matches Wire-Cell |
| generate_merge_ROIs (`RR.cxx:878-1162`) | every tight ROI that no surviving loose ROI covers becomes a new loose ROI (copy), linked to overlapping loose ROIs on w±1 | done earlier in reconcile | small |
| CheckROIs (`RR.cxx:1164-1287`) | for each link, keep it only if some overlapped bin has `contents > 3·rms` on **both** ROIs (`SR.cxx:61-81`, th = r_th_factor·tight-LF rms of each wire). Otherwise unlink | none | yes, it breaks noise-only connections before the next CleanUp |
| BreakROI (`RR.cxx:1878-2342`) | on the ROI contents (Wiener_tight without LF, baseline-subtracted). `PeakFinding(max_npeaks=200, sigma=2, th_percent=0.1)`, a port of TSpectrum `Search(h, 2, "nobackground new", 0.1)` per the comment at `RR.cxx:1912-1913` (class body not read, UNVERIFIED). Proceeds only if ≥2 peaks have height > `th_peak·rms` = 3·rms (`RR.cxx:1927-1934`). Valleys are minima between consecutive peaks, plus the edges. Peak/valley pairs are kept when `peak - valley > low_peak_sep_threshold = max(1200 e, sep_peak·rms = 6·rms)` (`RR.cxx:1906-1907, 2024-2075`). A linear baseline is re-subtracted between the kept valleys (`RR.cxx:2144-2172`). If that removes all of a contained tight ROI, its contents are restored (`RR.cxx:2183-2207`). Then **2 passes**: find runs of bins < 3·rms, take each run's minimum (an exact zero wins), and re-subtract a linear baseline between consecutive minima (`RR.cxx:2221-2331`). `BreakROI1` then **splits the ROI at every bin with \|content\| < 1e-3**, i.e. at the zeros the baseline fits created (`RR.cxx:2344-2460`). Runs inside the 2× loop (`OSP.cxx:2120-2123`) | `_split_at_valleys`: on \|C_wiener\| (not baseline-subtracted), simple 3-point local maxima > 3.5·rms(C_wiener). Split where valley < 0.3·min(adjacent peaks). Skip segments < 10 ticks, drop pieces < 3 ticks; one pass (`ex/roi.py:215-283`) | yes. The 1200 e floor alone makes Wire-Cell much harder to split on noise wiggles; \|x\| makes the example split on bipolar noise |
| ShrinkROI (`RR.cxx:1614-1853`) | flag bins (a) inside any contained tight ROI, or (b) where a front/back ROI is above **its** 3·rms **and** this ROI's own content is > its own 3·rms. New range = [first flag - r_pad, last flag + r_pad] clipped to the old range, r_pad = 5. No flags means the range is unchanged. Contents are re-baselined | `shrink_rois`: [tight_min - pad, tight_max + pad], then **re-expanded to the full range of any overlapping adjacent loose ROI**, clipped to the original; pad 10 (U/V) (`ex/roi.py:288-332`) | yes. The example's re-expansion largely undoes the shrink |
| CleanUpInductionROIs (`RR.cxx:1371-1612`) | **fake-signal removal**. `Good` = ROI with any content bin > `high_th = 1000·1.0` e **or** mean(contents) > `low_th = 500·1.0` e (`RR.cxx:1375-1378, 1468, 1544`). The quantity is the ROI's own contents: Wiener_tight without LF, after restore_baseline, with linear end-to-end baseline subtraction, and after Break/Shrink re-baselining. A non-Good ROI is **removed unless directly linked (front or back, one hop) to a Good ROI** (`RR.cxx:1473-1506`). The V-plane mean criterion was added at master; the comment says it was "historically ... not included" (`RR.cxx:1542-1544`) | **absent** | **yes**. With noise, this is the main purity cut in absolute electrons; without it every noise ROI that survives the σ-based steps is kept |
| CleanUpCollectionROIs (`RR.cxx:1289-1369`) | same rule on collection tight ROIs, with 1000 e peak or 500 e mean (comment: "electrons, about 1/2 of MIP per tick"), linked one hop to a Good ROI | **absent** (collection has no refinement at all) | **yes** |
| ExtendROIs (`RR.cxx:2971-3156`) | for every plane, ROIs on each wire are sorted. ext range = union of own [start, end] with the [start, end] of every front/back-linked ROI. If consecutive ext ranges on one wire overlap, both are cut at `(prev.end + next.start)/2`. `apply_roi` uses the ext range | not separate. The example's shrink re-expansion is the nearest analogue | moderate |
| multi-plane protection | off by default (`OSP.h:273`, `OSP.cxx:2092`). With it off, `proteced_rois` stays empty and BreakROI's protection does nothing (`RR.cxx:1880, 2087-2118`; that it is empty is ASSUMED, MP3ROI/MP2ROI bodies not read) | none | no (both off) |

Paper §3.2.4 describes only Clean, Break and Shrink, applied "iteratively ... in sequence". CheckROIs, the fake-signal thresholds and ExtendROIs appear only in draft text commented out in the source. There the fake-signal thresholds are 1200 e (collection) and 1500 e (isolated induction), which differ from the code's 500/1000. **VERIFIED** (source read)

---

## 5. Final charge

| aspect | Wire-Cell | paper | example | matters? |
|---|---|---|---|---|
| output masked | Gaus_wide decon (`OSP.cxx:2179-2184`). Also Wiener_wide for the "wiener" frame (`OSP.cxx:2159-2161`) | "applied to the deconvolved signal without the LF filter", Gaussian (§3.2, §3.2.4) | C_gauss (`ex/pipeline.py:112`, `wrap:231`) | filter width differs (§0) |
| pre-ROI baseline | collection: `restore_baseline` (two-pass median) on Gaus/Wiener output (`OSP.cxx:1901-1903`). Induction: none | - | none | small |
| induction per-ROI baseline | over the **ext** range `[s, e]` inclusive: `x - (x[s] + (x[e]-x[s])(i-s)/(e-s))` (`RR.cxx:131-142, 160-171`) | linear, so ROI boundaries are exactly zero (§3.2.4) | `np.linspace(seg[0], seg[-1])` over `[t_start, t_end)` (`ex/roi.py:383-387`), the same formula over the example's own ranges | same form; ranges differ |
| collection per-ROI | copy only, no baseline (`RR.cxx:195-211`) | - | copy only (`ex/roi.py:383`) | same |
| outside ROIs | 0 (`RR.cxx:124, 145-147`) | 0 | 0 | same |
| truncation | `int content = ...` truncates each sample to an integer number of electrons (`RR.cxx:137, 166, 206`) | - | float | negligible |
| negatives | `save_data` with `save_negative_charge=false` (`OSP.h:263`) sets **each sample** ≤ 0 to 0 and zeroes bad channels (`OSP.cxx:515-531`) | - | kept in C_charge. The wrapper clips at 0 only **after summing 2-wire × 16-tick cells** (`wrap:235-241`) | moderate. Per-sample clipping biases noise positive; per-cell clipping biases it less |

---

## 6. Example-only behaviour, and Wire-Cell steps the example omits

Example-only (no Wire-Cell counterpart):
- `roi_abs_threshold` (350/500/300 e) is used as the **whole** threshold when `noiseless=True` (`ex/roi.py:75-79`). Wire-Cell has only the `+1` e floor (`RF.cxx:464`). The values are the paper's quoted *typical* 3.5σ/5σ levels (§3.2.3), not a code parameter. **VERIFIED**
- \|signal\| thresholding everywhere: `find_rois_1d` (`ex/roi.py:33`) and `break_rois` (`ex/roi.py:232`). Wire-Cell is signed everywhere: `RF.cxx:488`, `RF.cxx:793`, `SR.cxx:110`. **VERIFIED**
- `noiseless` DC handling keeps F(0)=1 for the collection Wiener and for the Gaussian (`ex/deconv.py:178-179`). Wire-Cell always zeroes DC in time filters (`flag:true`). **VERIFIED**
- Tight ROIs copied to w±1 (`ex/roi.py:156-178`). **VERIFIED**
- Shrink re-expanded to adjacent loose ROI ranges (`ex/roi.py:315-324`). **VERIFIED**
- The ROI functions always receive `PLANE_PARAMS[plane]` and **ignore the `params` overrides** (`ex/pipeline.py:105-108`, `wrap:228-230`). Overrides change the filters but not `roi_th_factor`, `roi_pad` or `roi_abs_threshold`. **VERIFIED**

Wire-Cell steps the example omits:
- The tighter-LF (0.1 MHz) output joined into tight ROIs by OR.
- The loose algorithm: rebin ×6, 21·rms capped at 10k, 0.7·th edges, local-minimum walk with 25 e tolerance, prominence and ±3-bin rules, gap merge.
- The i/i+2 connectivity interpolation, for all planes.
- The `load_data` 3σ re-check on baseline-subtracted no-LF contents.
- Cluster-wide CleanUpROIs.
- CheckROIs, the link validation.
- BreakROI with its 1200 e / 6σ separation, re-baselining and zero-crossing splits, run 2×.
- Shrink by flagged bins.
- **CleanUpInductionROIs and CleanUpCollectionROIs (500 e mean / 1000 e peak fake-signal removal).**
- ExtendROIs.
- Median `restore_baseline`.
- The two-pass 5σ-clipped RMS.
- Use of the tight-LF RMS as the single noise scale for all refinement thresholds.
- Channel-mask handling (`bad` ranges excluded from the RMS; tight LF instead of loose LF near bad or lf_noisy channels).
- Per-sample zero clipping of the output.
- The master-only pre-decon `rebase_waveform`.

---

## Most consequential differences (ranked, for noisy input)

1. **Filter parameters are wrong by large factors** (§0). Wiener is about 2× too wide, Gaussian about 2.2-2.7× too wide, the induction wire filter about 3.6× too narrow, loose LF 4× too high and tight LF 1.75× too high. Both the paper figures and `filt.jsonnet` contradict the example's figure readings. This changes noise levels and RMS-based thresholds before any ROI logic runs.
2. **The loose-ROI algorithm is not implemented** (§3). Wire-Cell uses 6-tick rebinned sums, a 21·rms threshold (cap 10k), 0.7·th local-minimum edges, prominence rules and gap merging. The example uses unbinned \|x\| > 3.5σ.
3. **No absolute fake-signal removal** (§4). Wire-Cell drops ROIs with no bin > 1000 e and mean ≤ 500 e unless they are adjacent to a good ROI, on both induction and collection. The example has none, and has no collection refinement at all.
4. **The noise RMS differs in source and estimator** (§2). Wire-Cell uses the per-wire tight-LF (or collection Wiener) RMS from a 16/50/84-percentile seed followed by a \|x\| < 5σ RMS pass, and reuses that one RMS for every refinement threshold. The example uses pass 1 only, recomputed separately on each array.
5. **\|x\| vs signed thresholds, and th_factor 3.5 vs 3 on induction.** The example follows the paper's 3.5; the code uses 3.
6. **ROI-creating propagation** to w±1, and the shrink re-expansion, both make the example's ROIs larger and more numerous.

---

## Addendum (checked by the main analysis while porting the code; all VERIFIED by reading the cited lines)

- **CleanUpInductionROIs never removes U-plane ROIs at master.** The U branch
  (`RR.cxx:1460-1520`) builds `Good_ROIs`, collects `Bad_ROIs`, and then its final
  loop only computes `chid` (the `cout` is commented out). It never unlinks, erases
  or deletes anything. The V branch (`RR.cxx:1580-1606`) does remove. The audit
  above, §4b, describes the intended rule. The executed behaviour is: fake-signal
  removal acts on V and on collection only.
  - `sp/wirecell_sp.py` reproduces this.
  - The `wc_fixU` sweep setting applies the removal to U as well.
- **Wire-Cell zero-pads both FFT axes before deconvolving.**
  - Ticks: `m_fft_nticks = m_nticks + FR length/5` (`OSP.cxx:863-865`, fft_flag 0, as in uboone `sp.jsonnet`).
  - Wires: `m_fft_nwires = m_nwires + n_FR_paths - 1` (`OSP.cxx:878-887`).
  - This is exactly the cyclic readout-edge fix found empirically in Addition 1 of the README.
  - The example deconvolves cyclically on the bare readout.
  - The port pads by 200 ticks and 20 wires. The intent is VERIFIED; the exact pad sizes are ASSUMED, because the MicroBooNE FR length was not checked.
- **Wire-Cell divides by the response with no floor** (`m_c_data / c_resp`, `OSP.cxx:1274`). NaN/inf are zeroed afterwards (`OSP.cxx:1310-1320`). The example floors |R| at 1e-6·max, and the port keeps that floor so the two chains share the deconvolution.
- **PeakFinding defaults.** These are `PeakFinding(200, sigma, threshold, backgroundRemove=false, deconIterations=3, markov=true, averWindow=3)` (`sigproc_src_PeakFinding.h:12-13`). BreakROI calls it with `(r_max_npeaks=200, r_sigma=2, r_th_percent=0.1)`. The threshold is therefore 0.1 *percent* of the maximum (`PF:430-436`), which makes it effectively no cut.
  - The port is a line-by-line translation of `SearchHighRes` (`PF:65-479`): the Markov smoothing and the 3-iteration Gold deconvolution, vectorised.
  - Checked on synthetic peaks: it finds the injected positions.
- **Filter frequency axis.** `HfFilter`/`LfFilter` use `use_negative_freqs{true}` (`HfFilter.h:28`), `freq = i/N·2·max_freq`, folded. So the wire filters' `max_freq = 1` puts Nyquist (0.5 cycles/wire) at 1, as §0 states.
