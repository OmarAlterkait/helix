# The exact response of our simulation, and what it changes for the classical benchmark

2026-10-08. Follow-up question: "don't we have the exact kernel?"

**Short answer.** Yes, the forward model is exactly known, and I reproduced the stored sensor from it. For the classical chain the exact kernel changes nothing measurable: the least-squares kernel fitted to our sim already matches it to 1-4%.

The simulator is public; I cloned it into `reference/JAXTPC` @ 806f9b3 (HEAD). The production commit was **exported read-only** with `git archive` to `/pscratch/sd/o/oalter/helix_work/uboone_sp/jaxtpc_1ea2282`. Citations of the form `file:line` refer to that commit. Labels: [V] verified by reading the code or by running it; [A] assumed.

## Summary

1. **Provenance.** The data was made by JAXTPC commit **1ea2282** (2026-05-26). The tree was dirty, and HEAD has 69 more commits. [V]
2. **Perfect knowledge is established.** I re-simulated the clean sensor of ev260-263 (24 planes) from the step files with JAXTPC's own functions. [V]
   - Mean residual is 0.6% / 0.8% / 0.9% (U / V / Y), maximum 1.3%.
   - Total charge agrees to 0.9998.
   - 77-94% of the stored nonzero samples are reproduced exactly.
   - The floor of this test is the step files' 0.3 mm position quantisation, 0.75-1.1%; the re-simulation is at that floor.
3. **The exact effective kernel for per-wire hits truth, built from JAXTPC's code, is the kernel we had already fitted** (1.3 / 3.8 / 1.9% L2 apart). [V]
   - Forward-folding held-out hits gives the same residual: exact 0.030 / 0.031 / 0.021 vs fitted 0.027 / 0.035 / 0.017.
   - A drift-dependent kernel family, fed with each hit's true drift, does not improve on it: 0.031 / 0.031 / 0.020.
   - The 2-3% that remains is irreducible for any kernel acting on per-wire hits. The hits truth carries neither the sub-wire position nor the drift time; deposit times spread over ±1.2 ms.
4. **The example's kernel (`avg_response_*`) is the same JAXTPC kernel, plainly averaged over the pitch.** Your hypothesis is right, with one refinement. [V]
   - Running the example's recipe on the production kernel file reproduces it to 3e-8.
   - It omits JAXTPC's linear sub-tick interpolation: a half-tick shift, 13-20% in L2.
   - It also omits diffusion in the kernel as seen by per-wire hits: a further 5-9%.
   - With the half-tick interpolation added it equals our no-diffusion pitch average to 0.7-1.7%.
5. **Benchmark with the exact kernel.** The whole tuned classical chain was re-run on all 128 test events plus the tuning split. [V]
   - The fitted-kernel picks, frozen and applied to the exact-kernel deconvolution, give near-activity pp efficiency **0.507 [0.494-0.528]** at 0.1-0.2 MeV, vs 0.516 [0.502-0.533] with the fitted kernel.
   - Re-tuning on the exact-kernel tuning split gives 0.486 [0.473-0.500]. On U it picks a statistic that beats the fitted-kernel pick by 0.001 on the tuning objective but scores lower on test. That is selection noise, not a kernel effect.
   - Every individual statistic moves by at most 0.025.
   - The headline does not change: classical ~0.51, M4n 0.579, 64-label model 0.429.

## 1. The wire forward model that produced `wire_test_00_00_02`

### Provenance [V]

The `config` group of every sensor, hits and step file carries the run attributes:
- `git_commit` = 1ea22829e3ba…, `git_dirty` = True, `production_version` = test_00_00_02;
- `include_electronics` = False, `include_digitize` = True;
- `include_intrinsic_noise` = `include_coherent_noise` = False;
- `threshold_adc` = 2.0, `electrons_per_adc` = 182, `lifetime_us` = 10000, `velocity_cm_us` = 0.16;
- `pre_window_us` = `post_window_us` = 405, `num_time_steps` = 4321, `time_step_us` = 0.5;
- `recombination_model` = emb;
- `source_file` = `/sdf/data/neutrino/doraemon/test_00_00_02/run_*/edepsim_*.h5`.

The hits event attribute is `threshold` = 25.0. The batch timestamp is 2026-06-03.

**Config mismatch.** At 1ea2282 the only wire production config, `config/production_cubic_wireplane_config.yaml`, has `threshold_adc: 0.2` and `corr_threshold: 1.0`. The data used 2.0 and 25. Those values first appear in the committed `production_cubic_wireplane_doraemon_{100,300}micro.yaml` (commit 21d3a13, 2026-06-08). The 100-micro file was generated 2026-05-22 and the 300-micro file 2026-06-02.
- So the run used an uncommitted local config, consistent with `git_dirty`.
- Both doraemon files agree on every physics-relevant value (thresholds 2.0 ADC / 25 e-, `inter_thresh` 1.0); they differ only in padding and chunk sizes.
- The detector config (`config/cubic_wireplane_config.yaml`) is identical at 1ea2282 and at HEAD.

### Forward chain, per deposit [V, read; and reproduced by running, §2]

**Physics constants** (`cubic_wireplane_config.yaml`):
- Two volumes, x ∈ [-216, 0] and [0, 216] cm. The anode is at x_min (volume 0, drift −x) or x_max (volume 1, drift +x).
- Planes U / V / Y at 60° / −60° / 0°, 3 mm pitch, 0.6 / 0.3 / 0 cm from the anode.
- Drift velocity 1.6 mm/µs; D_L = 7.2 cm²/s; D_T = 12.0 cm²/s; lifetime 10 ms.
- E = 500 V/cm, EMB recombination, sampling 2 MHz, 12-bit ADC.
- Pedestals 1843 (U/V) and 410 (Y); gain_scale 1.0.

**Charge.** The step file's `charge` is the recombined charge before attenuation. The re-simulation needs attenuation to match: residual 0.6-0.9% with it, 8.6% without. [V]

**Drift** (`physics.py:147-209`):
- Local frame `x_local = drift_dir·(x_anode − x)`.
- The drift distance to plane p is `x_local − d_p` (`drift.py:58-87`), and t_drift = distance / v.
- Attenuation is `exp(−t_drift / 10 ms)` (`:177`).
- **Readout tick** is `t_drift + t0 + 405 µs` (`:180`). Deposits outside [0, 2160.5 µs) are zeroed (`:184`).
- t0 is the GEANT4 deposit time (`loader.py:443-445`).
- **In this production t0 spans about ±1.2 ms** (5-95% per volume: −1.6 ms … +1.2 ms; overlaid interactions or cosmics). The **step files do not store t0** (fields: positions, charge, de, dx, phi, photons). I recovered it per deposit group from the hits truth (§2). Consequence: **a hit's tick is not its drift time**, so no per-wire method can infer the diffusion level from the tick.

**Wire geometry** (`wires.py:24-62`):
- `r' = y sinθ + z cosθ`.
- The closest wire is `round(r'/pitch) + index_offset`.
- The sub-wire offset is `(r' − idx·pitch) / pitch` ∈ [−0.5, 0.5) (`wires.py:545`).

**Time binning** (`wires.py:541-542`): `time_index = floor(tick/0.5 µs)`, and `t_offset` is the fractional part ∈ [0, 1).

**Response kernel** (`tools/responses/{U,V,Y}_plane_kernel.npz`; the files are identical at HEAD):
- Shape is 160 time bins × 201 sub-wire bins: 0.5 µs, 0.1 pitch, ±10 wires.
- `time_zero_bin` = 128; `units` = ADC_per_electron.
- `scaling_mode` = first_principles, with 14 mV/fC gain, 1.2 post-amplifier, peaking time 2.18 µs and 2000 mV / 12 bit.
- `adc_per_electron` is 0.00515 (U/V) and 0.00533 (Y).
- So the **cold-electronics shaping and gain are already inside the kernel**. `include_electronics` = False means no additional RC⊗RC (`electronics.py`'s RC chain is not applied). `electrons_per_adc` = 182 is metadata only and never multiplies the signal (it is used only by the noise model).
- The wire kernel is left-right symmetric: ||K − K_mirror|| = 0.

**Diffusion table** (`kernels.py:114-190, 223`):
- `num_s = 16` levels, s ∈ linspace(0, 1). The default is `config.py:226`; `simulation.py:163-175` does not pass it, and `run_batch.py` has no option for it. HEAD has 32 (`kernels.py:193`), a post-production change.
- Each level is the kernel convolved with a separable Gaussian of σ_T = s·σ_T,max (wire) and σ_L = s·σ_L,max (time).
- The kernel is reflect-padded at the leading edges only (`:166`) and convolved with `SAME`.
- σ_T,max = √(2·D_T·t_max) = 0.18 cm = 0.60 pitch. σ_L,max = √(2·D_L/v²·t_max) = 0.87 µs = 1.74 ticks, for t_max = 216 cm / 0.16 cm/µs = 1350 µs (`geometry.py:48-52`).
- **Per deposit**, s = √(drift distance / 216 cm) (`simulation.py:380/391`), so σ ∝ √drift.

**Runtime interpolation** (`kernels.py:260-345`):
- Linear in s between table levels, and linear in the sub-wire offset between 0.1-pitch columns.
- For output wire position j, the column read is `center − 10j + 10·w_offset`.
- **Linear in time between adjacent bins**: `t·K[k] + (1 − t)·K[k+1]` (`:339`). The output's zero bin is `time_zero_bin − 1 = 127` (`:445`).
- Placement: `wires.py:567-640`, at wire `idx − wire_zero_bin_out + j` and tick `time_index − 127 + k`.

**Sensor.**
- Σ_deposits q·attenuation·contribution.
- Digitize: `round(signal·1.0 + pedestal)`, clip to [0, 4095], minus pedestal (`electronics.py:417-423`).
- Sparse save keeps |v| ≥ 2.0 ADC (`save.py:236`). That is the zero-suppression our analysis calls "|v| < 2 dropped".

**Hits truth** (`track_hits.py:900-1010`, `wires.py:250-376`):
- The **same deposit's attenuated charge, diffused by a Gaussian with σ(t_drift)**: σ_wire = √(2 D_T t_drift)/pitch and σ_time = √(2 D_L t_drift/v²)/0.5 µs (`wires.py:335-339`).
- It is CDF-integrated into **integer wires × ticks** over (2K_w+1) × (2K_t+1) cells, with K = ceil(3σ_max) = 2 wires × 6 ticks (`config.py:385-386`). Sub-bin centres sit at the tick-bin centre (`:344`).
- Summed per deposit group (≤ 5 deposits, 5 mm gap).
- Kept if > 1 e- in the box (`inter_thresh`) and > 25 e- per group × wire × tick at CSR encode. Charges are u16-quantised relative to the group peak.
- The angular-scaling factor is computed but commented out (`wires.py:330`).
- **So the hits truth is post-diffusion, post-attenuation charge per integer wire and tick, at the readout-tick time reference (including t0). It has no sub-wire information.** It holds 98.3% of the attenuated deposit charge. The other 1.7% is lost to the 25 e- per-group threshold, mostly in diffusion tails (ev260-263, all planes 0.979-0.987) [V].

## 2. Re-simulation from the step files (`v2/resim.py`, `v2/results/resim.json`) [V]

**Method.** The code calls JAXTPC's own functions at 1ea2282 on ev260-263:
- `correct_drift_for_plane`, `compute_wire_distances`, `prepare_deposit_for_response`;
- `load_response_kernels(num_s = 16)` with the config's σ maxima, `interpolate_diffusion_kernel_batch`;
- `accumulate_response_signals`, `_digitize_signal`, and the save threshold.

It runs on 1 GPU of the allocation. Inputs are the step positions (uint16 × 0.3 mm + origin, converted from global to local frame), the step `charge`, and attenuation.

**t0 recovery.** The step files carry no t0, so I recover one per deposit group: the hits charge-weighted mean (bin + 0.5)·0.5 µs minus the charge-weighted mean of (t_drift + 405 µs) of the group's deposits, pooled over the three planes. Deposits whose group left no hits are dropped, because their time is unknown and they lie outside the window. Groups with hits: 33-82% per volume.

| | U | V | Y |
|---|---|---|---|
| ‖stored − resim‖ / ‖stored‖, mean over 8 planes (max) | **0.0062** (0.0078) | **0.0083** (0.0099) | **0.0094** (0.0134) |
| stored nonzero samples reproduced exactly | 0.94 | 0.92 | 0.77 |
| Σ resim / Σ stored | 0.9998 | 0.9998 | 0.9999 |
| resim vs resim with positions jittered by U(±0.15 mm) (the test's floor) | 0.0075 | 0.0110 | 0.0090 |
| without attenuation | 0.086 | 0.086 | 0.087 |
| with t0 = 0 | 1.65 | 1.64 | 1.62 |

The remaining difference equals the floor set by the step files' position quantisation. JAXTPC 1ea2282 with the config above is therefore the exact forward model of our data.

## 3. The exact effective kernel for per-wire hits (`v2/exact_kernel.py`, `response/exact*_{U,V,Y}.npz`)

**Why one kernel cannot be exact.** The hits give per-wire, per-tick charge after diffusion, so a deposit's sub-wire position, its sub-tick time and its diffusion level are not in the truth. No translation-invariant kernel acting on hits can reproduce the sensor exactly.

**How the exact kernel is defined here.** It is the least-squares kernel over the deposit population, built from JAXTPC's two code paths on synthetic unit-charge point deposits. Each deposit has a uniform sub-wire offset, a uniform sub-tick offset and a drift giving s:
- the sensor through `interpolate_diffusion_kernel_batch` + `accumulate_response_signals`;
- the hits through `prepare_deposit_with_diffusion`.

The outputs are:
- **`exact_<P>`**: drift-averaged, with drift uniform in [0, 216] cm.
- **`exact_<P>_s<k>`**: a family at 9 fixed levels s = 0, 0.125, …, 1.
- **`exact0_<P>`**: s = 0.

Support is 21 wires × 211 ticks (dt ∈ [−150, 60]). That support is sufficient: < 1e-4 of the kernel energy sits in the outermost wires, < 3e-5 in the first and last 10 ticks.

**Kernel shapes** (`v2/results/kernel_shapes.json`): relative L2 difference from `exact_<P>` after the best integer time shift and gain.

| | U | V | Y |
|---|---|---|---|
| our fitted kernel | **0.013** | **0.038** | **0.019** |
| example `avg_response` | 0.181 (shift 1) | 0.217 (shift 1) | 0.105 |
| exact0 (no diffusion) | 0.085 | 0.086 | 0.053 |
| example vs exact0 | 0.165 | 0.201 | 0.135 |
| example + JAXTPC half-tick interpolation vs exact0 | **0.017** | **0.009** | **0.007** |
| family s = 0.5 vs drift-averaged | 0.014 | 0.007 | 0.005 |
| family s = 1 vs drift-averaged | 0.93 | 0.49 | 0.06 |

At s = 1 the hits are Gaussian-blurred (σ = 0.6 pitch, 1.7 ticks). Recovering a per-wire kernel then means dividing out that blur. The problem is ill-posed, and the least-squares solution is noise-dominated on U/V. This is one reason a drift-dependent family brings nothing.

**Forward-fold validation** (`v2/kernel_validate.py`, `v2/results/kernel_validate.json`). Held-out ev260-263, 24 planes. Metric: ‖C − store(H ⊛ R)‖ / ‖C‖ with a zero-padded linear convolution, then the sensor writer's rounding and |v| < 2 suppression.

| kernel | U | V | Y |
|---|---|---|---|
| example `avg_response` | 0.110 | 0.162 | 0.074 |
| exact0 (no-diffusion pitch average) | 0.058 | 0.081 | 0.034 |
| **exact (drift-averaged)** | **0.030** | **0.031** | **0.021** |
| exact family, each hit at its group's true drift (needs step positions; more truth than any per-wire method has) | 0.031 | 0.031 | 0.020 |
| our least-squares fit (8 probe-train events) | 0.027 | 0.035 | 0.017 |

The earlier figures of 0.041 / 0.039 / 0.021 used a cyclic forward model on ev260-262; this table uses a linear one on ev260-263.

What the validation shows [V]:
- The fitted and exact kernels are equivalent: the fit is 0.003 better on U and Y and 0.004 worse on V, both at the floor.
- A drift-dependent family gives no forward-model gain, even with oracle drift.
- Drift-dependent deconvolution is therefore unwarranted, and **I did not run it**. It would add no information, its s → 1 kernels are ill-conditioned, and the drift of a hit is unknown from its tick (t0 spread).
- The irreducible ~2-3% comes from information the per-wire truth does not carry: the sub-wire position, the sub-tick time and the hits threshold. The re-simulation from deposits reaches 0.6-0.9%.

## 4. The classical benchmark with the exact response

**Rerun.** The full tuned chain was re-run with the drift-averaged exact kernel (`v2/classical.py --kernel exact` with both `classical_extra.py` rounds) on ev000-095 and all 128 test events, scored with `v2/score.py --picks-from`. Output: `v2/results/exact/benchmark.json`.

Near-activity 1%-FPR efficiency at 0.1-0.2 MeV, per-plane thresholds, event bootstrap 16-84%:

| classical row | fitted kernel | exact kernel | U / V / Y (exact) |
|---|---|---|---|
| MicroBooNE default | 0.081 [0.067-0.095] | 0.078 [0.066-0.095] | 0.20 / 0.02 / 0.01 |
| default without the fake cut | 0.338 | 0.338 | 0.20 / 0.02 / 0.86 |
| tuned ROI charge, fitted-kernel picks frozen | 0.436 [0.421-0.453] | 0.436 [0.425-0.454] | 0.34 / 0.11 / 0.92 |
| **tuned best statistic, fitted-kernel picks frozen** | **0.516 [0.502-0.533]** | **0.507 [0.494-0.528]** | 0.34 / 0.30 / 0.92 |
| tuned best statistic, re-tuned on exact-kernel ev000-095 | - | 0.486 [0.473-0.500] | 0.27 / 0.30 / 0.93 |
| model M4n (presence) / 64-label (presence), unchanged | 0.579 / 0.429 | 0.579 / 0.429 | 0.50 / 0.38 / 0.90 (M4n) |

**SNR bins** (near, pp; tuned frozen picks), fitted → exact:
- 5-7: 0.30 → 0.31;
- 7-10: 0.80 → 0.79;
- 10-15: 0.98 → 0.98.

**Per statistic on test, fitted → exact:**
- `zlf06`: 0.495 → 0.482;
- `ztlf`: 0.491 → 0.489;
- `zor06a2`: 0.473 → 0.472;
- `zlf`: 0.438 → 0.434;
- `zw`: 0.303 → 0.302;
- Gaus_wide `g`: 0.260 → 0.262;
- oracle-ROI survival: 0.999 → 0.999.

**Re-tuning.** On U, the exact-kernel tuning split ranks `zlfa2` (J 0.353) over `zlf04` (0.352), `zor` (0.351) and `zlf06` (0.345); on the fitted-kernel tuning split `zlf06` won by 0.003. On test, `zlfa2` scores 0.27 on U against 0.34 for `zlf06`. The drop comes from candidates near-tied on 96 tuning events, not from the kernel. The frozen-picks row is the like-for-like comparison.

### Perfect-knowledge rungs in the Wire-Cell port (`v2/rungs.py`, `v2/results/rungs.json`; 128 test events)

**Setup.** Oracle-ROI Gaus_wide charge (truth ROIs) per fine cell, truth > 0 cells. Each cell gives median predicted/true and 68% half-width. Totals are the sum of predictions over all cells divided by the hits truth. Fitted → exact; the U and V rows are alike.

| rung (input) | U total | Y total | U 2-4k e- | U 8-16k e- | Y 2-4k e- | Y 16-64k e- |
|---|---|---|---|---|---|---|
| R0syn: hits ⊛ the same kernel, float | 0.992 → 0.992 | 0.998 → 0.998 | 0.91/0.71 → 0.91/0.71 | 1.16/0.39 → 1.16/0.39 | 0.96/0.59 → 0.96/0.59 | 0.96/0.10 → 0.96/0.10 |
| R0: stored clean sensor | 0.995 → 1.004 | 1.001 → 1.012 | 0.92/0.76 → 0.93/0.76 | 1.18/0.39 → 1.19/0.40 | 0.96/0.61 → 0.96/0.65 | 0.96/0.10 → 0.97/0.11 |
| R1: + incoherent noise | 1.005 → 1.015 | 1.007 → 1.018 | 1.05/0.81 → 1.07/0.82 | 1.18/0.41 → 1.19/0.41 | 1.01/0.63 → 1.01/0.66 | 0.96/0.10 → 0.97/0.11 |
| R2: + coherent noise, gate | 1.004 → 1.013 | 1.004 → 1.015 | 1.05/0.84 → 1.06/0.85 | 1.18/0.41 → 1.19/0.42 | 0.99/0.64 → 0.99/0.67 | 0.96/0.10 → 0.97/0.11 |

No-ROI significance `zlf` (the tuned U statistic family), near pp 0.1-0.2 MeV, fitted → exact:

| rung | pooled | U / V / Y | SNR 5-7 | SNR 7-10 |
|---|---|---|---|---|
| R1 | 0.459 → 0.459 | 0.355 / 0.149 / 0.928 → 0.359 / 0.144 / 0.932 | 0.22 → 0.22 | 0.69 → 0.68 |
| R2 | 0.438 → 0.434 | 0.315 / 0.133 / 0.922 → 0.311 / 0.126 / 0.924 | 0.20 → 0.20 | 0.65 → 0.64 |

What the rungs show [V]:
- **The kernel choice changes no rung beyond 0.01** in per-cell bias or width, or in detection.
- **The exact kernel's totals are 1-1.8% higher.** It is calibrated to the full deposit charge, while the hits truth holds 98.3% of it (§1). The fitted kernel, trained on hits, absorbs that loss (gain 0.99-1.01 relative to exact).
- **The per-cell width at perfect knowledge (R0syn) is already 0.71 at 2-4k e- and 0.39 at 8-16k e- on U.** That is the Wire-Cell filters' smearing across 2 × 16 cells (Gaus_wide σ_t = 2.9 ticks, wire filter). Noise adds little (R1: 0.81 / 0.41).
- **The model's charge head** has 0.44 / 0.11 at those bins (`aspects.md`). Its per-cell advantage over the classical chain therefore does not come from imperfect response knowledge on the classical side.

### Tuned chain, charge fidelity (aspects rerun with the exact kernel, `v2/results/exact/aspects.json`)

Tuned ROI charge, per cell, fitted → exact:
- U 4-8k e-: 0.91/0.72 → 0.91/0.72;
- V 8-16k e-: 1.04/0.45 → 1.04/0.45;
- Y 4-8k e-: 1.00/0.57 → 1.01/0.59. The Y pick moved from ×0.67 to ×0.83 of the threshold on re-tuning.

Total charge U / V / Y: 0.855 / 0.840 / 1.000 → 0.862 / 0.843 / 1.005. Empty cells above 1 Q0: 87k → 87k.

## What is verified, what is assumed

**Verified [V]:**
- the provenance attributes and the commit;
- the forward chain, read at 1ea2282;
- the re-simulation of the stored sensor to the position-quantisation floor;
- the exact effective kernels and their forward-fold residuals;
- the reconstruction of the example's recipe;
- the half-tick explanation of the example;
- hits completeness;
- the full classical rerun with the exact kernel.

**Assumed [A]:**
- that the uncommitted parts of the dirty production tree changed nothing physics-relevant beyond the config thresholds. The re-simulation at the floor supports this; the dirty `save` step writer, which stores no theta / t0, was evidently modified.
- that `num_s` = 16 was used. HEAD's 32 would differ only through the s-interpolation; the re-simulation used 16 and reached the floor.
- the t0 recovery: per group, from the hits.
- that a drift distribution uniform in x is right for the drift-averaged kernel. The fitted kernel, trained on real events, agrees with it to 1-4%.

## Reproduce (compute node, container; JAXTPC 1ea2282 exported with `git -C reference/JAXTPC archive 1ea2282 | tar -x -C <scratch>/jaxtpc_1ea2282`)

```bash
H=baselines/uboone_sp/v2; S=/pscratch/sd/o/oalter/helix_work/uboone_sp/v2
CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python $H/resim.py --events 260-263 --out $H/results/resim.json
for P in U V Y; do scripts/helix_run.sh python $H/exact_kernel.py --planes $P --out baselines/uboone_sp/response; done
JAX_PLATFORMS=cpu scripts/helix_run.sh python $H/kernel_validate.py --events 260-263 --out $H/results/kernel_validate.json
scripts/helix_run.sh python baselines/uboone_sp/dev/kernel_shapes.py          # -> $H/results/kernel_shapes.json
scripts/helix_run.sh python baselines/uboone_sp/dev/hits_completeness.py      # -> $H/results/hits_completeness.json
scripts/helix_run.sh python $H/classical.py --events 0-95,260-387 --kernel exact --out $S/cls_exact   # + classical_extra.py rounds 1, 2 (--kernel exact)
scripts/helix_run.sh python $H/score.py --cls $S/cls_exact --model $S/model --picks-from $H/results/benchmark.json --out $H/results/exact
# rungs: classical.py --settings NONE --full qor,g,zlf,zw --input {syn,clean,inc} --kernel {fit,exact} -> $S/rungs/<input>_<kernel>
#        (gate_<kernel> = the cls_full / cls_full_exact passes), then:
scripts/helix_run.sh python $H/rungs.py --root $S/rungs --out $H/results/rungs.json
```
