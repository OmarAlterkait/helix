# R2 (smart gate) qualification report — 2026-07-23

**Question:** does the coefficient-space smart gate (R2) qualify as the default
coherent removal, replacing classic multipass (R1)? Under which settings?

**Protocol** (`qual.py`): 12 doraemon events (run_0027575715, volume_0 U/V/Y,
noise-free clean truth) × 2 seeds × {colored, white} intrinsic-noise arms with
the faithful pimm-data injectors (1/f + β-coupled coherent; ENC intrinsic;
digitized). Metrics verbatim from `coherent_coeffs/smart.py::metrics`; end-to-end
= production threshold (per-band MAD, κ=1, approx included) → kept counts +
recon F0. `oracle` = coherent never injected (the ceiling). n = 24 per cell.

## Headline table (mean ± SE; win = per-cell F0 > R1)

```
noise   pl arm            F0±SE      coh_left  n_kept  kept/oracle  winF0>r1
colored U  r1         0.8928±0.0027    0.796   65516    1.079
colored U  r2_k3      0.8974±0.0022    0.676   67985    1.120        0.83
colored U  r2_k3.5    0.8934±0.0025    0.609   61658    1.016        0.46
colored U  r2_k4      0.8886±0.0028    0.630   59776    0.985        0.25
colored V  r1         0.8848±0.0018    0.732   58485    1.116
colored V  r2_k3      0.8890±0.0015    0.576   60216    1.149        0.79
colored V  r2_k3.5    0.8884±0.0015    0.451   54012    1.031        0.67
colored V  r2_k4      0.8865±0.0016    0.425   52245    0.997        0.46
colored Y  r1         0.9544±0.0007    1.156   52524    1.042
colored Y  r2_k3      0.9584±0.0007    0.612   55256    1.097        1.00
colored Y  r2_k3.5    0.9577±0.0007    0.517   51594    1.024        0.88
colored Y  r2_k4      0.9568±0.0008    0.515   50500    1.002        0.79

white   U  r1         0.8909±0.0028    0.964   90178    1.303
white   U  r2_k3      0.8977±0.0023    0.679   75200    1.087        0.88
white   U  r2_k3.5    0.8938±0.0024    0.607   70123    1.013        0.46
white   U  r2_k4      0.8894±0.0028    0.626   68960    0.996        0.29
white   V  r1         0.8818±0.0018    0.953   85567    1.438
white   V  r2_k3      0.8901±0.0014    0.572   65612    1.102        0.92
white   V  r2_k3.5    0.8885±0.0017    0.444   60600    1.018        0.75
white   V  r2_k4      0.8866±0.0018    0.422   59460    0.999        0.67
white   Y  r1         0.9523±0.0008    1.575   63431    1.098        1.00
white   Y  r2_k3      0.9585±0.0007    0.617   61475    1.064        1.00
white   Y  r2_k3.5    0.9580±0.0008    0.515   58445    1.012        1.00
white   Y  r2_k4      0.9570±0.0009    0.509   57679    0.998        0.83
```

## Findings

1. **R2 qualifies.** At k ≤ 3.5 it matches-or-beats R1 on F0 on every plane
   under BOTH noise models, with strictly better residual-coherent (coh_left
   ~0.4–0.7 vs R1's 0.7–1.6) and better off-track noise.
2. **The white-noise gap resolves in R2's favor — decisively.** Under the FM
   corpus's actual (white) noise model — never measured before — R1 *degrades
   badly on compression*: its kept counts blow up to 1.30×/1.44×/1.10× the
   oracle (U/V/Y), i.e. R1 leaks coherent power into kept coefficients, while
   R2@k4 stays at the oracle (0.996/0.999/0.998). The record's colored-noise
   conclusions transfer, and strengthen.
3. **Oracle-equality reproduces** under both spectra: R2@k4 kept counts within
   ±2% of the never-had-coherent ceiling (the RESULTS.md §6c claim, now
   re-established on this corpus and on white noise).
4. **kgate: the adopted 4.0 is NOT the right default.** At k4, U's F0 falls
   below R1 (win rate 0.25–0.29) — U is the F0-limited plane and k4 over-gates
   it, exactly as the record's own sweeps warned. k3 maximizes F0 everywhere
   (all win rates ≥ 0.79) at +6–15% coefficients over oracle; **k3.5 is the
   sweet spot** (F0 ≥ R1 everywhere, kept = oracle +1–3%); U at k3.5 is a
   statistical tie with R1 (win 0.46, ΔF0 +0.0006) — n≈100+ needed to resolve,
   or adopt k3 if F0 is prioritized over the last few % of compression.
5. **Cross-implementation parity** (canonical torch C vs verbatim-numpy A,
   k4, all 144 cells): max |ΔF0| = 0.0016, max kept-count difference = 0.23%.
   The known `sigc` median-interpolation divergence is immaterial; either
   bit-parity choice for the packaged version is scientifically safe.
6. **Safety without coherent noise** (first-ever measurement): on pure clean
   input the gate is a near-exact no-op (max |Δ| ≤ 0.0034 ADC — σ_c clamps at
   1e-6 and the gate passes nothing). On clean+intrinsic (no coherent) it
   subtracts a small benign common-mode: mean F0 cost **+0.0014** vs not
   running, off-track RMS *improves* slightly. Always-on default is safe;
   "if-needed" gating buys ~a milli of F0 at most.

## Recommendations

- **Promote R2 as the default removal** in the packaged front-end
  (`removal='gate'`), R1 retained as `'multipass'`.
- **Default `kgate=3.5`** (single value, both spectra), NOT the FM-adopted 4.0;
  document k3 as the F0-max option and k4 as the count-optimal option. NOTE:
  the FM cache was built at k4 — changing the default affects any regenerated
  corpus (fold into the corpus-rebuild decision; do not mix k within a corpus).
- Package with A-parity statistics (`quantile(0.5)` throughout, fixing the
  accidental `sigc` lower-middle median); re-capture goldens at that boundary.
- Confirm U@k3.5 vs R1 at n ≥ 100 events (cheap: ~minutes GPU) before freezing
  the number in DetectorConfig; add the (backend × removal) validity matrix and
  the faithful-injector test fixtures per the blast-radius report.

Raw rows: `results.jsonl` (this dir). Harness: `qual.py`.

---

## RERUN (n=129, density-stratified, GPU) — 2026-07-23, supersedes the kgate call above

The n=12 pass above was flagged by the completeness audit: the U@k3.5 "tie" was a
regime average of an arbitrary ev0-11 sample, SEs were seed-clustered, and the
RNG was non-reproducible. `qual3.py` re-runs with: density-STRATIFIED events
(scan 200, sample evenly across U-activity 775..967,838 nonzero px + the densest
20% = the parallel-track failure mode), **n=129 independent events** (1 seed
each), reproducible seeds, both noise models. GPU-optimized (~50x): noise via
`dense_ops` torch, R1 via helix jax backend (bit-matches numpy, 9.5e-7), threshold
batched on GPU. Faithfulness: GPU noise is the statistical-parity port (same
model, different RNG than the numpy pass) — a Monte-Carlo over 129 realizations,
not bit-comparable to the n=12 numbers; the paired design is preserved.

Win-rate vs R1 (frac. events R2 beats multipass on removal F0), all / dense (24):

```
             k3          k3.5        k4          kept/oracle (k3/k3.5/k4)
U colored   0.73/0.54   0.39/0.04   0.19/0.00   1.13 / 1.02 / 0.98
U white     0.74/0.46   0.50/0.08   0.28/0.00   1.09 / 1.02 / 1.00
V colored   0.87/0.67   0.81/0.54   0.68/0.42   1.17 / 1.04 / 1.00
V white     0.95/0.83   0.88/0.62   0.78/0.50   1.12 / 1.03 / 1.00
Y colored   0.99/1.00   0.95/0.83   0.90/0.67   1.11 / 1.03 / 1.01
Y white     1.00/1.00   0.99/0.96   0.97/0.88   1.07 / 1.02 / 1.00
```

**VERDICT: default kgate = 3.0**, not 3.5 (and not the FM-adopted 4.0). k3 is the
ONLY value where R2 >= R1 on every plane, both noise models, AND the dense-track
failure mode. Properly stratified, k3.5 *loses* to R1 on U (0.39/0.50 overall,
0.04/0.08 dense) — the reverse of the n=12 hint; k4 loses badly on dense U (0.00).
On the hardest dense-U events k3 F0 (0.8922) ties R1 (0.8923) while k3.5/k4 fall
behind. Cost: k3 keeps +11-17% coeffs over the oracle ceiling (k3.5 ~= oracle,
k4 ~= oracle) — a fidelity-first tradeoff, correct for a representation front-end.

Also settled: **gate_soft is decisively worse** on every plane (n=6/n=8 smoke,
qual2.py: U 0.839, V 0.869, Y 0.936) — catalog gap closed negatively.
**de2_clamp does not beat R1 on U** (informational arm, qual2.py) — confirms the
record's §6n disposition (opt-in only). Neither changes the default.

Document as: removal default = gate (R2); kgate=3.0 (fidelity/win-rate optimal,
qualified n=129 stratified); k3.5 = oracle-compression option; k4 = current FM
corpus value (changing it is a corpus-rebuild decision — never mix k in a corpus).

Rerun rows: `results3.jsonl`. Harness: `qual3.py` (decisive), `qual2.py` (+de2/soft).

---

## MULTIPASS (2026-07-23, multipass.py) — corrects "single-pass" and the record's "no-op"

Question: R1 iterates 3x; the smart gate is single-pass. Does multipass help it?
The record (§8) said iterating smart is "bit-identical" — but that was F0-only.
Multipass smart (R1-analog in coeff space: detect signal on the CLEANED bands ->
accumulate signal mask -> re-estimate block common-mode from ORIGINAL bands
excluding signal wires -> re-gate). 40 stratified events, k=3:

  mean |signal_lost(1p)-signal_lost(2p)| = 0.00067   (F0: TRUE no-op, as recorded)

But on the axes the record didn't check, 2-pass is strictly better:
```
pl arm       sig_lost%  coh_left  stripe   coeffs   (oracle)
U  smart_1p    11.801    0.690    0.415    69216    1.11x
U  smart_2p    11.793    0.643    0.324    62281    1.00x   (62110)
V  smart_1p    11.233    0.558    0.420    58689    1.16x
V  smart_2p    11.191    0.495    0.330    51777    1.02x   (50721)
Y  smart_1p     4.474    0.604    0.428    54753    1.11x
Y  smart_2p     4.454    0.544    0.333    50473    1.02x   (49549)
```
- coeffs drop ~10% to ~oracle; stripe ~-22%; coh_left ~-7%; F0 & ontrack unchanged.
- 3p == 2p (converged). Mechanism: purer coherent estimate (signal excluded via
  cleaned-band detection; coherent is rank-1 so few clean wires suffice) removes
  more coherent -> fewer noise coeffs survive threshold -> oracle compression,
  at NO signal cost (the gate already protects large=signal common-modes).

VERDICT UPDATE: default = smart gate, kgate=3, **2 passes**. This resolves the
1-pass k3-vs-k3.5 tradeoff — 2-pass k3 gives k3's signal preservation AND
oracle-level compression simultaneously. The packaged gate should take npass
(default 2). The record's "multipass no-op" holds only for F0; on the
compression that feeds the model it is a real, free gain.

---

## (k1,k2) 2-pass GRID (2026-07-23, grid.py) — 5x5, 100 stratified events

Different kgate for pass 1 vs pass 2. Full 5x5 in {2.5,3,3.5,4,4.5}. Confirms the
two passes DECOUPLE: pass-1 k1 protects signal (its cleaned bands seed the
signal detection); pass-2 k2 does coherent cleanup + compression on the purified
(signal-excluded) estimate. So signal_lost tracks mainly k1 (low k1 = best),
coeffs/oracle & stripe track mainly k2 (higher k2 = more removed, tighter
compression). Best operating points at coeffs<=1.02:
  U: (k1=2.5, k2=3.0)  V: (k1=2.5, k2=3.5)  Y: (k1=2.5, k2=3.5)

Key cells (sig_lost% / coeffs-oracle / stripe, avg over U,V,Y):
```
  k1=3.0 k2=3.0 (2-pass k3)   8.890 / 1.017 / 0.335
  k1=2.5 k2=3.0 (fidelity)    8.804 / 1.021 / 0.361
  k1=2.5 k2=3.5               8.988 / 0.998 / 0.222
  k1=3.0 k2=3.5 (oracle-comp) 9.063 / 0.995 / 0.187
```
- Asymmetric (k2>k1) beats the diagonal: **(k1=2.5, k2=3.5)** gives ~diagonal-k3
  fidelity (8.99% vs 8.89%) at TRUE oracle compression (0.998 vs 1.017) and much
  better coherent removal (stripe 0.222 vs 0.335). **(k1=2.5, k2=3.0)** is the
  pure-fidelity corner (8.80%) at ~1.02 coeffs.
- Recommended default: **k1=2.5, k2=3.5, 2 passes** — fidelity ≈ single/2-pass
  k3, compression at oracle, coherent removal best of the fidelity-preserving
  cells. (Or k1=2.5,k2=3.0 if a hair more fidelity is worth ~2% more coeffs.)
  Grid is flat around k1∈{2.5,3}, k2∈{3,3.5} — all beat single-pass and R1.
Heatmaps: grid_heatmap.png (3 planes x 3 metrics). Rows grid.jsonl.
