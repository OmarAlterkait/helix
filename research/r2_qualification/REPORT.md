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
