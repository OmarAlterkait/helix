# What the noise model does to each band

The corpus is built with the MEASURED (MicroBooNE, arXiv:1705.07341) incoherent
spectrum. The research cache defaulted to WHITE, because `star_tpc.py` called
`AddNoise(coherent=True, incoherent=True)` without naming a spectrum and
`series_spectrum` defaulted to `None`.

Everything downstream follows from that one change: the normalisation table, which
coefficients survive thresholding, the categorical bin grid, and m113 being
out-of-distribution on this corpus. This file records the measurement, not the
inference — reproduce with `scripts/compare_noise_bands.py`.

Band <-> frequency at 2 MHz sampling (Nyquist 1 MHz), DWT levels (4,4,3,2):

| band | | frequency |
|---|---|---|
| 0 | A4 | 0 – 62.5 kHz |
| 1 | D4 | 62.5 – 125 kHz |
| 2 | D3 | 125 – 250 kHz |
| 3 | D2 | 250 – 500 kHz |
| — | D1 | 500 kHz – 1 MHz — **dropped by the tokenizer** |

The measured spectrum is non-zero only over **10–490 kHz**, peaking near 100 kHz.
So relative to white at equal total power it concentrates noise into A4/D4/D3,
depletes D2, and puts **nothing at all** in D1.

## 1. Prediction, from the spectrum alone

| band | white power | colored power | √(col/white) |
|---|---|---|---|
| 0 A4 | 0.062 | 0.225 | 1.90 |
| 1 D4 | 0.062 | 0.279 | 2.11 |
| 2 D3 | 0.125 | 0.395 | 1.78 |
| 3 D2 | 0.250 | 0.101 | **0.64** |
| (D1) | 0.500 | 0.000 | 0.00 |

## 2. Measured — the same 6 events built both ways

`norm_sigma`, the MAD of the **noisy** coefficients, averaged over planes. This is
the quantity every coefficient is divided by:

| band | colored | white | col/white | predicted |
|---|---|---|---|---|
| 0 A4 | 3.1271 | 1.9266 | **1.62** | 1.90 |
| 1 D4 | 3.5345 | 1.9079 | **1.85** | 2.11 |
| 2 D3 | 2.8989 | 1.8568 | **1.56** | 1.78 |
| 3 D2 | 1.4164 | 1.8140 | **0.78** | 0.64 |

**Direction exact, magnitude compressed toward 1** — and the compression is
expected, because `series_spectrum` shapes only ONE of three contributions to
that MAD:

* the incoherent model is `ENC = sqrt(white_x^2 + (y + z*L)^2)`, and the
  `white_x = 0.90` term is flat whatever the series spectrum is;
* the **coherent** component (2.5 ADC rms, corner 20 kHz, slope 1.5) is
  unchanged between the two runs;
* signal contributes to the MAD too.

So a prediction made from the series spectrum alone must overshoot, and it does,
consistently across all four bands.

## 3. Consequences, also measured

Noise sets the threshold, so the surviving SUPPORT changes — not just the scale:

| band | n coeff colored | n white | ratio | tgt range colored | tgt range white |
|---|---|---|---|---|---|
| 0 A4 | 312,721 | 362,610 | 0.862 | [−7.08, +7.08] | [−7.50, +7.56] |
| 1 D4 | 218,409 | 282,565 | 0.773 | [−6.49, +6.46] | [−7.05, +7.00] |
| 2 D3 | 237,958 | 284,030 | 0.838 | [−6.01, +6.01] | [−6.41, +6.40] |
| 3 D2 | 167,919 | 148,567 | **1.130** | [−5.22, +5.14] | [−5.02, +4.92] |

More noise → higher threshold → fewer survivors. Bands 0–2 lose 14–23% of their
coefficients under colored noise; D2 **gains** 13%.

## 4. Why the bin grids differ

The categorical grid is 128 bins uniform in `tgt = arcsinh(clean / norm_sigma)`
over `[p0.05, p99.95]`. Both inputs moved, so the grid moved:

| band | m113 grid (white) | corpus grid (colored) | width | bin width |
|---|---|---|---|---|
| 0 A4 | [−7.28, +7.80] | [−6.85, +7.07] | −7.7% | 0.1178 → 0.1088 |
| 1 D4 | [−7.06, +7.02] | [−6.34, +6.38] | −9.6% | 0.1099 → 0.0994 |
| 2 D3 | [−6.40, +6.34] | [−5.89, +5.88] | −7.6% | 0.0995 → 0.0920 |
| 3 D2 | [−5.00, +4.90] | [−5.15, +5.07] | **+3.2%** | 0.0774 → 0.0798 |

**The white-noise tgt ranges measured in section 3 reproduce m113's grid**
([−7.05,+7.00] vs [−7.06,+7.02] for D4; [−6.41,+6.40] vs [−6.40,+6.34] for D3;
[−5.02,+4.92] vs [−5.00,+4.90] for D2). That is the closing link: m113's bin
edges are a white-noise artefact, and rebuilding them from white noise recovers
them.

Fit against the corpus, on HELD-OUT events — fraction landing in the ±inf
catch-all bins, design target ~0.1%:

| band | m113 edges | corpus edges |
|---|---|---|
| 0 | 0.016% | 0.145% |
| 1 | 0.010% | 0.124% |
| 2 | 0.022% | 0.138% |
| 3 | **0.218%** | 0.126% |

m113's grid is too WIDE in bands 0–2 (outer bins under-used, resolution wasted)
and too NARROW in D2 (0.2% of the mass collapsed into a single catch-all bin).

## 5. What follows

* **m113 is out-of-distribution on this corpus** — not a pipeline defect, a
  consequence of fixing the noise model. It remains a valid architecture and
  checkpoint-loading reference; it is not a performance reference.
* **Bins must be derived per corpus** (`scripts/derive_coeff_bins.py`). They are
  training-set statistics; a model cannot invent them, and inheriting them across
  a noise-model change silently mis-sizes the head.
* **`mae_ddp`'s recorded metrics cannot serve as a physics A/B** for a model
  trained here. Loop mechanics are pinned instead, bit-exactly, by
  `tests/test_training_parity.py`.
