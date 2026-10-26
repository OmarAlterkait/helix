# Optical: one codec for every light system

`helix.optical` compresses optical waveforms — SBND PMT and X-ARAPUCA data, and goop's simulated
PMTs — with one method. A light system is described by its measured single-photon response;
everything else is derived from it.

## Method (`helix.optical.codec`)

Per event and system:

1. **Noise and pedestal per channel, from where there is no light.** The matched-filter
   statistic (the window minus a local baseline, correlated with the zero-mean unit-norm core of
   the response — its shortest span around the peak holding 98% of its energy) marks light at
   significance `k`; the noise of the statistic, of every Haar level, and the window's clipped
   mean are measured outside it (widened by the core span) and medianed over the channel's
   windows. A window median of integer codes is off by up to half a count, and light below the
   trigger lifts a lit window's own estimate — either becomes fake or lost light.
2. **The light.** Photon amplitudes `a >= 0` minimising `|y - h*a|^2 / 2 + lambda sum(a)` (FISTA,
   then refitted without the penalty on its support), `y` the window minus its channel's
   pedestal, `lambda = k x statistic noise x |h|` (a photon enters where its matched-filter
   significance exceeds `k`). The light `h*a` covers isolated photons, pile-up and slow dense
   light alike.
3. **Stored.** The integer Haar transform of the RAW ADC samples: coarse sums always; a difference
   where the light's difference reaches `q sigma_j` (`q = 1`: keep exactly where the signal is
   worth a noise sigma — the error-optimal rule applied to the estimate; nothing is chosen by its
   own noisy value), or where the data depart from the light model by `5 sigma_j` (a wrong
   response on big pulses, artefacts that are not light; a noise coefficient passes with
   probability ~6e-7).

Decoding never uses the light estimate: it is the exact inverse Haar transform of the stored raw
coarse sums and differences; a missing difference splits its block's sum evenly.

Settings: `OpticalSpec(response, polarity, k=3, level, q=1, residual=5)`. SBND: `level` 6 (PMT 2 ns,
X-ARAPUCA 16 ns ticks); goop: 7 (1 ns). Batching groups windows within 25% padding (goop: 0.67 s
per event on a 2080 Ti).

## Responses (`helix.optical.calibrate.fit_response`)

Measured the way the codec uses them: photons from the codec's own deconvolution at significance
5; the core (±40-80 ns) by regression on isolated single photons, each against its own local
level, with equal weight for every arrival time mod 8 samples (cancels noise locked to the
sample clock); the tail by least squares over all photons with the core fixed; the tail as an
exponential model if it describes it (goop: `-0.015 exp(-t / 632 ns)`), else smoothed in log time
(SBND PMT: ringing at 16-80 ns, 20-80 standard errors, and the AC undershoot to ~600 ns).
A stack of matched-filter maxima — the earlier estimate — selects noise peaks in quiet windows
and returns the noise's autocorrelation (digitizer clock lines). Recovery of a known response
injected on noise (rms error, units of the peak): from the true response goop core 0.006, tail
0.0004, SBND PMT 0.013 / 0.004; from a 30% wider start, one pass, goop 0.017 / 0.007, PMT
0.021 / 0.004 (the start's error enters through the photons it deconvolves; passes reduce it).
On data, late light correlated with each photon (argon's slow component) trades against a longer
tail: one pass from the previous response. The X-ARAPUCA response (16 ns, short self-triggered
windows, few isolated photons) is not recovered reliably and stays the mean isolated pulse.

## Validation (`helix.optical.validate`)

Known light — single photons, and 100 / 500-photon flashes (30% 6 ns, 70% 1.5 us) — injected on
real light-free windows (SBND) or on goop's exact noise model; error = signal dropped + noise
kept, in a fixed reference sigma_j measured on the noise before injection; the oracle keeps
exactly the true differences above sigma_j.

Previous support rule vs this codec (SBND-data-helix `archive/scripts/rule_truth2.py`, region scoring):
PMT per photon 13.0 values at 25.0 sigma^2 vs 6.4 at 11.5 (oracle 6.2 / 9.1); per flash 698 at 1310
vs 484 at 827 (oracle 495 / 645); X-ARAPUCA per photon 27.5 at 58.6 vs 9.2 at 17.3. goop
(SBND-data-helix `archive/scripts/goop_compare.py`, exact truth): whole-chunk error 3.5k sigma^2 at ~1950 values vs >= 5.7k
for coif3 at any threshold; late single photons 450 vs >= 1230.

## Files

* Optical shards (`helix.optical.shard`): one modality per file next to the TPC coefficient
  shards, joined on `(run, source_file, event)`; exact integer streams; `read_optical`.
* Tokens (`helix.optical.tokens`): 128-sample tree blocks with a stored difference + coarse
  tokens (128 slots, like a TPC token); `shard_tokens` decodes and tokenizes an event's windows
  together (`shard.read_event`; one SBND event on a CPU core: PMT ~0.3 s, X-ARAPUCA ~0.08 s).
  Not yet fed to the model.

## Running

Optical shards are written by the build pipeline (`helix/build.py`), next to the TPC shards:

* SBND: `scripts/build_sbnd.py --input RUN/*.h5 --output OUT/RUN/sbnd --device cuda` —
  TPC `sbnd_coeff_NNNN.h5` + optical `sbnd_optical_NNNN.h5` (`--no-optical`: TPC only).
* goop: `scripts/build_goop.py --input light_output.h5 --output OUT/RUN/goop --device cuda` —
  `goop_optical_NNNN.h5`, with goop's photon counts per PMT in `truth/pe_counts`.
* A whole SBND corpus: `scripts/corpus.sh all sbnd OUT RUN_DIR...` (`docs/RUNBOOK.md` §3).
  `corpus.sh` calibrates a TPC `norm_sigma` before building, so it does not serve an
  optical-only source (goop) as is.

Per-system overrides of the codec settings: `--optical-config` (json / yaml).
