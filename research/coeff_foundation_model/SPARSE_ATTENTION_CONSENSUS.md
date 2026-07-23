# Sparse-attention design — 6-agent consensus (2026-07-14)

Six independent Fable agents (2 panels × {conservative, moderate, aggressive}) designed the geometry-gated
sparse attention. They CONVERGED on one invariant core; only the "aggressiveness" of squeezing the
non-cross-plane tiers varied. The two aggressive agents independently MEASURED real events and independently
killed the same aggressive cuts. Supersedes/updates ATTENTION_CONNECTIVITY.md.

## INVARIANT CORE (unanimous, all 6, all aggressiveness levels)
1. **Cross-plane epipolar tier is DIRECT token<->token (1-hop), NEVER routed through a hub/register.**
   This is THE decision (named #1 by all 6). It is the 3D-physics path; hubbing it recreates the measured
   Perceiver ~2x failure.
2. **Gate cross-plane by drift-time SUPPORT-OVERLAP** (key/query time-supports intersect, dilated +-5 ticks),
   NOT a fixed center-distance +-dt (clips coarse bands: A4 spans 128 ticks, partner center ~85 ticks off)
   and NOT by wire and NOT by nearest-K-in-time (misses the latent-wire correspondent). Knob-free, from tokenizer.
3. **Bands pooled in physical (wire, t_phys)** -> within-plane window covers cross-band siblings 1-hop; no
   separate band mechanism.
4. **Register spine (per-plane + global) = island-freedom by CONSTRUCTION (occupancy-independent theorem),
   NEVER load-bearing.** Hubs carry only lowest-info residual (long-range within-plane, cross-volume).
5. **Cross-volume: no direct edges** (other drift volume = different charge); route via global registers.
6. **RoPE split:** wire RoPE within-plane (required; wire_rope=0 freezes recon), TIME-ONLY cross-plane
   (wire non-metric across planes). Tiering enables both -> the wire-RoPE fix.
7. **All gates in PHYSICAL units** (wires, drift ticks) -> generalizes across detectors/occupancy. NO kNN,
   no learned gating, no occupancy-dependent thresholds.
Two principled wins to adopt at any level: **same-volume-only cross-plane** (~2.5x, physics-free) and
**global-register read-back** (diameter 4->3).

## AGGRESSIVENESS KNOB (the ONLY thing that varied) + where it saturates
| level | within-plane | cross-plane | sparsity | diameter |
| conservative | full dense plane-clique (no per-plane regs) | all planes/bands | ~5x | 2 |
| moderate | local window + per-plane+global regs | all planes/bands | ~15x | 4 |
| aggressive | tiny window + regs | same-VOLUME, restrict BANDS | ~70-200x | 3-4 |
Both aggressive agents: "partially collapses to moderate" -- skeleton is structurally forced, no tier
removable. **~70x safe** (support-overlap, all bands); **~200x needs the coarse-band-only GAMBLE** (fine-band
cross-plane -> 2-hop; validate on 3D probes; 2x fallback). ~1000x only via hubbing cross-plane = Perceiver trap.

## RECOMMENDATION (simplicity/generalization priority)
Build the invariant core; default the within-plane tier to CONSERVATIVE (full dense plane-clique: 2 knobs,
islands impossible by construction, diameter 2, strict subgraph of full attention -> WARM-STARTS from
existing checkpoints, ~5x). Escalate to moderate window+registers only when N grows (finer patches, D1 band,
multi-event). Aggressive band-restriction = probe-gated lever, not default.

## TIMING / near-term motivation
Full attention still affordable at N~31k (architecture-map agent: building sparse is premature for scaling
alone). BUT the validated 3D lever -- plane-masking -- leaves ~83% of tokens visible under full attention
(the measured ~6x slowdown). The sparse cross-plane scheme makes exactly that pretext cheap. So **sparse
attention + plane-mask COMPOUND** -- the concrete reason to build it. See [[coeff-fm-3d-probe-metric]],
[[coeff-fm-mup-lr]].
