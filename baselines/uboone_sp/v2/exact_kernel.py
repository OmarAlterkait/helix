#!/usr/bin/env python3
"""The EXACT effective response of our simulation for per-wire hits truth, built
from JAXTPC's own code at the production commit (1ea2282, exported read-only to
/pscratch/.../jaxtpc_1ea2282 with `git archive`).

    scripts/helix_run.sh python baselines/uboone_sp/v2/exact_kernel.py --out baselines/uboone_sp/response

Forward model reproduced (production config, see results/exact_kernel.md):
  sensor   q * attenuation * DKernel(s, w_offset, t_offset) placed by
           wires.accumulate_response_signals (kernels.load_response_kernels with
           num_s = 16, sigma maxima from geometry.calculate_max_diffusion_sigmas
           for a 216 cm max drift; s = sqrt(drift_cm / 216)); no electronics
           (kernel is ADC per electron incl. cold-electronics shaping); gain 1.
  hits     wires.prepare_deposit_with_diffusion: the SAME deposit's charge,
           Gaussian-CDF diffused over (2K_w+1) x (2K_t+1) integer wires x ticks
           with sigma(drift_time) -- i.e. post-diffusion charge per integer wire
           and tick (the 25 e- CSR threshold is a separate truncation, not modelled here).
Per-wire hits do not carry the sub-wire position or the drift, so the best
translation-invariant kernel mapping hits -> sensor is the least-squares solution
over the deposit population. It is computed here on synthetic point deposits run
through those two JAXTPC functions (uniform sub-wire offset and sub-tick offset):
  exact_<P>_s<k>.npz   for a fixed diffusion level s_k (drift = s_k^2 * 216 cm)
  exact_<P>.npz        drift-averaged: drift uniform in [0, 216] cm
  exact0_<P>.npz       s = 0 (no diffusion): equals the plain sub-wire / sub-tick
                       average of the JAXTPC kernel
Same layout as response/fit_<P>.npz (R_avg (n_t, n_w), wire_offsets, time_zero_bin).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
JAXTPC = "/pscratch/sd/o/oalter/helix_work/uboone_sp/jaxtpc_1ea2282"
S_GRID = np.linspace(0.0, 1.0, 9)
MAX_DRIFT, V, DT = 216.0, 0.16, 0.5            # cm, cm/us, us   (cubic_wireplane_config.yaml:8,68; readout 2 MHz)
D_T, D_L = 12.0e-6, 7.2e-6                     # cm^2/us          (cubic_wireplane_config.yaml:69-70)
PITCH = 0.3


def jaxtpc():
    sys.path.insert(0, JAXTPC)
    import jax
    jax.config.update("jax_platform_name", "cpu")
    from tools.kernels import load_response_kernels, interpolate_diffusion_kernel_batch
    from tools.wires import accumulate_response_signals, prepare_deposit_with_diffusion
    from tools.geometry import calculate_max_diffusion_sigmas
    return jax, load_response_kernels, interpolate_diffusion_kernel_batch, accumulate_response_signals, \
        prepare_deposit_with_diffusion, calculate_max_diffusion_sigmas


def synth(plane, s_sampler, n_img, rng, nw=360, nt=1600, n_dep=300):
    """Yield (hits image, sensor image) pairs of synthetic unit-charge point deposits."""
    jax, load_rk, interp, accum, prep, calc = jaxtpc()
    import jax.numpy as jnp
    _, _, sig_t, sig_l = calc(MAX_DRIFT, V, D_T, D_L, PITCH, DT)
    RK = load_rk(response_path=os.path.join(JAXTPC, "tools", "responses"), num_s=16, time_spacing=DT,
                 max_sigma_trans_unitless=sig_t, max_sigma_long_unitless=sig_l)[plane]
    K_w, K_t = max(1, int(np.ceil(3.0 * sig_t))), max(1, int(np.ceil(3.0 * sig_l)))
    prep_v = jax.vmap(prep, in_axes=(0,) * 11 + (None,) * 9)
    for _ in range(n_img):
        s = s_sampler(n_dep)
        drift_cm = s ** 2 * MAX_DRIFT
        drift_t = drift_cm / V
        w = rng.integers(30, nw - 30, n_dep)
        u = rng.uniform(-0.5, 0.5, n_dep)                  # closest_wire_dist / pitch
        tick = rng.uniform(250, nt - 250, n_dep) * DT      # readout tick (us)
        t_idx = np.floor(tick / DT).astype(np.int32)
        t_off = tick / DT - t_idx
        contrib = interp(RK.DKernel, jnp.asarray(s, jnp.float32), jnp.asarray(u, jnp.float32),
                         jnp.asarray(t_off, jnp.float32), RK.wire_spacing, RK.num_wires)
        S = accum(jnp.asarray(w, jnp.int32), jnp.asarray(t_idx, jnp.int32), jnp.ones(n_dep, jnp.float32), contrib,
                  nw, nt, RK.num_wires, RK.kernel_height, RK.wire_zero_bin, RK.time_zero_bin)
        one = jnp.ones(n_dep, jnp.float32)
        wi, ti, val = prep_v(one, jnp.asarray(drift_t, jnp.float32), jnp.asarray(tick, jnp.float32),
                             jnp.asarray(drift_cm, jnp.float32), jnp.asarray(w, jnp.int32),
                             jnp.asarray(u * PITCH, jnp.float32), one, one * 0.5, one * 0.5, one,
                             jnp.ones(n_dep, bool), K_w, K_t, PITCH, DT, D_L, D_T, V, nw, nt)
        H = np.zeros((nw, nt))
        np.add.at(H, (np.asarray(wi).ravel(), np.asarray(ti).ravel()), np.asarray(val).ravel())
        yield H, np.asarray(S, np.float64)


def fit(plane, s_sampler, n_img, seed):
    from sp.response import LSAccumulator
    rng = np.random.default_rng(seed)
    acc = LSAccumulator(10, 150, 60)
    for H, S in synth(plane, s_sampler, n_img, rng):
        acc.add(H, S)
    return acc.solve(ridge=1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-img", type=int, default=24)
    ap.add_argument("--planes", default="UVY")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = {}
    for P in a.planes:
        R, wo, tzb = fit(P, lambda n: np.sqrt(np.random.default_rng().uniform(0, 1, n)), a.n_img * 2, 1)
        np.savez(os.path.join(a.out, f"exact_{P}.npz"), R_avg=R, wire_offsets=wo, time_zero_bin=tzb,
                 note="drift-averaged exact effective kernel (drift uniform 0-216 cm), JAXTPC 1ea2282")
        for k, s in enumerate(S_GRID):
            Rk, _, _ = fit(P, lambda n, s=s: np.full(n, s), a.n_img, 100 + k)
            np.savez(os.path.join(a.out, f"exact_{P}_s{k}.npz"), R_avg=Rk, wire_offsets=wo, time_zero_bin=tzb, s=s)
            if k == 0:
                np.savez(os.path.join(a.out, f"exact0_{P}.npz"), R_avg=Rk, wire_offsets=wo, time_zero_bin=tzb)
        meta[P] = dict(sum=float(R.sum()), peak=float(np.abs(R).max()))
        print(P, meta[P], flush=True)
    json.dump(dict(s_grid=S_GRID.tolist(), meta=meta), open(os.path.join(a.out, "exact_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
