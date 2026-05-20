"""Run HELIX pipeline on 200 events × 6 planes × 5 noise seeds.

Noise generation, digitization, and metrics all on GPU via JAX.
Coherent noise waveforms precomputed per-event (NumPy, small: ~31 groups).
"""

import sys, os, time
import numpy as np

sys.path.insert(0, '/home/oalterka/desktop_linux/pimm-data/src')
sys.path.insert(0, '/home/oalterka/desktop_linux/JAXTPC')

import helix._backend as _b
_b._backend = 'jax'
import jax
import jax.numpy as jnp
from functools import partial

from pimm_data.jaxtpc import JAXTPCDataset
from tools.coherent_noise import generate_group_waveforms, broadcast_to_wires
from helix.config import DetectorConfig
from helix.coherent import remove_coherent
from helix.wavelet import sparsify, reconstruct

print(f'JAX devices: {jax.devices()}')

DATA_ROOT = '/home/oalterka/desktop_linux/JAXTPC/sample_edepsim/test_00_00_01/run_0026628546/output_200'
OUT_DIR = '/home/oalterka/desktop_linux/helix/figures'
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = 5
SEED_BASE = 1000
N_TICKS = 4321

PLANE_INFO = {
    'volume_0_U': {'n_wires': 1969, 'pedestal': 1843, 'wire_len': (0.42, 4.63), 'plane_type': 'U'},
    'volume_0_V': {'n_wires': 1969, 'pedestal': 1843, 'wire_len': (0.42, 4.63), 'plane_type': 'V'},
    'volume_0_Y': {'n_wires': 1443, 'pedestal': 410,  'wire_len': (2.33, 2.33), 'plane_type': 'Y'},
    'volume_1_U': {'n_wires': 1969, 'pedestal': 1843, 'wire_len': (0.42, 4.63), 'plane_type': 'U'},
    'volume_1_V': {'n_wires': 1969, 'pedestal': 1843, 'wire_len': (0.42, 4.63), 'plane_type': 'V'},
    'volume_1_Y': {'n_wires': 1443, 'pedestal': 410,  'wire_len': (2.33, 2.33), 'plane_type': 'Y'},
}

config = DetectorConfig(group_size=64, mask_threshold_nsigma=3.0, num_time_steps=N_TICKS,
                        temporal_dilation_ticks=11, n_passes=3)

SIGMA_CACHE = {}
for pk, info in PLANE_INFO.items():
    nw = info['n_wires']
    if nw not in SIGMA_CACHE:
        wl = np.linspace(info['wire_len'][0], info['wire_len'][1], nw)
        SIGMA_CACHE[nw] = jnp.array(np.sqrt(0.90**2 + (0.79 + 0.22 * wl)**2).astype(np.float32))


@partial(jax.jit, static_argnums=(1, 2))
def add_noise_jax(clean_plus_coh, nw, ped, sigma_w, key):
    intrinsic = sigma_w[:, None] * jax.random.normal(key, (nw, N_TICKS), dtype=jnp.float32)
    noisy = jnp.round(clean_plus_coh + intrinsic + ped).clip(0, 4095) - ped
    return noisy


@partial(jax.jit, static_argnums=())
def compute_metrics_jax(clean, recon, cleaned, coh_rms):
    sig = jnp.abs(clean) > 0
    n_sig = jnp.sum(sig).astype(jnp.float32)
    total_charge = jnp.sum(jnp.abs(clean) * sig)

    err = (recon - clean) * sig
    f0 = 1.0 - jnp.sum(jnp.abs(err)) / jnp.maximum(total_charge, 1.0)
    signal_bias = jnp.sum(err) / jnp.maximum(n_sig, 1.0)
    signal_rms = jnp.sqrt(jnp.sum(err**2) / jnp.maximum(n_sig, 1.0))

    noise_pix = ~sig
    n_noise = jnp.sum(noise_pix).astype(jnp.float32)
    noise_err = (recon - clean) * noise_pix
    noise_rms = jnp.sqrt(jnp.sum(noise_err**2) / jnp.maximum(n_noise, 1.0))

    coh_err = (cleaned - clean) * noise_pix
    coh_after = jnp.sqrt(jnp.sum(coh_err**2) / jnp.maximum(n_noise, 1.0))
    coh_rejection = coh_rms / jnp.maximum(coh_after, 1e-10)

    return f0, signal_bias, signal_rms, noise_rms, coh_rejection, n_sig, total_charge


def main():
    ds = JAXTPCDataset(data_root=DATA_ROOT, split='all', modalities=('sensor',))
    n_events = len(ds)
    plane_keys = list(PLANE_INFO.keys())
    n_planes = len(plane_keys)

    metric_names = ['f0', 'signal_bias', 'signal_rms', 'noise_rms', 'coh_rejection',
                    'sparsity', 'n_signal', 'total_charge', 'n_kept', 'n_total']
    results = {m: np.full((n_events, n_planes, N_SEEDS), np.nan, dtype=np.float64)
               for m in metric_names}

    total_runs = n_events * n_planes * N_SEEDS
    print(f'Running {n_events} events x {n_planes} planes x {N_SEEDS} seeds = {total_runs:,} runs\n')

    # Warmup
    print('Warming up JIT...', flush=True)
    for nw in [1969, 1443]:
        d = jnp.zeros((nw, N_TICKS), dtype=jnp.float32)
        sw = SIGMA_CACHE[nw]
        ped = 1843 if nw == 1969 else 410
        n = add_noise_jax(d, nw, ped, sw, jax.random.PRNGKey(0)); jax.block_until_ready(n)
        c = remove_coherent(n, config, sigma_per_wire=sw); jax.block_until_ready(c)
        sr = sparsify(c, config)
        r = reconstruct(sr, config, N_TICKS); jax.block_until_ready(r)
        _ = compute_metrics_jax(d, r, c, jnp.float32(2.5)); jax.block_until_ready(_[0])
    print('JIT warm.\n', flush=True)

    t0 = time.perf_counter()
    for evt_idx in range(n_events):
        t_evt = time.perf_counter()
        sample = ds[evt_idx]
        raw = sample['sensor']['raw']

        for p_idx, pk in enumerate(plane_keys):
            if pk not in raw:
                continue
            info = PLANE_INFO[pk]
            nw = info['n_wires']
            ped = info['pedestal']
            sigma_w = SIGMA_CACHE[nw]

            sp = raw[pk]
            clean_np = np.zeros((nw, N_TICKS), dtype=np.float32)
            clean_np[sp['wire'], sp['time']] = sp['value']
            clean = jnp.array(clean_np)

            # Precompute coherent noise waveforms for all seeds (NumPy, ~31 groups, fast)
            coh_all = []
            for seed_idx in range(N_SEEDS):
                seed = SEED_BASE + evt_idx * 100 + seed_idx
                ng = (nw + 63) // 64
                wf = generate_group_waveforms(ng, N_TICKS, beta=0.15, rms_adc=2.5,
                                               rng=np.random.default_rng(seed + 999))
                coh_all.append(jnp.array(broadcast_to_wires(wf, nw, 64)))

            for seed_idx in range(N_SEEDS):
                seed = SEED_BASE + evt_idx * 100 + seed_idx
                key = jax.random.PRNGKey(seed)

                noisy = add_noise_jax(clean + coh_all[seed_idx], nw, ped, sigma_w, key)
                cleaned = remove_coherent(noisy, config, sigma_per_wire=sigma_w)
                sr = sparsify(cleaned, config)
                recon = reconstruct(sr, config, N_TICKS)

                f0, bias, srms, nrms, crej, nsig, tcharge = compute_metrics_jax(
                    clean, recon, cleaned, jnp.float32(2.5))

                results['f0'][evt_idx, p_idx, seed_idx] = float(f0)
                results['signal_bias'][evt_idx, p_idx, seed_idx] = float(bias)
                results['signal_rms'][evt_idx, p_idx, seed_idx] = float(srms)
                results['noise_rms'][evt_idx, p_idx, seed_idx] = float(nrms)
                results['coh_rejection'][evt_idx, p_idx, seed_idx] = float(crej)
                results['sparsity'][evt_idx, p_idx, seed_idx] = sr.sparsity
                results['n_signal'][evt_idx, p_idx, seed_idx] = float(nsig)
                results['total_charge'][evt_idx, p_idx, seed_idx] = float(tcharge)
                results['n_kept'][evt_idx, p_idx, seed_idx] = sr.n_kept
                results['n_total'][evt_idx, p_idx, seed_idx] = sr.n_total

        elapsed = time.perf_counter() - t_evt
        total = time.perf_counter() - t0
        eta = total / (evt_idx + 1) * (n_events - evt_idx - 1)
        print(f'  event {evt_idx:>3d}/{n_events}  {elapsed:.1f}s  '
              f'(elapsed {total:.0f}s, eta {eta:.0f}s)', flush=True)

    out_path = os.path.join(OUT_DIR, 'metrics_200evt_5seed.npz')
    np.savez(out_path,
             plane_keys=np.array(plane_keys),
             plane_types=np.array([PLANE_INFO[pk]['plane_type'] for pk in plane_keys]),
             n_events=n_events, n_planes=n_planes, n_seeds=N_SEEDS,
             **results)
    print(f'\nSaved to {out_path}')
    print(f'Total: {time.perf_counter() - t0:.0f}s ({total_runs / (time.perf_counter() - t0):.1f} runs/s)')


if __name__ == '__main__':
    main()
