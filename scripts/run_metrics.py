"""Run HELIX pipeline on 200 events x 6 planes x 5 noise seeds.

Noise model matches JAXTPC (FFT-shaped series + white).
Intrinsic noise on GPU (JAX), coherent noise on CPU (NumPy).

Three modes: coherent_only, wavelet_only, full pipeline.
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

# -- Noise model (matches JAXTPC tools/noise.py) ----------------------

NOISE_X = 0.90
NOISE_Y = 0.79
NOISE_Z = 0.22

_noise_npz = np.load(os.path.join(os.path.dirname(__file__),
                                   '../../JAXTPC/config/noise_spectrum.npz'))

def get_noise_spectrum(n_ticks, sampling_rate_hz=2e6):
    n_freq = n_ticks // 2 + 1
    freqs = np.fft.rfftfreq(n_ticks, d=1.0 / sampling_rate_hz)
    spectrum = np.interp(freqs, _noise_npz['spectrum_freqs_hz'],
                         _noise_npz['spectrum_shape'])
    energy = np.sum(spectrum**2)
    if energy > 0:
        spectrum = spectrum / np.sqrt(energy) * np.sqrt(n_freq)
    return spectrum.astype(np.float32)

SPECTRUM_JAX = jnp.array(get_noise_spectrum(N_TICKS))

SERIES_RMS_CACHE = {}
SIGMA_CACHE = {}
for pk, info in PLANE_INFO.items():
    nw = info['n_wires']
    if nw not in SIGMA_CACHE:
        wl = np.linspace(info['wire_len'][0], info['wire_len'][1], nw)
        SERIES_RMS_CACHE[nw] = jnp.array((NOISE_Y + NOISE_Z * wl).astype(np.float32))
        SIGMA_CACHE[nw] = jnp.array(np.sqrt(NOISE_X**2 + (NOISE_Y + NOISE_Z * wl)**2).astype(np.float32))


@partial(jax.jit, static_argnums=(0, 1))
def generate_intrinsic_jax(nw, n_ticks, series_rms, key):
    """JAXTPC-matching intrinsic noise on GPU: FFT-shaped series + flat white."""
    n_freq = n_ticks // 2 + 1
    k1, k2, k3 = jax.random.split(key, 3)
    r = jax.random.normal(k1, (nw, n_freq)) * SPECTRUM_JAX
    i = jax.random.normal(k2, (nw, n_freq)) * SPECTRUM_JAX
    cpx = r + 1j * i
    cpx = cpx.at[:, 0].set(cpx[:, 0].real)
    shaped = jnp.fft.irfft(cpx, n=n_ticks, axis=1)
    cur_rms = jnp.maximum(jnp.std(shaped, axis=1, keepdims=True), 1e-10)
    shaped = shaped / cur_rms * series_rms[:, None]
    white = jax.random.normal(k3, (nw, n_ticks), dtype=jnp.float32) * NOISE_X
    return shaped + white


@partial(jax.jit, static_argnums=())
def digitize_jax(signal, ped):
    """Add pedestal, quantize, clip to 12-bit ADC, subtract pedestal."""
    return jnp.round(signal + ped).clip(0, 4095) - ped


@partial(jax.jit, static_argnums=())
def compute_metrics_jax(clean, recon, cleaned, coh_rms):
    sig = jnp.abs(clean) > 0
    n_sig = jnp.maximum(jnp.sum(sig).astype(jnp.float32), 1.0)
    total_charge = jnp.sum(jnp.abs(clean) * sig)

    err = (recon - clean) * sig
    f0 = 1.0 - jnp.sum(jnp.abs(err)) / jnp.maximum(total_charge, 1.0)
    signal_bias = jnp.sum(err) / n_sig
    signal_rms = jnp.sqrt(jnp.sum(err**2) / n_sig)

    noise_pix = ~sig
    n_noise = jnp.maximum(jnp.sum(noise_pix).astype(jnp.float32), 1.0)
    noise_err = (recon - clean) * noise_pix
    noise_rms = jnp.sqrt(jnp.sum(noise_err**2) / n_noise)

    coh_err = (cleaned - clean) * noise_pix
    coh_after = jnp.sqrt(jnp.sum(coh_err**2) / n_noise)
    coh_rejection = coh_rms / jnp.maximum(coh_after, 1e-10)

    return f0, signal_bias, signal_rms, noise_rms, coh_rejection, n_sig, total_charge


def main():
    ds = JAXTPCDataset(data_root=DATA_ROOT, split='all', modalities=('sensor',))
    n_events = len(ds)
    plane_keys = list(PLANE_INFO.keys())
    n_planes = len(plane_keys)

    MODES = ['coherent_only', 'wavelet_only', 'full']
    metric_names = ['f0', 'signal_bias', 'signal_rms', 'noise_rms', 'coh_rejection',
                    'sparsity', 'n_signal', 'total_charge', 'n_kept', 'n_total']
    results = {mode: {m: np.full((n_events, n_planes, N_SEEDS), np.nan, dtype=np.float64)
                      for m in metric_names}
               for mode in MODES}

    total_runs = n_events * n_planes * N_SEEDS
    print(f'Running {n_events} events x {n_planes} planes x {N_SEEDS} seeds x 3 modes\n')

    # Warmup
    print('Warming up JIT...', flush=True)
    for nw in [1969, 1443]:
        d = jnp.zeros((nw, N_TICKS), dtype=jnp.float32)
        sw = SIGMA_CACHE[nw]
        sr = SERIES_RMS_CACHE[nw]
        intr = generate_intrinsic_jax(nw, N_TICKS, sr, jax.random.PRNGKey(0))
        jax.block_until_ready(intr)
        n = digitize_jax(d + intr, jnp.float32(1843))
        jax.block_until_ready(n)
        c = remove_coherent(n, config, sigma_per_wire=sw); jax.block_until_ready(c)
        sp = sparsify(c, config)
        r = reconstruct(sp, config, N_TICKS); jax.block_until_ready(r)
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
            series_rms = SERIES_RMS_CACHE[nw]

            sp = raw[pk]
            clean_np = np.zeros((nw, N_TICKS), dtype=np.float32)
            clean_np[sp['wire'], sp['time']] = sp['value']
            clean = jnp.array(clean_np)

            for seed_idx in range(N_SEEDS):
                seed = SEED_BASE + evt_idx * 100 + seed_idx

                # Coherent noise (NumPy, ~18ms)
                ng = (nw + 63) // 64
                wf = generate_group_waveforms(ng, N_TICKS, beta=0.15, rms_adc=2.5,
                                               rng=np.random.default_rng(seed + 999))
                coh = jnp.array(broadcast_to_wires(wf, nw, 64))

                # Intrinsic noise (JAX/GPU, ~4ms)
                intrinsic = generate_intrinsic_jax(nw, N_TICKS, series_rms,
                                                    jax.random.PRNGKey(seed))

                # Digitize (JAX/GPU, ~2ms)
                noisy = digitize_jax(clean + coh + intrinsic, jnp.float32(ped))

                # -- Mode 1: coherent removal only --
                cleaned = remove_coherent(noisy, config, sigma_per_wire=sigma_w)
                f0, bias, srms, nrms, crej, nsig, tcharge = compute_metrics_jax(
                    clean, cleaned, cleaned, jnp.float32(2.5))
                for k, v in zip(metric_names[:7],
                                [f0, bias, srms, nrms, crej, nsig, tcharge]):
                    results['coherent_only'][k][evt_idx, p_idx, seed_idx] = float(v)
                results['coherent_only']['sparsity'][evt_idx, p_idx, seed_idx] = 0.0
                results['coherent_only']['n_kept'][evt_idx, p_idx, seed_idx] = 0
                results['coherent_only']['n_total'][evt_idx, p_idx, seed_idx] = 0

                # -- Mode 2: wavelet only (no coherent removal) --
                sr_wo = sparsify(noisy, config)
                recon_wo = reconstruct(sr_wo, config, N_TICKS)
                f0, bias, srms, nrms, crej, nsig, tcharge = compute_metrics_jax(
                    clean, recon_wo, noisy, jnp.float32(2.5))
                for k, v in zip(metric_names[:7],
                                [f0, bias, srms, nrms, crej, nsig, tcharge]):
                    results['wavelet_only'][k][evt_idx, p_idx, seed_idx] = float(v)
                results['wavelet_only']['sparsity'][evt_idx, p_idx, seed_idx] = sr_wo.sparsity
                results['wavelet_only']['n_kept'][evt_idx, p_idx, seed_idx] = sr_wo.n_kept
                results['wavelet_only']['n_total'][evt_idx, p_idx, seed_idx] = sr_wo.n_total

                # -- Mode 3: full pipeline --
                sr_full = sparsify(cleaned, config)
                recon_full = reconstruct(sr_full, config, N_TICKS)
                f0, bias, srms, nrms, crej, nsig, tcharge = compute_metrics_jax(
                    clean, recon_full, cleaned, jnp.float32(2.5))
                for k, v in zip(metric_names[:7],
                                [f0, bias, srms, nrms, crej, nsig, tcharge]):
                    results['full'][k][evt_idx, p_idx, seed_idx] = float(v)
                results['full']['sparsity'][evt_idx, p_idx, seed_idx] = sr_full.sparsity
                results['full']['n_kept'][evt_idx, p_idx, seed_idx] = sr_full.n_kept
                results['full']['n_total'][evt_idx, p_idx, seed_idx] = sr_full.n_total

        elapsed = time.perf_counter() - t_evt
        total = time.perf_counter() - t0
        eta = total / (evt_idx + 1) * (n_events - evt_idx - 1)
        print(f'  event {evt_idx:>3d}/{n_events}  {elapsed:.1f}s  '
              f'(elapsed {total:.0f}s, eta {eta:.0f}s)', flush=True)

    save_dict = dict(
        plane_keys=np.array(plane_keys),
        plane_types=np.array([PLANE_INFO[pk]['plane_type'] for pk in plane_keys]),
        n_events=n_events, n_planes=n_planes, n_seeds=N_SEEDS,
        modes=np.array(MODES),
    )
    for mode in MODES:
        for m in metric_names:
            save_dict[f'{mode}_{m}'] = results[mode][m]

    out_path = os.path.join(OUT_DIR, 'metrics_200evt_5seed_3mode.npz')
    np.savez(out_path, **save_dict)
    print(f'\nSaved to {out_path}')
    print(f'Total: {time.perf_counter() - t0:.0f}s')


if __name__ == '__main__':
    main()
