"""Shared data / noise / metric utilities for the wire-denoise study.

Clean truth: doraemon run_0026628546 (noise-free, 2-ADC threshold) read via
pimm_data. Noise added synthetically with the JAXTPC model:
  intrinsic = FFT-shaped colored series (noise_spectrum.npz) + flat white
  coherent  = per-group waveform (tools.coherent_noise) broadcast to 64-wire groups
Metric matches production compute_metrics_jax (F0 on signal pixels).
"""
import os
import sys
import os as _os
import numpy as np

sys.path.insert(0, '/sdf/home/o/omara/neutrino_group/omara/pimm-data/src')
sys.path.insert(0, '/sdf/home/o/omara/neutrino_group/omara/JAXTPC')

DATA_ROOT = '/sdf/home/o/omara/neutrino_data/omara/doraemon'
SPLIT = 'run_0026628546'
N_TICKS = 4321
CACHE = '/sdf/group/neutrino/omara/helix/temp/doraemon_run/explore_cache'
os.makedirs(CACHE, exist_ok=True)

# Canonical plane per type (volume_0); geometry from the doraemon config.
PLANES = {
    'U': {'group': 'volume_0_U', 'n_wires': 1969, 'pedestal': 1843, 'wire_len': (0.42, 4.63)},
    'V': {'group': 'volume_0_V', 'n_wires': 1969, 'pedestal': 1843, 'wire_len': (0.42, 4.63)},
    'Y': {'group': 'volume_0_Y', 'n_wires': 1443, 'pedestal': 410,  'wire_len': (2.33, 2.33)},
}

NOISE_X, NOISE_Y, NOISE_Z = 0.90, 0.79, 0.22  # white RMS; series RMS = Y + Z*wire_len
# Stale absolute path fixed 2026-08-15: this named
# /sdf/home/o/omara/neutrino_group/omara/JAXTPC/config/noise_spectrum.npz, which
# no longer resolves, so importing anything under research/coherent_coeffs died
# at IMPORT time (cc_common -> common). NOISE_SPECTRUM_NPZ overrides; the default
# is the live JAXTPC config.
_NPZ_PATH = _os.environ.get(
    "NOISE_SPECTRUM_NPZ",
    "/sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz")
_NPZ = np.load(_NPZ_PATH)

_DS = None
def _ds():
    global _DS
    if _DS is None:
        from pimm_data.jaxtpc import JAXTPCDataset
        _DS = JAXTPCDataset(data_root=DATA_ROOT, split=SPLIT, modalities=('sensor',))
    return _DS


# ----------------------------------------------------------------------------- data
def load_clean(events, ptype):
    """(len(events), n_wires, N_TICKS) float32 pedestal-subtracted clean truth."""
    info = PLANES[ptype]
    nw, grp = info['n_wires'], info['group']
    key = os.path.join(CACHE, f'clean_{ptype}_{events[0]}_{events[-1]}_{len(events)}.npy')
    if os.path.exists(key):
        return np.load(key)
    ds = _ds()
    out = np.zeros((len(events), nw, N_TICKS), np.float32)
    for i, e in enumerate(events):
        raw = ds[e]['sensor']['raw'][grp]
        out[i, raw['wire'], raw['time']] = raw['value']
    np.save(key, out)
    return out


# ----------------------------------------------------------------------------- noise
def _spectrum(n_ticks, fs_hz=2e6):
    n_freq = n_ticks // 2 + 1
    freqs = np.fft.rfftfreq(n_ticks, d=1.0 / fs_hz)
    spec = np.interp(freqs, _NPZ['spectrum_freqs_hz'], _NPZ['spectrum_shape'])
    e = np.sum(spec ** 2)
    if e > 0:
        spec = spec / np.sqrt(e) * np.sqrt(n_freq)
    return spec.astype(np.float32)


def intrinsic_noise(nw, ptype, rng, n_ticks=N_TICKS):
    """JAXTPC intrinsic noise: FFT-shaped colored series + flat white (per wire)."""
    spec = _spectrum(n_ticks)
    n_freq = len(spec)
    lo, hi = PLANES[ptype]['wire_len']
    wl = np.linspace(lo, hi, nw)
    series_rms = (NOISE_Y + NOISE_Z * wl).astype(np.float32)
    r = rng.standard_normal((nw, n_freq)) * spec
    im = rng.standard_normal((nw, n_freq)) * spec
    cpx = r + 1j * im
    cpx[:, 0] = cpx[:, 0].real
    shaped = np.fft.irfft(cpx, n=n_ticks, axis=1)
    cur = np.maximum(shaped.std(axis=1, keepdims=True), 1e-10)
    shaped = shaped / cur * series_rms[:, None]
    white = rng.standard_normal((nw, n_ticks)) * NOISE_X
    return (shaped + white).astype(np.float32)


def coherent_noise(nw, rng, n_ticks=N_TICKS, group_size=64, beta=0.15, rms_adc=2.5):
    """JAXTPC coherent noise (group waveforms broadcast to wires) via tools.coherent_noise."""
    from tools.coherent_noise import generate_group_waveforms, broadcast_to_wires
    ng = (nw + group_size - 1) // group_size
    wf = generate_group_waveforms(ng, n_ticks, beta=beta, rms_adc=rms_adc, rng=rng)
    return broadcast_to_wires(wf, nw, group_size).astype(np.float32)


def digitize(signal, pedestal):
    """Add pedestal, round, clip to 12-bit, subtract pedestal (production digitize)."""
    return np.round(signal + pedestal).clip(0, 4095).astype(np.float32) - pedestal


def make_noisy(clean, ptype, seed, coherent=False):
    """clean (E,nw,T) -> digitized noisy with intrinsic (+ optional coherent) noise.

    One independent noise realization per event; deterministic in seed."""
    info = PLANES[ptype]
    nw, ped = info['n_wires'], info['pedestal']
    out = np.empty_like(clean)
    for i in range(clean.shape[0]):
        rng = np.random.default_rng(seed * 100003 + i)
        n = intrinsic_noise(nw, ptype, rng)
        if coherent:
            n = n + coherent_noise(nw, np.random.default_rng(seed * 100003 + i + 7919))
        out[i] = digitize(clean[i] + n, ped)
    return out


# ----------------------------------------------------------------------------- metric
def f0_noise(clean, recon):
    """(F0 on signal pixels, RMS on non-signal pixels) — matches compute_metrics_jax."""
    sig = np.abs(clean) > 0
    tc = float(np.abs(clean)[sig].sum())
    f0 = 1.0 - float(np.abs(recon - clean)[sig].sum()) / max(tc, 1e-9)
    off = ~sig
    nrms = float(np.sqrt(np.mean((recon - clean)[off] ** 2))) if off.any() else 0.0
    return f0, nrms


def aggregate(clean_stack, recon_stack):
    """Pooled F0/noise over a stack of (nw,T) images (pool pixels, not avg of ratios)."""
    sig = np.abs(clean_stack) > 0
    tc = float(np.abs(clean_stack)[sig].sum())
    f0 = 1.0 - float(np.abs(recon_stack - clean_stack)[sig].sum()) / max(tc, 1e-9)
    off = ~sig
    nrms = float(np.sqrt(np.mean((recon_stack - clean_stack)[off] ** 2)))
    return f0, nrms


# ----------------------------------------------------------------------------- split
def train_test_events(n_train=24, n_test=12, stride=37):
    """Disjoint event id lists drawn across shards (stride avoids same-shard clumping)."""
    ids = list(range(0, 20000, stride))
    return ids[:n_train], ids[n_train:n_train + n_test]


if __name__ == '__main__':
    import time
    tr, te = train_test_events()
    print('train', tr[:5], '... test', te[:5])
    t = time.perf_counter()
    c = load_clean(te[:2], 'Y')
    print('loaded clean Y', c.shape, f'{time.perf_counter()-t:.1f}s', 'occ%', 100*(c != 0).mean())
    n = make_noisy(c, 'Y', seed=1, coherent=False)
    print('noisy Y', n.shape, 'off-pixel std (noise floor)', float(n[c == 0].std()))
    print('self F0 (noisy vs clean):', f0_noise(c, n))
