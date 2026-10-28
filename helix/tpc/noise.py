"""The forward noise model of LArTPC wire planes, torch on the grids' device: what the detector adds
to a signal before the ADC (:func:`add_intrinsic_noise`) and the ADC (:func:`digitize`).

Two additive components:

* **incoherent** — per-channel independent noise, ENC = ``sqrt(white^2 + (y + z*L)^2)`` (MicroBooNE
  model, arXiv:1705.07341): a frequency-shaped series component renormalised to ``y + z*L`` per channel
  (the measured spectrum by default, as JAXTPC ``_noise_core``) plus flat white noise;
* **coherent** — one waveform per GROUP of ``group_size`` contiguous channels, adjacent groups
  anti-correlated by ``beta``, renormalised to ``rms_adc`` after the coupling (JAXTPC
  ``tools/coherent_noise.py``). Not divided by sqrt(N): it must survive averaging over the group,
  which is what the chain's coherent gate (on the readout's groups, the same ``wire // group_size``)
  removes.

The same functions build the simulation corpus (``scripts/build_sim.py``) and add fresh noise at
load time (:class:`helix.data.transforms.AddNoise`). Realisations come from per-event torch generators
(seeded by the caller): reproducible on one device type, not across devices, and statistically, not
bit-, equal to JAXTPC's numpy (tests/_noise_reference.py, the numpy port, is the check).
"""

from __future__ import annotations

import numpy as np
import torch

# ── Defaults: JAXTPC config/noise_spectrum.npz and its simulation.coherent_noise block
#    (tests/test_forward_noise.py, tests/test_dense_chain.py). ──
# (white_x [ADC], series_y [ADC], series_z [ADC/m]); wire length L is in METERS.
DEFAULT_ENC = (0.90, 0.79, 0.22)
DEFAULT_SAMPLING_RATE_HZ = 2.0e6  # 2 MHz (0.5 us / tick)
# coherent component (JAXTPC tools/coherent_noise.py defaults)
DEFAULT_COH_RMS_ADC = 2.5
DEFAULT_COH_CORNER_FREQ_HZ = 20_000.0
DEFAULT_COH_SLOPE = 1.5
DEFAULT_COH_BETA = 0.15
DEFAULT_GROUP_SIZE = 64

#: Sentinel for the MEASURED series spectrum shipped in ``data/``. It is the
#: default because white is the wrong answer and is silently easy to get:
#: ``AddNoise(coherent=True, incoherent=True)`` without a spectrum would give flat
#: noise. The measured spectrum concentrates power in 10-490 kHz and has NONE above
#: 500 kHz, so white misplaces noise across every wavelet band
#: (docs/archive/NOISE_BANDS.md).
#:
#: Pass ``series_spectrum=None`` for white — still available, but now an explicit
#: choice rather than what you get by saying nothing.
DEFAULT_SERIES_SPECTRUM = "microboone"

_SPECTRA = {"microboone": "noise_spectrum.npz"}
_spectrum_cache = {}


def load_series_spectrum(name=DEFAULT_SERIES_SPECTRUM):
    """Named spectrum -> ``(freqs_hz, amps)`` from the packaged ``data/`` file."""
    if name not in _SPECTRA:
        raise KeyError(f"unknown series spectrum {name!r}; known: {sorted(_SPECTRA)}")
    if name not in _spectrum_cache:
        import os
        path = os.path.join(os.path.dirname(__file__), "data", _SPECTRA[name])
        with np.load(path) as d:
            _spectrum_cache[name] = (d["spectrum_freqs_hz"].copy(),
                                     d["spectrum_shape"].copy())
    return _spectrum_cache[name]


def resolve_series_spectrum(spec):
    """``None`` -> white; a name -> the packaged measurement; a tuple -> itself."""
    if spec is None or isinstance(spec, tuple) and len(spec) == 2 and not isinstance(spec[0], str):
        return spec
    if isinstance(spec, str):
        return load_series_spectrum(spec)
    return spec


def _series_spectrum_shape(n_ticks, series_spectrum, sampling_rate_hz):
    """Interpolated series amplitude spectrum (shape only).

    The absolute scale is irrelevant: the series component is renormalised to
    ``series_rms`` per channel (as JAXTPC ``_noise_core``), so only the spectral *shape* matters. ``series_spectrum``
    is ``(freqs_hz, amps)`` (as in noise_spectrum.npz) or ``None`` → flat/white.
    """
    series_spectrum = resolve_series_spectrum(series_spectrum)
    if series_spectrum is None:
        return None
    freqs_emp, amps_emp = series_spectrum
    freqs = np.fft.rfftfreq(n_ticks, d=1.0 / sampling_rate_hz)
    return np.interp(freqs, np.asarray(freqs_emp, dtype=np.float64),
                     np.asarray(amps_emp, dtype=np.float64))


# decorrelate the per-component torch RNG streams from each other. Masked to 63
# bits so they don't fight the 63-bit seed mask in manual_seed.
_INCOH_SEED_SALT = 0x9E3779B97F4A7C15 & ((1 << 63) - 1)
_COH_SEED_SALT = 0xD1B54A32D192ED03 & ((1 << 63) - 1)


def _series_spectrum_torch(n_ticks, series_spectrum, sampling_rate_hz, device):
    series_spectrum = resolve_series_spectrum(series_spectrum)
    spec = _series_spectrum_shape(n_ticks, series_spectrum, sampling_rate_hz)
    if spec is None:
        return None
    return torch.as_tensor(spec, dtype=torch.float32, device=device)


def _incoherent_torch(shape, wire_lengths_m, *, gen, enc, series_spectrum,
                      sampling_rate_hz, device):
    """Incoherent noise (n_channels, n_ticks): the shaped series renormalised to ``y + z·L`` per
    channel, plus flat white."""
    n_ch, n_ticks = int(shape[0]), int(shape[1])
    white_x, series_y, series_z = enc
    L = torch.as_tensor(wire_lengths_m, dtype=torch.float32, device=device).reshape(-1)
    if L.numel() == 1:
        L = L.expand(n_ch)
    if L.numel() != n_ch:
        raise ValueError(f"wire_lengths has {L.numel()} entries, expected {n_ch}")
    series_rms = (series_y + series_z * L)[:, None]
    n_freq = n_ticks // 2 + 1

    real = torch.randn(n_ch, n_freq, generator=gen, device=device)
    imag = torch.randn(n_ch, n_freq, generator=gen, device=device)
    spec = _series_spectrum_torch(n_ticks, series_spectrum, sampling_rate_hz, device)
    if spec is not None:
        real = real * spec
        imag = imag * spec
    cpx = torch.complex(real, imag)
    cpx[:, 0] = cpx[:, 0].real
    if n_ticks % 2 == 0:
        cpx[:, -1] = cpx[:, -1].real
    shaped = torch.fft.irfft(cpx, n=n_ticks, dim=1)
    cur = shaped.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-10)
    shaped = shaped / cur * series_rms

    white = torch.randn(n_ch, n_ticks, generator=gen, device=device) * white_x
    return (shaped + white).to(torch.float32)


def _coherent_spectrum_torch(n_ticks, corner_freq_hz, spectral_slope,
                             sampling_rate_hz, device):
    """A(f) = 1/(1 + f/f_corner)^(slope/2), A(0) = 0 (no DC)."""
    freqs = torch.fft.rfftfreq(n_ticks, d=1.0 / sampling_rate_hz, device=device)
    spec = 1.0 / (1.0 + freqs / corner_freq_hz) ** (spectral_slope / 2.0)
    spec[0] = 0.0
    return spec.to(torch.float32)


def _coherent_torch(n_channels, n_ticks, *, gen, group_size, rms_adc,
                    corner_freq_hz, spectral_slope, beta, sampling_rate_hz, device):
    """Coherent noise (n_channels, n_ticks): all groups' shaped waveforms at once, adjacent-group
    anti-correlation ``w - beta*(w_{g-1}+w_{g+1})``, the RMS set to ``rms_adc`` by the MEASURED RMS
    after the coupling (JAXTPC 6998d81: the analytic pre-coupling norm runs ~2% high at the default
    beta), broadcast by ``wire // group_size``."""
    n_groups = (n_channels + group_size - 1) // group_size
    spec = _coherent_spectrum_torch(n_ticks, corner_freq_hz, spectral_slope,
                                    sampling_rate_hz, device)            # (n_freq,)
    real = torch.randn(n_groups, spec.numel(), generator=gen, device=device) * spec
    imag = torch.randn(n_groups, spec.numel(), generator=gen, device=device) * spec
    imag[:, 0] = 0.0                                  # DC real (zero imag, alias-free)
    if n_ticks % 2 == 0:
        imag[:, -1] = 0.0                            # Nyquist real (even n)
    base = torch.fft.irfft(torch.complex(real, imag), n=n_ticks, dim=1)  # (n_groups, n_ticks)

    z = base.new_zeros(1, n_ticks)
    left = torch.cat([z, base[:-1]], dim=0)
    right = torch.cat([base[1:], z], dim=0)
    wav = base - beta * (left + right)

    realized = wav.pow(2).mean().sqrt()
    if float(realized) > 0:
        wav = wav * (rms_adc / realized)

    wire_to_group = torch.arange(n_channels, device=device) // group_size
    return wav[wire_to_group].to(torch.float32)


def add_intrinsic_noise(grids, geom, *, seeds, enc=DEFAULT_ENC,
                        coherent=True, incoherent=False,
                        sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
                        group_size=DEFAULT_GROUP_SIZE, coh_rms=DEFAULT_COH_RMS_ADC,
                        coh_corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ,
                        coh_spectral_slope=DEFAULT_COH_SLOPE, beta=DEFAULT_COH_BETA,
                        series_spectrum=DEFAULT_SERIES_SPECTRUM):
    """Add fresh forward noise to per-plane dense grids in place; returns ``grids``.

    ``grids`` : ``{plane_id: (B, W, T)}`` on-device. ``geom`` : registry with
    ``n_wires``/``n_ticks`` (+ ``wire_lengths`` per plane when ``incoherent``).
    ``seeds`` : ``(B,)`` per-event content-addressed seeds.

    Each component draws from its own per-event torch Generator (seeded by ``seeds[b]``),
    advanced across planes in canonical-id order.
    """
    if not grids:
        return grids
    dev = next(iter(grids.values())).device
    B = next(iter(grids.values())).shape[0]
    # sorted so the per-event cross-plane draw order is a deterministic function
    # of the canonical plane id (not dict insertion / HDF5 enumeration order).
    gids = sorted(grids.keys())

    if coherent:
        for b in range(B):
            gen = torch.Generator(device=dev)
            gen.manual_seed((int(seeds[b]) ^ _COH_SEED_SALT) & ((1 << 63) - 1))
            for gid in gids:
                _, W, T = grids[gid].shape
                grids[gid][b] += _coherent_torch(
                    W, T, gen=gen, group_size=group_size, rms_adc=coh_rms,
                    corner_freq_hz=coh_corner_freq_hz, spectral_slope=coh_spectral_slope,
                    beta=beta, sampling_rate_hz=sampling_rate_hz, device=dev)

    if incoherent:
        for b in range(B):
            gen = torch.Generator(device=dev)
            gen.manual_seed((int(seeds[b]) ^ _INCOH_SEED_SALT) & ((1 << 63) - 1))
            for gid in gids:
                _, W, T = grids[gid].shape
                wl = geom[gid].get('wire_lengths')
                if wl is None:
                    raise ValueError(
                        f"incoherent noise needs 'wire_lengths' for plane {gid} "
                        "(configure wire_lengths_per_plane / the geometry registry)")
                grids[gid][b] += _incoherent_torch(
                    (W, T), wl, gen=gen, enc=enc, series_spectrum=series_spectrum,
                    sampling_rate_hz=sampling_rate_hz, device=dev)
    return grids


def digitize(grids, pedestal, n_bits=12, adc_max=None, gain=1.0):
    """Quantize per-plane grids to ADC codes, pedestal-subtracted: ``round(g*gain + ped).clip(0,
    adc_max) - ped`` (JAXTPC ``electronics._digitize_signal``). ``pedestal`` is a scalar or a
    ``{plane_id: ped}`` dict. ``adc_max`` defaults to ``(1 << n_bits) - 1``.
    """
    amax = float(adc_max) if adc_max is not None else float((1 << int(n_bits)) - 1)
    out = {}
    for gid, g in grids.items():
        ped = float(pedestal[gid]) if isinstance(pedestal, dict) else float(pedestal)
        out[int(gid)] = torch.round(g * gain + ped).clamp(0.0, amax) - ped
    return out
