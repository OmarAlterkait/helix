"""helix's LArTPC transforms, registered into pimm-data's registry.

The KERNELS live in ``helix.tpc`` (numpy/torch, no pimm_data import -- that is
what keeps the DSP path installable on numpy alone). The registered WRAPPERS
live here, because ``@TRANSFORMS.register_module()`` requires importing
``pimm_data.transform``, which pulls 36 pimm_data modules and torch. Measured,
not assumed: ``helix.tpc`` imports today with neither.

Going to dense is NOT here. ``Densify`` stays in pimm-data: scattering sparse
rows into per-key grids is a data-layer operation that any detector family
wants. What is here is the LArTPC forward model applied to those grids -- the
ENC/coherent noise and the ADC digitisation -- which is physics, and helix's.

A config reaches these with ``custom_imports=["helix.data.transforms"]``.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from pimm_data.transform import TRANSFORMS
from pimm_data.batch_transforms import content_seed, _seeds_for

from helix.tpc import dense_ops
from helix.tpc.noise import (generate_noise, digitize, DEFAULT_ENC,
                             DEFAULT_SERIES_SPECTRUM, DEFAULT_SAMPLING_RATE_HZ,
                             DEFAULT_GROUP_SIZE, DEFAULT_COH_RMS_ADC,
                             DEFAULT_COH_CORNER_FREQ_HZ, DEFAULT_COH_SLOPE,
                             DEFAULT_COH_BETA)


@TRANSFORMS.register_module()
class AddNoise:
    """Inject forward detector noise on dense sensor planes (after ``Densify``).

    Thin adapter over :func:`helix.tpc.noise.add_noise`. Operates on a ``sensor``
    sub-dict that already carries ``dense`` (so it must run *after* ``Densify``):
    each plane image is perturbed in place by incoherent and/or per-group
    coherent noise. Tags (``incoherent`` / ``coherent`` and the parameters) are
    set from the config.

    Reproducibility: the per-event RNG is derived from a stable event id
    (``sub['name']``, surfaced by the dataset) hashed with ``base_seed`` — so the
    same event gets the same noise every epoch regardless of which DataLoader
    worker happens to draw it, without touching numpy's global RNG state.

    Default tags are ``incoherent=False, coherent=True``: JAXTPC output already
    carries incoherent noise, so the common load-time use is adding the coherent
    component it omits. Set ``incoherent=True`` (with ``wire_lengths_m``) to add
    incoherent noise to noise-free input.
    """

    scope = 'sample'  # per-event independent; placeable pre- OR post-collate

    # The defaults come FROM helix.tpc.noise, not from literals repeating it.
    # This signature imported all five of these constants and then re-typed their
    # values inline (group_size=64, coh_rms=2.5, 20000.0, 1.5, beta=0.15). They
    # agreed, so nothing failed -- but retuning the forward model in noise.py
    # would have left the registered transform silently on the old values. Same
    # drift that DetectorConfig's third copy of the ENC triple allowed.
    def __init__(self, incoherent=False, coherent=True,
                 group_size=DEFAULT_GROUP_SIZE,
                 wire_lengths_m=None, enc=DEFAULT_ENC,
                 series_spectrum=DEFAULT_SERIES_SPECTRUM,
                 sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
                 coh_rms=DEFAULT_COH_RMS_ADC,
                 coh_corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ,
                 coh_spectral_slope=DEFAULT_COH_SLOPE, beta=DEFAULT_COH_BETA,
                 base_seed=0, dense_key='dense', planes=None, name_key='name',
                 geom=None, modality=None, coherent_numpy=False, offset_key='offset'):
        self.incoherent = bool(incoherent)
        # flat collated-batch path (post-collate, torch): geom + flat dense key.
        self.geom = geom
        self.modality = modality
        self.coherent_numpy = bool(coherent_numpy)
        self.offset_key = offset_key
        self.coherent = bool(coherent)
        self.group_size = int(group_size)
        self.wire_lengths_m = wire_lengths_m
        self.enc = tuple(enc)
        self.series_spectrum = series_spectrum
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.coh_rms = float(coh_rms)
        self.coh_corner_freq_hz = float(coh_corner_freq_hz)
        self.coh_spectral_slope = float(coh_spectral_slope)
        self.beta = float(beta)
        self.base_seed = int(base_seed)
        self.dense_key = dense_key
        self.planes = planes
        self.name_key = name_key

    def _event_rng(self, name):
        h = hashlib.blake2b(str(name).encode('utf-8'), digest_size=8).digest()
        seed = (int.from_bytes(h, 'little') ^ self.base_seed) & ((1 << 64) - 1)
        return np.random.default_rng(seed)

    def __call__(self, data):
        # DISPATCH: a collated batch carries the flat <mod>_offset key (torch grids
        # {gid:(B,W,T)}); a per-event sub-dict does not (numpy grids {gid:(W,T)}).
        pfx = f'{self.modality}_' if self.modality is not None else ''
        if (pfx + self.offset_key) in data:
            return self._noise_batch(data, pfx)
        sub = data
        if sub.get('readout_type') == 'pixel':
            return sub
        dense = sub.get(self.dense_key)
        if dense is None:
            raise KeyError(
                f"AddNoise: no {self.dense_key!r} in the sensor modality — run "
                "Densify before AddNoise.")
        rng = self._event_rng(sub.get(self.name_key, ''))
        labels = self.planes if self.planes is not None else list(dense)
        for label in labels:
            if label not in dense:
                continue
            img = dense[label]
            noise = generate_noise(
                img.shape, rng=rng, wire_lengths_m=self.wire_lengths_m,
                incoherent=self.incoherent, coherent=self.coherent,
                enc=self.enc, series_spectrum=self.series_spectrum,
                sampling_rate_hz=self.sampling_rate_hz,
                group_size=self.group_size, coh_rms=self.coh_rms,
                coh_corner_freq_hz=self.coh_corner_freq_hz,
                coh_spectral_slope=self.coh_spectral_slope, beta=self.beta)
            dense[label] = img + noise
        return sub

    def _noise_batch(self, batch, pfx):
        """Post-collate path: add fresh per-event noise to the collated grids
        ``batch[<mod>_dense]`` (torch, on the inputs' device). Seeds self-derive
        from ``batch['name']`` folded with base_seed/_epoch/_rank."""
        grids = batch[pfx + self.dense_key]
        seeds = _seeds_for(batch, self.base_seed)
        if seeds is None:                              # no 'name' -> position fallback
            seeds = [content_seed(f"_idx{i}", self.base_seed)
                     for i in range(next(iter(grids.values())).shape[0])]
        dense_ops.add_intrinsic_noise(
            grids, self.geom, seeds=seeds, enc=self.enc,
            coherent=self.coherent, incoherent=self.incoherent,
            sampling_rate_hz=self.sampling_rate_hz, group_size=self.group_size,
            coh_rms=self.coh_rms, coh_corner_freq_hz=self.coh_corner_freq_hz,
            coh_spectral_slope=self.coh_spectral_slope, beta=self.beta,
            series_spectrum=self.series_spectrum, coherent_numpy=self.coherent_numpy)
        return batch


@TRANSFORMS.register_module()
class Digitize:
    """Quantize dense sensor planes to integer ADC codes (production digitize).

    Per plane: ``round(img*gain + pedestal).clip(0, adc_max) - pedestal`` — the
    pedestal-subtracted output of JAXTPC's ``_digitize_signal`` / the doraemon
    ``make_noisy`` path. Run it LAST in the dense chain
    (``Densify -> AddNoise -> Digitize``), so the analog ``signal + noise`` is
    quantized exactly as the detector would.

    Pedestal resolution (raw-ADC offset that sets where 0 and saturation fall):
    the ``pedestal`` arg wins (a scalar applied to every plane, or a
    ``{plane_label: pedestal}`` dict), else the per-plane pedestal surfaced by
    the reader (``sub['pedestal']``), else ``0`` — or raise when
    ``require_pedestal=True``. ``adc_max`` defaults to ``(1 << n_bits) - 1``
    (12-bit → 4095). Wire readout only (pixel streams pass through unchanged).

    Parameters
    ----------
    n_bits : int
        Code depth; sets ``adc_max = (1 << n_bits) - 1`` when ``adc_max`` is None.
    adc_max : float or None
        Explicit max code (overrides ``n_bits``).
    gain : float
        Scale applied before adding the pedestal (1.0 = input already in ADC).
    pedestal : float, dict, or None
        Pedestal override (scalar or per-plane). None → use ``sub['pedestal']``.
    require_pedestal : bool
        Raise if a plane has no resolvable pedestal (instead of defaulting to 0).
    dense_key : str
        Which dense field to quantize. Default ``'dense'``.
    planes : list or None
        Restrict to these plane labels (None → all dense planes).
    """

    scope = 'sample'  # per-event independent; placeable pre- OR post-collate

    def __init__(self, n_bits=12, adc_max=None, gain=1.0, pedestal=None,
                 require_pedestal=False, dense_key='dense', planes=None,
                 geom=None, modality=None, offset_key='offset'):
        self.n_bits = int(n_bits)
        self.adc_max = adc_max
        self.gain = float(gain)
        self.pedestal = pedestal
        self.require_pedestal = bool(require_pedestal)
        self.dense_key = dense_key
        self.planes = planes
        self.geom = geom or {}
        self.modality = modality
        self.offset_key = offset_key

    def _pedestal_for(self, label, sub):
        if isinstance(self.pedestal, dict):
            if label in self.pedestal:
                return float(self.pedestal[label])
        elif self.pedestal is not None:
            return float(self.pedestal)
        ped_map = sub.get('pedestal', {})
        if label in ped_map:
            return float(ped_map[label])
        if self.require_pedestal:
            raise KeyError(
                f"Digitize: no pedestal for plane {label!r}; surface it from "
                "the reader (sensor file pedestal attr) or pass pedestal=.")
        return 0.0

    def __call__(self, data):
        # DISPATCH: collated batch (flat <mod>_offset present) -> torch path;
        # per-event sub-dict otherwise.
        pfx = f'{self.modality}_' if self.modality is not None else ''
        if (pfx + self.offset_key) in data:
            return self._digitize_batch(data, pfx)
        sub = data
        if sub.get('readout_type') == 'pixel':
            return sub
        dense = sub.get(self.dense_key)
        if dense is None:
            raise KeyError(
                f"Digitize: no {self.dense_key!r} in the sensor modality — run "
                "Densify (and AddNoise) before Digitize.")
        marker = f'_digitized_{self.dense_key}'
        if sub.get(marker):
            raise RuntimeError(
                f"Digitize: {self.dense_key!r} already digitized — digitize is "
                "not idempotent (pedestal/gain round-trip); run it at most once.")
        labels = self.planes if self.planes is not None else list(dense)
        for label in labels:
            if label not in dense:
                continue
            ped = self._pedestal_for(label, sub)
            dense[label] = digitize(dense[label], ped, n_bits=self.n_bits,
                                    adc_max=self.adc_max, gain=self.gain)
        sub[marker] = True
        return sub

    def _digitize_batch(self, batch, pfx):
        """Post-collate path: quantize the collated grids ``batch[<mod>_dense]``
        (torch, on the inputs' device) via dense_ops. Pedestal: the ``pedestal``
        arg wins, else per-plane from ``geom``."""
        ped = self.pedestal
        if ped is None:
            ped = {gid: e.get('pedestal', 0) for gid, e in self.geom.items()}
        batch[pfx + self.dense_key] = dense_ops.digitize(
            batch[pfx + self.dense_key], ped, n_bits=self.n_bits,
            adc_max=self.adc_max, gain=self.gain)
        return batch


# The wire-plane recipe builders. They name AddNoise/Digitize, so they live
# with them rather than in pimm-data's generic batch_transforms.
def sensor_dense_cfg(geom, *, modality='sensor', device=None, base_seed=0,
                     coherent=True, incoherent=False, digitize=True, n_bits=12,
                     dense_key=None, **noise_kw):
    """The standard dense sensor chain as plain ``dict(type=…)`` configs:
    ``[ToDevice?, Densify, AddNoise, Digitize?]``.

    Run it with the ordinary ``Compose`` (these are ``scope='sample'`` transforms —
    there is **no batch-transform runner**): ``Compose([*sensor_dense_cfg(geom,
    device='cuda'), my_user_gpu_fn])(batch)``. Noise self-seeds from ``batch['name']``
    (``base_seed`` here; epoch/rank from optional ``batch['_epoch']``/``_rank``).

    Densify is opt-in and additive (the sparse COO is never replaced). ``modality=
    'sensor'`` writes flat ``sensor_dense``; ``modality=None`` writes bare ``dense``.
    """
    if dense_key is None:
        dense_key = 'dense'
    cfg = []
    if device is not None:
        cfg.append(dict(type='ToDevice', device=device))
    cfg.append(dict(type='Densify', geom=geom, modality=modality,
                    dense_key=dense_key))
    cfg.append(dict(type='AddNoise', geom=geom, modality=modality,
                    base_seed=base_seed, coherent=coherent, incoherent=incoherent,
                    dense_key=dense_key, **noise_kw))
    if digitize:
        cfg.append(dict(type='Digitize', geom=geom, modality=modality,
                        n_bits=n_bits, dense_key=dense_key))
    return cfg


def build_sensor_gpu_stages(geom, *, modality=None, **kw):
    """Convenience: the dense sensor chain as a runnable ``Compose``.

    ``stages = build_sensor_gpu_stages(geom, device='cuda'); batch = stages(batch)``.
    There is no separate batch-transform runner — the dense ops are ordinary
    ``scope='sample'`` transforms run by ``Compose`` (``ToDevice`` is the device step;
    noise self-seeds). Default ``modality=None`` (bare-batch); pass ``modality=
    'sensor'`` for flat ``sensor_*``.
    """
    from pimm_data.transform import Compose
    return Compose(sensor_dense_cfg(geom, modality=modality, **kw))
