"""Basis provenance — a hashable descriptor of *how* a CoeffEvent was produced.

`BasisDescriptor` captures everything that defines the coefficient basis and the
processing that filled it: the wavelet transform (wavelet/level/mode + the padded
length, from which the per-band lengths derive), the coherent-removal spec, the
threshold spec, and the tokenize normalization scalar. Its ``digest`` is a stable
sha256 over the canonical form — the join key stamped into a shard's ``/config``
so every event in the shard shares one basis and provenance is never duplicated
per event.

This is decode + regeneration provenance in one small object; the on-disk
``/config`` stores its fields as attrs (once per shard).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def derive_band_lengths(wavelet: str, level: int, mode: str, padded_length: int) -> tuple[int, ...]:
    """Per-band lengths ``[len(cA), len(cD_L), …, len(cD_1)]`` for the basis.

    Derived by decomposing a zero vector of the padded length — the single
    source of truth the on-disk ``band_lengths`` is validated against on read.
    """
    import pywt

    z = np.zeros(padded_length, dtype=np.float32)
    coeffs = pywt.wavedec(z, wavelet, level=level, mode=mode)
    return tuple(int(c.shape[-1]) for c in coeffs)


def padded_length_of(band_lengths, wavelet: str, level: int, mode: str) -> int:
    """Effective DWT input length that produced ``band_lengths`` (via inverse).

    Mode-agnostic: recovers the padded length even when periodization pads
    internally, so the basis records a padded length consistent with the actual
    coefficients rather than assuming a specific pad convention.
    """
    import pywt

    z = [np.zeros((1, int(L)), dtype=np.float32) for L in band_lengths]
    return int(pywt.waverec(z, wavelet, mode=mode, axis=-1).shape[-1])


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def descriptor_digest(d: dict) -> str:
    """Stable sha256 over a dict, canonicalized (sorted keys, compact)."""
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class BasisDescriptor:
    """How a CoeffEvent was produced. Fields map 1:1 onto a shard's ``/config``."""

    wavelet: str = "coif3"
    level: int = 4
    mode: str = "periodization"
    n_ticks_raw: int = 4321
    pad: int = 15                                   # padded length = n_ticks_raw + pad
    band_lengths: tuple[int, ...] = ()              # PADDED per-band lengths (load-bearing)
    removal: dict = field(default_factory=dict)     # kind, kgate, ksig, npass, group_size
    threshold: dict = field(default_factory=dict)   # method, func, scale, per_band_sigma, ...
    sigma_norm: float = 2.6                          # SIGMA (tokenize normalization)

    @property
    def padded_length(self) -> int:
        return self.n_ticks_raw + self.pad

    @property
    def n_bands(self) -> int:
        return len(self.band_lengths)

    def with_derived_band_lengths(self) -> "BasisDescriptor":
        """Return a copy whose band_lengths are (re)derived from the basis."""
        bl = derive_band_lengths(self.wavelet, self.level, self.mode, self.padded_length)
        return BasisDescriptor(
            wavelet=self.wavelet, level=self.level, mode=self.mode,
            n_ticks_raw=self.n_ticks_raw, pad=self.pad, band_lengths=bl,
            removal=dict(self.removal), threshold=dict(self.threshold),
            sigma_norm=self.sigma_norm)

    def validate(self) -> None:
        """Fail loudly if band_lengths disagree with the generating basis."""
        want = derive_band_lengths(self.wavelet, self.level, self.mode, self.padded_length)
        got = tuple(int(x) for x in self.band_lengths)
        if got != want:
            raise ValueError(
                f"band_lengths {got} inconsistent with basis "
                f"({self.wavelet} L{self.level} {self.mode} @ {self.padded_length}) "
                f"→ expected {want}. A wrong Lb silently corrupts every (wire,tau).")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["band_lengths"] = [int(x) for x in self.band_lengths]
        return d

    def digest(self) -> str:
        return descriptor_digest(self.to_dict())
