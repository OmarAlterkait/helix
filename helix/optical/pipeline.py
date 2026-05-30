"""End-to-end optical pipeline: stored chunks → batched DWT sparsify → metrics.

Operates on goop's stored chunks (the "stitches") directly. All chunks of an
event (both sides) are pad-batched and run through the shared
helix.core.sparsify in a single backend call — on the torch backend that is one
batched GPU DWT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from helix.core import backend as _backend
from helix.core.wavelet import sparsify, reconstruct, SparseResult, ThresholdSpec
from helix.optical.config import OpticalConfig
from helix.optical import io as _io
from helix.optical.metrics import chunk_metrics


@dataclass
class OpticalResult:
    sparse: SparseResult
    metrics: dict
    lengths: np.ndarray
    pmt_id: np.ndarray
    side: np.ndarray
    t0_ns: np.ndarray


def process_event(
    path,
    event_key: str,
    config: OpticalConfig,
    *,
    threshold: ThresholdSpec | None = None,
    compute_metrics: bool = True,
    to_numpy: bool = True,
) -> OpticalResult:
    """Sparsify all stored chunks of one event with the active backend.

    VisuShrink threshold uses a padding-independent per-chunk noise sigma; the
    surviving coefficients are quantized to ``config.quant_bits`` for storage.
    """
    th = threshold or config.threshold
    ec = _io.read_event_chunks(path, event_key, config)
    batch, lengths = _io.pad_batch(ec.chunks, config.dwt_level)
    sigma = _io.chunk_noise_sigma(ec.chunks)            # padding-free noise estimate

    x = _to_backend(batch)
    result = sparsify(x, wavelet=config.wavelet, level=config.dwt_level,
                      mode=config.dwt_mode, threshold=th, sigma=_to_backend(sigma))
    if config.quant_bits:
        result.coeffs = _quantize(result.coeffs, config.quant_bits)

    metrics = {}
    if compute_metrics:
        recon = reconstruct(result, batch.shape[1])
        metrics = chunk_metrics(batch, _as_numpy(recon), lengths)
        metrics["compression"] = result.compression
        metrics["sparsity"] = result.sparsity

    if to_numpy:
        result.sigma_per_band = np.asarray(result.sigma_per_band)
    return OpticalResult(sparse=result, metrics=metrics, lengths=lengths,
                         pmt_id=ec.pmt_id, side=ec.side, t0_ns=ec.t0_ns)


def _quantize(coeffs, bits: int):
    """Per-signal uniform quantization of the coefficient bands (numpy or torch)."""
    if not bits or bits >= 32:
        return coeffs
    if isinstance(coeffs, list) and coeffs and not isinstance(coeffs[0], np.ndarray):
        import torch
        if isinstance(coeffs[0], torch.Tensor):
            out = []
            for c in coeffs:
                amax = c.abs().amax(-1, keepdim=True).clamp_min(1e-12)
                step = amax / ((1 << (bits - 1)) - 1)
                out.append(torch.round(c / step) * step)
            return out
    return _io.quantize_coeffs(coeffs, bits)            # numpy list


def _to_backend(arr: np.ndarray):
    be = _backend.get_backend()
    if be == "numpy":
        return arr
    if be == "jax":
        import jax.numpy as jnp
        return jnp.asarray(arr)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.as_tensor(arr, device=dev)


def _as_numpy(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(arr)
