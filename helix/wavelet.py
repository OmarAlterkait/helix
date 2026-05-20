"""Wavelet sparsification: per-wire DWT → threshold → sparse representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from helix.config import DetectorConfig
from helix._backend import get_backend


@dataclass
class SparseResult:
    """Output of wavelet sparsification."""
    coeffs: Any
    n_kept: int
    n_total: int
    sigma_per_band: np.ndarray

    @property
    def sparsity(self) -> float:
        return 1.0 - self.n_kept / max(self.n_total, 1)


def sparsify(
    image: Any,
    config: DetectorConfig,
) -> SparseResult:
    """Per-wire DWT → hard threshold → sparse coefficients.

    Uses GPU matmul when JAX backend is active, pywt on CPU otherwise.
    """
    if get_backend() == "jax":
        return _sparsify_jax(image, config)
    return _sparsify_numpy(image, config)


def _sparsify_numpy(image, config):
    from helix._numpy_ops import per_wire_dwt, estimate_subband_sigma, hard_threshold

    img_np = np.array(image, dtype=np.float32, copy=True)
    coeffs = per_wire_dwt(img_np, config.wavelet, config.dwt_level)
    sigma = estimate_subband_sigma(coeffs)
    thresholded = hard_threshold(coeffs, sigma, config.threshold_kappa, config.threshold_include_approx)

    n_kept = sum(int(np.count_nonzero(c)) for c in thresholded)
    n_total = sum(c.size for c in thresholded)

    return SparseResult(coeffs=thresholded, n_kept=n_kept, n_total=n_total, sigma_per_band=sigma)


def _sparsify_jax(image, config):
    import jax
    import jax.numpy as jnp
    from helix._jax_ops import wavelet_pipeline_jax

    image_j = jnp.asarray(image, dtype=jnp.float32)
    recon, thresholded, sigmas, n_kept, n_total = wavelet_pipeline_jax(
        image_j, config.wavelet, config.dwt_level,
        config.threshold_kappa, config.threshold_include_approx)
    jax.block_until_ready(recon)

    return SparseResult(
        coeffs=thresholded,
        n_kept=int(n_kept),
        n_total=int(n_total),
        sigma_per_band=np.asarray(sigmas),
    )


def reconstruct(result: SparseResult, config: DetectorConfig, n_time: int) -> Any:
    """Inverse DWT from sparse coefficients."""
    if get_backend() == "jax":
        import jax.numpy as jnp
        from helix._jax_ops import idwt_matmul
        coeffs_j = jnp.asarray(result.coeffs)
        return idwt_matmul(coeffs_j, config.wavelet, n_time, config.dwt_level)

    # NumPy path: coeffs is a list of arrays
    from helix._numpy_ops import per_wire_idwt
    return per_wire_idwt(result.coeffs, config.wavelet, n_time)
