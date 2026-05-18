"""Backend detection: JAX (GPU/CPU) vs NumPy fallback."""

from __future__ import annotations

_backend: str | None = None


def get_backend() -> str:
    global _backend
    if _backend is not None:
        return _backend
    try:
        import jax
        devices = jax.devices()
        _backend = "jax"
    except (ImportError, RuntimeError):
        _backend = "numpy"
    return _backend


def array_module():
    if get_backend() == "jax":
        import jax.numpy as jnp
        return jnp
    import numpy as np
    return np
