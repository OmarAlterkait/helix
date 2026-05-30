"""Backend selection + lazy, per-backend op dispatch.

Design goal (the whole point of this module): heavy frameworks — ``jax`` and
``torch`` — must be imported ONLY when a backend that needs them is actually
selected and used, never at ``import helix`` time. NumPy is the always-available
default.

Each op *family* (e.g. wavelet, coherent) ships one module per backend, named
``<family>_<backend>.py`` (e.g. ``wavelet_ops_numpy.py``, ``wavelet_ops_jax.py``,
``wavelet_ops_torch.py``). ``ops("helix.core.wavelet_ops")`` imports and returns
only the active backend's module — so the framework import happens inside that
file, lazily, the first time the op is called.

Selection precedence:  set_backend()  >  $HELIX_BACKEND  >  "numpy".
"""
from __future__ import annotations

import os
import importlib
from functools import lru_cache

VALID_BACKENDS = ("numpy", "jax", "torch")
_DEFAULT = "numpy"
_override: str | None = None


def set_backend(name: str) -> None:
    """Force the active backend process-wide (overrides $HELIX_BACKEND)."""
    if name not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {VALID_BACKENDS}, got {name!r}")
    global _override
    _override = name


def get_backend() -> str:
    """Return the active backend name without importing any framework."""
    if _override is not None:
        return _override
    env = os.environ.get("HELIX_BACKEND")
    if env is not None:
        if env not in VALID_BACKENDS:
            raise ValueError(
                f"$HELIX_BACKEND must be one of {VALID_BACKENDS}, got {env!r}")
        return env
    return _DEFAULT


@lru_cache(maxsize=None)
def import_backend_module(base: str, backend: str):
    """Import ``f'{base}_{backend}'`` (cached per (base, backend) pair)."""
    return importlib.import_module(f"{base}_{backend}")


def ops(base: str):
    """Return the active backend's op module for family ``base``.

    Example: ``ops("helix.core.wavelet_ops")`` -> the wavelet_ops_<backend> module.
    """
    return import_backend_module(base, get_backend())


def array_namespace():
    """Return the array module (np / jnp / torch) for the active backend.

    Imports the framework lazily — only call when you actually need it.
    """
    backend = get_backend()
    if backend == "numpy":
        import numpy as np
        return np
    if backend == "jax":
        import jax.numpy as jnp
        return jnp
    import torch
    return torch
