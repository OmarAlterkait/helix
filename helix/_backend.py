"""Back-compat shim — backend selection moved to helix.core.backend.

Legacy code set ``helix._backend._backend = 'jax'``; use
``helix.core.backend.set_backend('jax')`` instead.
"""
from helix.core.backend import get_backend, set_backend, array_namespace

__all__ = ["get_backend", "set_backend", "array_module"]


def array_module():
    return array_namespace()
