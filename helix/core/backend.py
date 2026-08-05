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


# ── per-VALUE dispatch ───────────────────────────────────────────────────────
# The functions above dispatch on the ACTIVE BACKEND. These dispatch on an array
# ITSELF, which is what the pipeline needs: a stage receives whatever the stage
# before it produced, and must neither drag a device array to the host nor
# assume it is already there. Keeping this in one place is deliberate — the same
# question ("is this on an accelerator, and how do I touch it?") was previously
# answered separately, and jax-only, at four different sites, which is exactly
# why the torch path did not work end to end.

def kind_of(a) -> str:
    """``'jax'`` | ``'torch'`` | ``'numpy'`` — which framework owns ``a``."""
    mod = type(a).__module__
    if mod.startswith("jax"):
        return "jax"
    if mod.startswith("torch"):
        return "torch"
    return "numpy"


def is_device(a) -> bool:
    """True when ``a`` lives on an accelerator.

    A jax array is treated as on-device unconditionally (that is the jax model);
    a torch tensor only when its device is not CPU — a CPU tensor should take the
    host path, since moving it to a device to extract sparse rows would be pure
    overhead.
    """
    k = kind_of(a)
    if k == "jax":
        return True
    if k == "torch":
        return a.device.type != "cpu"
    return False


def to_numpy(a, dtype=None):
    """Host numpy copy of ``a``, whatever owns it (torch needs detach+cpu)."""
    import numpy as np
    if kind_of(a) == "torch":
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype) if dtype is not None else np.asarray(a)


#: reduction length above which q50 sorts instead of selecting. 4096 sits well
#: past the gate's per-block axes (group_size=64) and well below the flattened
#: bands (0.5-4.3M) where kthvalue collapses.
_Q50_SORT_ABOVE = 4096


def torch_q50(x, dim: int):
    """``np.median``-equivalent for torch: the AVERAGE of the two middle values.

    ``torch.median`` returns the LOWER of the two middles; ``np.median`` and
    ``np.quantile(..., 0.5)`` average them. On even-length inputs the two differ,
    and every σ in this library is a median-absolute-deviation — so the choice
    propagates straight into a threshold, and a threshold is a DISCONTINUOUS
    function of its input. Using torch.median makes σ systematically smaller than
    numpy's, which lowers the threshold and keeps coefficients numpy drops
    (measured: torch kept 1-2 extra per event, never fewer — the asymmetry that
    exposed this).

    ``coherent_gate_ops_numpy._sigc`` documents the same trap for the gate; this
    is the wavelet-threshold half of it.

    Two implementations, chosen by the REDUCTION LENGTH, because their costs
    invert. ``kthvalue`` is a selection and wins for short axes; on a long one it
    is pathological on CUDA — measured over a flattened coif3 band it went 3.5 ms
    at 0.53M elements to **61 ms at 4.27M**, against 1.1 ms for a full sort. That
    one call was the whole cost of ``threshold_bands`` (112 ms of a 159 ms plane).
    Sorting is O(n log n) but hits a vastly better CUDA kernel, and it also has no
    size ceiling — unlike ``torch.quantile``, which is comparably fast here but
    refuses inputs above ~2**24 (reachable on optical chunk lengths).
    """
    import torch
    n = x.shape[dim]
    if n <= _Q50_SORT_ABOVE:                      # short axis: selection is cheaper
        lo = torch.kthvalue(x, (n + 1) // 2, dim=dim).values
        if n % 2:
            return lo
        hi = torch.kthvalue(x, n // 2 + 1, dim=dim).values
        return (lo + hi) * 0.5
    s, _ = torch.sort(x, dim=dim)
    m = n // 2
    if n % 2:
        return s.select(dim, m)
    return (s.select(dim, m - 1) + s.select(dim, m)) * 0.5
