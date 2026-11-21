"""numpy or torch under one spelling: the few array operations the cell joins are written in.

The joins from fine cells to tokens (:mod:`helix.model.cells_join`) and the key packing under them
(:mod:`helix.model.tokenize`, :mod:`helix.data.cells`) run in two places: on numpy in a DataLoader
worker, building the training labels, and on the event's device at inference, where a numpy round
trip per call cost most of an event's time. One implementation serves both: a function asks
:func:`namespace` for the namespace of its inputs and spells every operation through it, so the two
cannot drift apart.

What differs between the libraries and is handled here, not at the call sites:

* dtypes and devices -- ``xp.int64`` etc.; ``xp.asarray`` puts a constant on the inputs' device;
* argument spellings -- ``clip`` (torch's ``clamp``, with an array bound), ``concat``, ``size``.

What a call site must still do itself: torch promotes ``int64 / int`` and ``int64 + 0.5`` to
float32 where numpy promotes to float64, so integer arithmetic that turns into float casts to the
float dtype it means first (``xp.astype(x, xp.float64)``). float64 arithmetic is IEEE on both, so
an integer result computed through it is bit-identical; float32 transcendental functions
(``sin``/``cos``) agree to rounding.

torch is imported only when a torch tensor is passed.
"""
from __future__ import annotations

import numpy as np


def is_torch(x):
    """Whether ``x`` is a torch tensor (decided without importing torch)."""
    return type(x).__module__.split(".")[0] == "torch"


def namespace(*xs):
    """The namespace of the arrays ``xs``: torch's, on the first tensor's device, if any of them
    is a tensor, else numpy's (lists and scalars included)."""
    for x in xs:
        if is_torch(x):
            return _Torch(x.device)
    return NUMPY


class _Numpy:
    int64, float32, float64 = np.int64, np.float32, np.float64

    asarray = staticmethod(np.asarray)
    where = staticmethod(np.where)
    floor = staticmethod(np.floor)
    round = staticmethod(np.round)
    sin, cos = staticmethod(np.sin), staticmethod(np.cos)
    stack = staticmethod(np.stack)
    concat = staticmethod(np.concatenate)
    unique = staticmethod(np.unique)

    @staticmethod
    def unique_inverse(x):
        """``(values, inverse)``: the sorted unique values and each element's index into them."""
        return np.unique(x, return_inverse=True)

    @staticmethod
    def astype(x, dtype):
        return x.astype(dtype)

    @staticmethod
    def full(shape, value, dtype):
        return np.full(shape, value, dtype)

    @staticmethod
    def arange(n, dtype=np.int64):
        return np.arange(n, dtype=dtype)

    @staticmethod
    def clip(x, lo, hi):
        return np.clip(x, lo, hi)

    @staticmethod
    def searchsorted(a, v):
        return np.searchsorted(a, v)

    @staticmethod
    def argsort(x):
        return np.argsort(x, kind="stable")

    @staticmethod
    def size(x):
        return int(x.size)

    @staticmethod
    def numpy(x):
        return np.asarray(x)


NUMPY = _Numpy()


class _Torch:
    def __init__(self, device):
        import torch
        self._t, self.device = torch, device
        self.int64, self.float32, self.float64 = torch.int64, torch.float32, torch.float64
        self.where, self.floor, self.round = torch.where, torch.floor, torch.round
        self.sin, self.cos = torch.sin, torch.cos

    def asarray(self, x, dtype=None):
        return self._t.as_tensor(x, dtype=dtype, device=self.device)

    def astype(self, x, dtype):
        return x.to(dtype)

    def full(self, shape, value, dtype):
        return self._t.full(shape if isinstance(shape, tuple) else (shape,), value, dtype=dtype,
                            device=self.device)

    def arange(self, n, dtype=None):
        return self._t.arange(n, dtype=dtype or self._t.int64, device=self.device)

    def clip(self, x, lo, hi):
        """``np.clip``: either bound a number, a tensor or None."""
        if lo is not None:
            x = self._t.maximum(x, lo) if is_torch(lo) else self._t.clamp(x, min=lo)
        if hi is not None:
            x = self._t.minimum(x, hi) if is_torch(hi) else self._t.clamp(x, max=hi)
        return x

    def searchsorted(self, a, v):
        return self._t.searchsorted(a, v.contiguous())

    def argsort(self, x):
        return self._t.argsort(x, stable=True)

    def stack(self, xs, axis=0):
        return self._t.stack(list(xs), axis)

    def concat(self, xs, axis=0):
        return self._t.cat(list(xs), axis)

    def unique(self, x):
        return self._t.unique(x, sorted=True)

    def unique_inverse(self, x):
        return self._t.unique(x, sorted=True, return_inverse=True)

    @staticmethod
    def size(x):
        return int(x.numel())

    @staticmethod
    def numpy(x):
        return x.detach().cpu().numpy()
