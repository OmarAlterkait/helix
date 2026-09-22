"""Rotary position embedding, in a form that is one kernel per rotation.

``layers.apply_rope`` runs ~14 kernels per call over strided views — ``cos``,
``sin``, two ``repeat_interleave``, a negate, a ``stack`` over
``v[..., 1::2]`` / ``v[..., 0::2]``, two multiplies and an add, PER HALF, then a
``cat``. Measured on an A100 against that card's 1,285 GiB/s: **3.2-9.8 % of the
memory roofline**, and 31 % of the encoder+decoder trunk's device time — more
than the MLP and roughly 4x the attention it feeds (``docs/PERFORMANCE.md`` §3).

Two things cause it, and both are avoidable without changing the arithmetic:

* ``cos``/``sin`` are recomputed in all 32 calls of a forward pass, from angle
  tensors that are constant across layers. :func:`rope_tables` builds them once.
* the rotation is expressed over strided halves. Over a ``(T, h, hd/2, 2)``
  view it is 7 contiguous kernels with no ``repeat_interleave`` and no ``cat``.

:func:`apply_rope_fused` is **bit-exact** against ``layers.apply_rope`` in fp32:
same products, same order, same accumulation — see ``tests/test_fast_path.py``.
It is not bit-exact if the tables are built in bf16, which is why
:func:`rope_tables` defaults to the dtype of the angles rather than of the
activations.
"""
from __future__ import annotations

import torch


def rope_tables(ang_t, ang_w, dtype=None):
    """-> (cos, sin), each ``(T, 1, hd/2)``, covering BOTH axial halves.

    ``layers.apply_rope`` rotates the first ``hd/2`` dims by ``ang_t`` and the
    second by ``ang_w``. The concatenated table does both in one pass.

    ``ang_w=None`` reproduces ``apply_rope``'s behaviour for a disabled axis
    EXACTLY: identity on the second half (cos 1, sin 0), i.e. those dims are
    left unrotated rather than reallocated to the live axis. That is a defect
    (see ``docs/REVIEW_FIELD.md`` §1.3) and it is preserved here deliberately —
    this module changes cost, not arithmetic.
    """
    if ang_w is None:
        c = torch.cat([torch.cos(ang_t), torch.ones_like(ang_t)], -1)
        s = torch.cat([torch.sin(ang_t), torch.zeros_like(ang_t)], -1)
    else:
        c = torch.cat([torch.cos(ang_t), torch.cos(ang_w)], -1)
        s = torch.cat([torch.sin(ang_t), torch.sin(ang_w)], -1)
    if dtype is not None:
        c, s = c.to(dtype), s.to(dtype)
    return c[:, None, :].contiguous(), s[:, None, :].contiguous()


def apply_rope_fused(x, cos, sin):
    """``x`` ``(T, h, hd)`` rotated by precomputed tables. Bit-exact in fp32.

    ``apply_rope`` computes, for pair ``i``::

        out[2i]   = v[2i]   * cos_i - v[2i+1] * sin_i
        out[2i+1] = v[2i+1] * cos_i + v[2i]   * sin_i

    which over a ``(..., hd/2, 2)`` view is exactly the two expressions below,
    in the same operand order.
    """
    T, h, hd = x.shape
    v = x.view(T, h, hd // 2, 2)
    a, b = v[..., 0], v[..., 1]
    return torch.stack([a * cos - b * sin, b * cos + a * sin], -1).view(T, h, hd)
