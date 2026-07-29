"""CoeffEvent — one event's sparse wavelet coefficients (detector-neutral).

The in-memory unit of the coeff corpus. Flat, columnar sparse rows across all
planes of one event: ``(band, plane_gid, wire, tau) -> value`` (RAW, un-normalized),
plus the per-(gid,band) threshold sigma and enough geometry (per-gid wire counts)
to invert. Provenance/basis lives in ``basis`` (see `provenance.BasisDescriptor`).

This supersedes the per-plane `SparseResult`: `from_sparse_results` flattens a
``{gid: SparseResult}`` dict into the columnar rows; `reconstruct_images` inverts
back to ``{gid: image}``. The flat shape mirrors the on-disk corpus (plane_gid a
column, no per-plane groups) and the model's flat-row consumption.

Coordinate convention: ``band`` indexes the wavedec band list ``[cA, cD_L, …, cD_1]``
(0 = approx); ``wire`` is the signal (row) index; ``tau`` the within-band coeff
index. The packed ``idx = wire*band_lengths[band] + tau`` is derived at tokenize,
never stored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helix.core.provenance import BasisDescriptor
from helix.core.wavelet import SparseResult, reconstruct, FlatBands


def _is_device(a) -> bool:
    return type(a).__module__.startswith("jax")


_COMPACT = None
_XFER_CAP = 1 << 16          # static transfer prefix; only ever grows
# Sized to the REAL kept-coefficient count (~55k/plane at 0.65% occupancy), not a
# guess. The prefix must be static so the slice never retraces, but an oversized
# one is paid on every transfer: at 1<<19 we shipped 524288 elements to recover
# ~55k (2.61 ms/plane); at 1<<16 the transfer is ~9x smaller. It grows
# monotonically, so a denser detector simply settles one or two rungs higher.


def _xfer_cap(n):
    """Static prefix length for the compacted transfer (monotonic, so the slice
    shape is stable and never retraces)."""
    global _XFER_CAP
    while n > _XFER_CAP:
        _XFER_CAP <<= 1
    return _XFER_CAP


def _compact_fn():
    """Jitted O(N) stream compaction: cumsum → scatter.

    The output is sized to the FULL band (not the nonzero count), so the jit
    specializes on the band SHAPE alone — ~10 compiles for a TPC event and never
    any more. Sizing it to the count instead (even rounded to a power of two)
    makes the shape vary per band *per event*, which recompiles constantly: that
    cost 9.8 s/event in a real 36-event build while a repeated-event profile
    showed 0.6 s and hid it entirely.
    """
    global _COMPACT
    if _COMPACT is None:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _c(flat):
            n = flat.size
            m = flat != 0
            pos = jnp.cumsum(m) - 1
            tgt = jnp.where(m, pos, n)                   # zeros scatter out of range
            idx = jnp.zeros(n, jnp.int32).at[tgt].set(
                jnp.arange(n, dtype=jnp.int32), mode="drop")
            val = jnp.zeros(n, jnp.float32).at[tgt].set(flat, mode="drop")
            return idx, val

        _COMPACT = _c
    return _COMPACT


def nonzero_rows(cband, n=None):
    """``(wire_idx, tau_idx, value)`` of the nonzeros of one band.

    On a device (jax) array the extraction runs ON DEVICE and only the sparse rows
    (~1% of the band) come back. Pulling the dense band to the host and running
    ``np.nonzero`` costs ~740 ms/event over the 51M dense coefficients and
    dominated the GPU build; ``jnp.nonzero`` is sort-based and still costs
    ~236 ms. This uses an O(N) cumsum+scatter compaction (verified identical to
    ``np.nonzero``), sized by band SHAPE so the jit never recompiles per event.
    """
    if not _is_device(cband):
        wi, ti = np.nonzero(cband)
        return wi.astype(np.int32), ti.astype(np.int32), cband[wi, ti].astype(np.float32)

    import jax.numpy as jnp
    if n is None:                                       # caller may batch this sync
        n = int(jnp.count_nonzero(cband))
    if n == 0:
        z = np.empty(0, np.int32)
        return z, z.copy(), np.empty(0, np.float32)
    idx, val = _compact_fn()(cband.ravel())
    # Transfer only a STATIC prefix, not the full band. The compaction output is
    # band-sized (that keeps its own shape static), but shipping all of it back
    # moves ~410 MB/event to recover ~6 MB of sparse rows. A monotonic cap keeps
    # the slice shape stable — `idx[:n]` with a per-event n would retrace, which
    # is the trap this whole path already hit twice.
    cap = _xfer_cap(n)
    flat = np.asarray(idx[:cap], np.int32)[:n]
    Lb = cband.shape[1]
    return (flat // Lb).astype(np.int32), (flat % Lb).astype(np.int32), \
        np.asarray(val[:cap], np.float32)[:n]


def _flat_rows(fb, band_lengths, gid):
    """Sparse rows of a whole plane from ONE flat compaction.

    ``fb.flat`` is ``(n_wires, sum(band_lengths))``; a flat position p maps to
    ``wire = p // C``, ``col = p % C``, then band/tau come from the static column
    offsets. One kernel per plane instead of one per band.
    """
    import jax.numpy as jnp
    flat = fb.flat
    C = int(flat.shape[-1])
    n = int(jnp.count_nonzero(flat))
    if n == 0:
        z = np.empty(0, np.int32)
        return np.empty(0, np.uint8), z, z.copy(), np.empty(0, np.float32)
    idx, val = _compact_fn()(flat.ravel())
    cap = _xfer_cap(n)
    pos = np.asarray(idx[:cap], np.int64)[:n]
    vals = np.asarray(val[:cap], np.float32)[:n]
    wire = (pos // C).astype(np.int32)
    col = (pos % C).astype(np.int64)
    offs = np.concatenate([[0], np.cumsum(np.asarray(band_lengths, np.int64))])
    band = (np.searchsorted(offs, col, side="right") - 1).astype(np.uint8)
    tau = (col - offs[band]).astype(np.int32)
    return band, wire, tau, vals


@dataclass
class CoeffEvent:
    # flat sparse coeff rows (one event, all planes) — n = total kept coeffs
    band: np.ndarray            # uint8   (n,)   band index into [cA, cD_L, …, cD_1]
    plane_gid: np.ndarray       # int32   (n,)   canonical plane id (v*3 + {U,V,Y})
    wire: np.ndarray            # int32   (n,)   signal/row index within the plane
    tau: np.ndarray             # int32   (n,)   within-band coeff index
    value: np.ndarray           # float32 (n,)   RAW coeff (un-normalized)
    # per-plane bookkeeping, indexed by position in `gids`
    gids: np.ndarray            # int32   (G,)   sorted unique plane gids in this shard's plane set
    n_wires: np.ndarray         # int32   (G,)   signals per plane (for reconstruct)
    sigma_threshold: np.ndarray  # float32 (G, n_bands)  per-(gid,band) threshold σ
    # provenance + identity
    basis: BasisDescriptor
    run: str = ""
    source_file: str = ""
    event: int = -1

    # ---- construction ---------------------------------------------------

    @classmethod
    def from_sparse_results(
        cls,
        results: dict[int, SparseResult],
        *,
        basis: BasisDescriptor,
        run: str = "",
        source_file: str = "",
        event: int = -1,
    ) -> "CoeffEvent":
        """Flatten ``{gid: SparseResult}`` (numpy band-list coeffs) into rows.

        Extracts the nonzero ``(band, wire, tau) -> value`` of every band, and
        records per-gid wire counts + per-band threshold σ. `basis.band_lengths`
        must match the results' band shapes.
        """
        gids = np.array(sorted(results), dtype=np.int32)
        n_bands = basis.n_bands
        b_l, p_l, w_l, t_l, v_l = [], [], [], [], []
        n_wires = np.zeros(len(gids), dtype=np.int32)
        sigma = np.zeros((len(gids), n_bands), dtype=np.float32)
        for gi, gid in enumerate(gids):
            res = results[int(gid)]
            coeffs = res.coeffs
            if not isinstance(coeffs, (list, FlatBands)):
                raise TypeError("from_sparse_results expects a band list or FlatBands")
            if len(coeffs) != n_bands:
                raise ValueError(
                    f"gid {gid}: {len(coeffs)} bands but basis has {n_bands}")
            nw = coeffs[0].shape[0]
            n_wires[gi] = nw
            # sigma_threshold FIRST — BOTH paths need it. (It used to sit after
            # the FlatBands branch, whose `continue` skipped it, so every jax-built
            # shard carried sigma_threshold == 0 and therefore norm_sigma == 0 —
            # silently destroying the normalization table the tokenizer needs.)
            if res.sigma_per_band is None:
                raise ValueError(f"gid {gid}: sigma_per_band is None (needed for sigma_threshold)")
            spb = np.asarray(res.sigma_per_band, dtype=np.float32).ravel()
            if spb.shape[0] != n_bands:
                raise ValueError(
                    f"gid {gid}: sigma_per_band has {spb.shape[0]} entries, expected {n_bands}")
            if not np.all(np.isfinite(spb)):
                raise ValueError(f"gid {gid}: sigma_per_band is non-finite ({spb})")
            sigma[gi, :] = spb
            if isinstance(coeffs, FlatBands):
                # The SAME per-band length check the list path does below. It is
                # load-bearing here, not redundant: _flat_rows derives band and
                # tau purely from basis.band_lengths and ignores fb.lens, so a
                # FlatBands whose lens disagree gets its coefficients silently
                # re-tagged (a marker at band 4 tau 0 was recorded as tau 8).
                # Reachable because event_coeff_event derives the event-wide basis
                # from ONE arbitrary plane and applies it to all of them.
                if tuple(int(x) for x in coeffs.lens) != tuple(int(x) for x in basis.band_lengths):
                    raise ValueError(
                        f"gid {gid}: FlatBands lens {tuple(int(x) for x in coeffs.lens)} != "
                        f"basis.band_lengths {tuple(int(x) for x in basis.band_lengths)} — "
                        f"band/tau would be mis-derived for every coefficient of this plane.")
                # ONE compaction for the whole plane instead of one per band, and
                # band/tau are derived from the flat column on the host.
                bb, ww, tt, vv = _flat_rows(coeffs, basis.band_lengths, gid)
                if bb.size:
                    b_l.append(bb); p_l.append(np.full(bb.size, gid, np.int32))
                    w_l.append(ww); t_l.append(tt); v_l.append(vv)
                continue
            # one stacked count for the whole plane (was one sync per band)
            if _is_device(coeffs[0]):
                import jax.numpy as _jnp
                nnz = [int(x) for x in np.asarray(
                    _jnp.stack([_jnp.count_nonzero(c) for c in coeffs]))]
            else:
                nnz = [None] * len(coeffs)
            for b, cband in enumerate(coeffs):
                if cband.shape[-1] != basis.band_lengths[b]:
                    raise ValueError(
                        f"gid {gid} band {b}: length {cband.shape[-1]} != "
                        f"band_lengths[{b}]={basis.band_lengths[b]}")
                if cband.shape[0] != nw:
                    raise ValueError(
                        f"gid {gid} band {b}: {cband.shape[0]} wires != band-0 count {nw}")
                wi, ti, vals = nonzero_rows(cband, nnz[b])   # device-side when jax
                if wi.size == 0:
                    continue
                b_l.append(np.full(wi.size, b, dtype=np.uint8))
                p_l.append(np.full(wi.size, gid, dtype=np.int32))    # int32: gid can exceed 255
                w_l.append(wi)
                t_l.append(ti)
                v_l.append(vals)

        def _cat(parts, dt):
            return np.concatenate(parts).astype(dt) if parts else np.empty(0, dtype=dt)

        return cls(
            band=_cat(b_l, np.uint8), plane_gid=_cat(p_l, np.int32),
            wire=_cat(w_l, np.int32), tau=_cat(t_l, np.int32),
            value=_cat(v_l, np.float32),
            gids=gids, n_wires=n_wires, sigma_threshold=sigma,
            basis=basis, run=run, source_file=source_file, event=event)

    # ---- inversion ------------------------------------------------------

    def to_band_lists(self) -> dict[int, list[np.ndarray]]:
        """Rebuild ``{gid: [cA, cD_L, …, cD_1]}`` dense band arrays (zeros filled)."""
        bl = self.basis.band_lengths
        out: dict[int, list[np.ndarray]] = {}
        for gi, gid in enumerate(self.gids):
            nw = int(self.n_wires[gi])
            out[int(gid)] = [np.zeros((nw, bl[b]), dtype=np.float32) for b in range(len(bl))]
        for i in range(self.value.shape[0]):
            gid = int(self.plane_gid[i])
            out[gid][int(self.band[i])][int(self.wire[i]), int(self.tau[i])] = self.value[i]
        return out

    def reconstruct_images(self, n_time: int | None = None) -> dict[int, np.ndarray]:
        """Inverse DWT per plane → ``{gid: (n_wires, n_time) image}`` (numpy backend)."""
        nt = self.basis.n_ticks_raw if n_time is None else n_time
        bands = self.to_band_lists()
        return {
            gid: reconstruct(
                SparseResult(coeffs=b, n_kept=0, n_total=0, sigma_per_band=None,
                             wavelet=self.basis.wavelet, level=self.basis.level,
                             mode=self.basis.mode),
                nt)
            for gid, b in bands.items()
        }

    @property
    def n_coeff(self) -> int:
        return int(self.value.shape[0])
