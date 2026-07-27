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
from helix.core.wavelet import SparseResult, reconstruct


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
            if not isinstance(coeffs, list):
                raise TypeError("from_sparse_results expects numpy band-list coeffs")
            if len(coeffs) != n_bands:
                raise ValueError(
                    f"gid {gid}: {len(coeffs)} bands but basis has {n_bands}")
            # Materialise device (jax) arrays to host ONCE per band. The
            # extraction below is numpy-side (np.nonzero + fancy indexing); left
            # on-device each of those pulls the whole band across PCIe again.
            coeffs = [np.asarray(c) for c in coeffs]
            nw = coeffs[0].shape[0]
            n_wires[gi] = nw
            if res.sigma_per_band is None:
                raise ValueError(f"gid {gid}: sigma_per_band is None (needed for sigma_threshold)")
            spb = np.asarray(res.sigma_per_band, dtype=np.float32).ravel()
            if spb.shape[0] != n_bands:
                raise ValueError(
                    f"gid {gid}: sigma_per_band has {spb.shape[0]} entries, expected {n_bands}")
            sigma[gi, :] = spb
            for b, cband in enumerate(coeffs):
                if cband.shape[-1] != basis.band_lengths[b]:
                    raise ValueError(
                        f"gid {gid} band {b}: length {cband.shape[-1]} != "
                        f"band_lengths[{b}]={basis.band_lengths[b]}")
                if cband.shape[0] != nw:
                    raise ValueError(
                        f"gid {gid} band {b}: {cband.shape[0]} wires != band-0 count {nw}")
                wi, ti = np.nonzero(cband)
                if wi.size == 0:
                    continue
                b_l.append(np.full(wi.size, b, dtype=np.uint8))
                p_l.append(np.full(wi.size, gid, dtype=np.int32))    # int32: gid can exceed 255
                w_l.append(wi.astype(np.int32))
                t_l.append(ti.astype(np.int32))
                v_l.append(cband[wi, ti].astype(np.float32))

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
