"""CoeffEvent — one event's sparse wavelet coefficients (detector-neutral), and its provenance.

The in-memory unit of the coeff corpus. Flat, columnar sparse rows across all
planes of one event: ``(band, plane_gid, wire, tau) -> value`` (RAW, un-normalized),
plus the per-(gid,band) threshold sigma and enough geometry (per-gid wire counts)
to invert. Provenance/basis lives in ``basis`` (:class:`BasisDescriptor`, below).

This supersedes the per-plane `SparseResult`: `from_sparse_results` flattens a
``{gid: SparseResult}`` dict into the columnar rows; `reconstruct_images` inverts
back to ``{gid: image}``. The flat shape mirrors the on-disk corpus (plane_gid a
column, no per-plane groups) and the model's flat-row consumption.

Coordinate convention: ``band`` indexes the wavedec band list ``[cA, cD_L, …, cD_1]``
(0 = approx); ``wire`` is the signal (row) index; ``tau`` the within-band coeff
index. The packed ``idx = wire*band_lengths[band] + tau`` is derived at tokenize,
never stored.

**Basis provenance** — a hashable descriptor of *how* a CoeffEvent was produced.
:class:`BasisDescriptor` captures everything that defines the coefficient basis and the
processing that filled it: the wavelet transform (wavelet/level/mode + the padded
length, from which the per-band lengths derive), the coherent-removal spec, the
threshold spec, and the tokenize normalization scalar. Its ``digest`` is a stable
sha256 over the canonical form — the join key stamped into a shard's ``/config``
so every event in the shard shares one basis and provenance is never duplicated
per event. This is decode + regeneration provenance in one small object; the on-disk
``/config`` stores its fields as attrs (once per shard).

**Code provenance** — which checkout produced something: :func:`describe_checkout` (a run's
``provenance.json``) and :func:`code_version` (every shard's ``code``), on one ``git`` call
helper.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from helix.core.wavelet import SparseResult, reconstruct


# ---- basis provenance ------------------------------------------------------------------------

def derive_band_lengths(wavelet: str, level: int, mode: str, padded_length: int) -> tuple[int, ...]:
    """Per-band lengths ``[len(cA), len(cD_L), …, len(cD_1)]`` for the basis.

    Derived by decomposing a zero vector of the padded length — the single
    source of truth the on-disk ``band_lengths`` is validated against on read.
    """
    import pywt

    z = np.zeros(padded_length, dtype=np.float32)
    coeffs = pywt.wavedec(z, wavelet, level=level, mode=mode)
    return tuple(int(c.shape[-1]) for c in coeffs)


def padded_length_of(band_lengths, wavelet: str, level: int, mode: str) -> int:
    """Effective DWT input length that produced ``band_lengths`` (via inverse).

    Mode-agnostic: recovers the padded length even when periodization pads
    internally, so the basis records a padded length consistent with the actual
    coefficients rather than assuming a specific pad convention.
    """
    import pywt

    z = [np.zeros((1, int(L)), dtype=np.float32) for L in band_lengths]
    return int(pywt.waverec(z, wavelet, mode=mode, axis=-1).shape[-1])


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def descriptor_digest(d: dict) -> str:
    """Stable sha256 over a dict, canonicalized (sorted keys, compact)."""
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class BasisDescriptor:
    """How a CoeffEvent was produced. Fields map 1:1 onto a shard's ``/config``."""

    wavelet: str = "coif3"
    level: int = 4
    mode: str = "periodization"
    n_ticks_raw: int = 4321
    pad: int = 15                                   # padded length = n_ticks_raw + pad
    band_lengths: tuple[int, ...] = ()              # PADDED per-band lengths (load-bearing)
    removal: dict = field(default_factory=dict)     # kind, kgate, ksig, npass, group_size
    threshold: dict = field(default_factory=dict)   # method, func, scale, per_band_sigma, ...
    sigma_norm: float = 2.6                          # SIGMA (tokenize normalization)

    @property
    def padded_length(self) -> int:
        return self.n_ticks_raw + self.pad

    @property
    def n_bands(self) -> int:
        return len(self.band_lengths)

    def with_derived_band_lengths(self) -> "BasisDescriptor":
        """Return a copy whose band_lengths are (re)derived from the basis."""
        bl = derive_band_lengths(self.wavelet, self.level, self.mode, self.padded_length)
        return BasisDescriptor(
            wavelet=self.wavelet, level=self.level, mode=self.mode,
            n_ticks_raw=self.n_ticks_raw, pad=self.pad, band_lengths=bl,
            removal=dict(self.removal), threshold=dict(self.threshold),
            sigma_norm=self.sigma_norm)

    def validate(self) -> None:
        """Fail loudly if band_lengths disagree with the generating basis."""
        want = derive_band_lengths(self.wavelet, self.level, self.mode, self.padded_length)
        got = tuple(int(x) for x in self.band_lengths)
        if got != want:
            raise ValueError(
                f"band_lengths {got} inconsistent with basis "
                f"({self.wavelet} L{self.level} {self.mode} @ {self.padded_length}) "
                f"→ expected {want}. A wrong Lb silently corrupts every (wire,tau).")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["band_lengths"] = [int(x) for x in self.band_lengths]
        return d

    def digest(self) -> str:
        return descriptor_digest(self.to_dict())


# ---- code provenance -------------------------------------------------------------------------

def _git(root, *args, timeout=10):
    """``git args`` run in ``root``: its stripped stdout, or None (no git, not a repo, timeout).
    Never raises and never blocks past ``timeout``."""
    import subprocess
    try:
        r = subprocess.run(("git",) + args, cwd=root, capture_output=True, text=True,
                           timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def describe_checkout(root):
    """``{root, commit, dirty, branch}`` for a git checkout, best-effort.

    Never raises and never blocks: a missing git, a tarball install with no
    ``.git``, or a slow filesystem all degrade to ``commit=None`` rather than
    taking down a training run at step 0.

    ``dirty`` is the field that matters. A commit hash alone says which code was
    COMMITTED, not which code ran; every long run in this project so far was
    launched from a working tree with uncommitted edits, so a hash without a
    dirty flag would have been quietly wrong.
    """
    import os

    out = {"root": root, "commit": None, "dirty": None, "branch": None}
    if not os.path.isdir(root):
        return out
    # Ask git where the repo is instead of testing for a `.git` DIRECTORY: a worktree's
    # `.git` is a FILE (a gitdir pointer), and a src-layout package resolves `root` to
    # `<repo>/src`, which is inside the repo but holds no `.git` at all.
    top = _git(root, "rev-parse", "--show-toplevel")
    if not top:
        return out
    out["root"] = top
    out["commit"] = _git(root, "rev-parse", "HEAD")
    out["branch"] = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    st = _git(root, "status", "--porcelain")
    if st is not None:
        out["dirty"] = bool(st)
    return out


def code_version():
    """Which helix produced this shard: ``{version, git, git_dirty}``.

    ``geom_sha256``/``spectrum_sha256`` pin the DSP's external INPUTS, and
    ``basis_digest`` pins the declared basis — but nothing recorded the CODE, so
    two shards built by different trees compared as identical (run_0027575715 was
    built by two working trees a few commits apart, and no field could show it).

    ATOMIC: both git fields or neither. A commit recorded with no dirtiness (a timeout on the
    second call) serialised differently from its neighbours, and coeff_verify reported
    "shards were built by DIFFERENT helix versions" for shards of ONE commit (a 32-way array
    all running `git status` against one NFS checkout).
    """
    import os

    import helix
    out = {"version": getattr(helix, "__version__", "unknown")}
    root = os.path.dirname(os.path.dirname(os.path.abspath(helix.__file__)))
    commit = _git(root, "rev-parse", "--short", "HEAD", timeout=30)
    if commit is not None:
        dirty = _git(root, "status", "--porcelain", timeout=30)
        if dirty is not None:
            out["git"] = commit
            out["git_dirty"] = bool(dirty)
    return out


# ---- the event -------------------------------------------------------------------------------

def nonzero_rows(cband):
    """``(wire_idx, tau_idx, value)`` of the nonzeros of one band (numpy or torch).

    A torch tensor is compacted where it lives (``torch.nonzero``) and only the
    sparse rows (~1% of the band) come back to the host.
    """
    if type(cband).__module__.startswith("torch"):
        import torch
        nz = torch.nonzero(cband, as_tuple=True)
        vals = cband[nz]
        return (nz[0].to(torch.int32).cpu().numpy(),
                nz[1].to(torch.int32).cpu().numpy(),
                vals.to(torch.float32).cpu().numpy())
    wi, ti = np.nonzero(cband)
    return wi.astype(np.int32), ti.astype(np.int32), cband[wi, ti].astype(np.float32)


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
    # per-wire validity (G, max(n_wires)) bool, False = excluded channel (dead, noisy,
    # not instrumented): no input and no target for the model. None = all valid.
    wire_valid: np.ndarray | None = None

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
                raise TypeError("from_sparse_results expects a band list")
            if len(coeffs) != n_bands:
                raise ValueError(
                    f"gid {gid}: {len(coeffs)} bands but basis has {n_bands}")
            nw = coeffs[0].shape[0]
            n_wires[gi] = nw
            # sigma_threshold FIRST, before any per-band early exit. (It once sat
            # after a branch whose `continue` skipped it, so those shards carried
            # sigma_threshold == 0 and therefore norm_sigma == 0 — silently
            # destroying the normalization table the tokenizer needs.)
            if res.sigma_per_band is None:
                raise ValueError(f"gid {gid}: sigma_per_band is None (needed for sigma_threshold)")
            spb = np.asarray(res.sigma_per_band, dtype=np.float32).ravel()
            if spb.shape[0] != n_bands:
                raise ValueError(
                    f"gid {gid}: sigma_per_band has {spb.shape[0]} entries, expected {n_bands}")
            if not np.all(np.isfinite(spb)):
                raise ValueError(f"gid {gid}: sigma_per_band is non-finite ({spb})")
            sigma[gi, :] = spb
            for b, cband in enumerate(coeffs):
                if cband.shape[-1] != basis.band_lengths[b]:
                    raise ValueError(
                        f"gid {gid} band {b}: length {cband.shape[-1]} != "
                        f"band_lengths[{b}]={basis.band_lengths[b]}")
                if cband.shape[0] != nw:
                    raise ValueError(
                        f"gid {gid} band {b}: {cband.shape[0]} wires != band-0 count {nw}")
                wi, ti, vals = nonzero_rows(cband)
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
        """Inverse DWT per plane → ``{gid: (n_wires, n_time) image}`` (numpy arrays)."""
        nt = self.basis.n_ticks_raw if n_time is None else n_time
        return {gid: reconstruct(b, self.basis.wavelet, nt).cpu().numpy()
                for gid, b in self.to_band_lists().items()}

    @property
    def n_coeff(self) -> int:
        return int(self.value.shape[0])
