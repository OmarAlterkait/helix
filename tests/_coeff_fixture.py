"""A standalone writer of coefficient shards from per-event dicts, for reader tests: it writes
what helix.core.coeff_io writes, and also shards the codec would refuse (to test that the
reader refuses them)."""
import hashlib
import json

import h5py
import numpy as np

from helix.core.coeff_io import coord_digest

_STR = h5py.string_dtype(encoding="utf-8")


def basis_digest(*, wavelet, dwt_level, dwt_mode, n_ticks_raw, pad, band_lengths,
                 removal_json="{}", threshold_json="{}", sigma_norm=2.6) -> str:
    """sha256 of the canonical basis dict — mirrors ``BasisDescriptor.digest()``.

    Field NAMES are the descriptor's (``level``/``mode``), not the ``/config``
    attr names (``dwt_level``/``dwt_mode``). helix re-derives this on every read
    and refuses the shard on mismatch, so pimm-data computes it rather than
    trusting a caller-supplied string — a wrong digest produced a shard that
    pimm-data wrote happily and helix rejected. Pinned by the cross-repo golden.
    """
    d = dict(wavelet=str(wavelet), level=int(dwt_level), mode=str(dwt_mode),
             n_ticks_raw=int(n_ticks_raw), pad=int(pad),
             band_lengths=[int(x) for x in band_lengths],
             removal=json.loads(str(removal_json) or "{}"),
             threshold=json.loads(str(threshold_json) or "{}"),
             sigma_norm=float(sigma_norm))
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ds(grp, name, data, comp):
    kw = dict(compression=comp, compression_opts=4, shuffle=True) if comp == "gzip" else {}
    grp.create_dataset(name, data=data, **kw)


def write_coeff_shard(path, events, *, band_lengths, gids, n_wires, basis_attrs,
                      norm_sigma=None, dataset_name="", file_index=0,
                      global_event_offset=0, compression="gzip",
                      coords=True, noise=None, provenance=None):
    """Write a flat-columnar coeff shard (standalone; the layout helix reads).

    ``events`` is a list of per-event dicts with keys ``band, plane_gid, wire,
    tau, value, sigma_threshold, run, source_file, event``. All events share the
    plane set (``gids`` / ``n_wires``) and the basis. ``basis_attrs`` supplies the
    ``/config`` scalar attrs (wavelet, dwt_level, dwt_mode, n_ticks_raw, pad,
    sigma_norm, basis_digest, removal_json, threshold_json).

    ``coords=False`` writes a values-only shard for a co-supported target (see
    :func:`coord_digest`); ``noise`` records how the input was noised, so two
    shards built with different noise are distinguishable after the fact.

    ``provenance`` writes ``/config@provenance_json`` — the ``code`` block and the
    ``geom``/``spectrum`` hashes that :mod:`pimm_data.coeff_verify` compares
    across a corpus. This writer omitted it entirely, so no shard it produced
    could exercise those checks, and every fixture-based test of the verifier was
    silently testing the legacy path.
    """
    if not events:
        raise ValueError("write_coeff_shard: no events")

    # validate the basis contract so a pimm-written shard is not silently malformed
    # (helix's reader validates on read; fail here at write time instead).
    _REQUIRED = ("wavelet", "dwt_level", "dwt_mode", "n_ticks_raw", "pad", "sigma_norm")
    missing = [k for k in _REQUIRED if k not in basis_attrs]
    if missing:
        raise ValueError(f"basis_attrs missing required keys {missing}")
    # helix's reader re-hashes the basis and json-parses these on every read, so
    # anything it would reject must fail HERE — otherwise pimm-data happily writes
    # shards that the reference codec refuses, and the cross-repo contract is only
    # enforced in one direction.
    for k in ("removal_json", "threshold_json"):
        if k in basis_attrs:
            try:
                json.loads(str(basis_attrs[k]))
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"basis_attrs[{k!r}] is not parseable JSON: {e}") from e
    import pywt
    padded = int(basis_attrs["n_ticks_raw"]) + int(basis_attrs["pad"])
    want = tuple(int(c.shape[-1]) for c in pywt.wavedec(
        np.zeros(padded, np.float32), str(basis_attrs["wavelet"]),
        level=int(basis_attrs["dwt_level"]), mode=str(basis_attrs["dwt_mode"])))
    if tuple(int(x) for x in band_lengths) != want:
        raise ValueError(
            f"band_lengths {tuple(int(x) for x in band_lengths)} inconsistent with basis "
            f"({basis_attrs['wavelet']} L{basis_attrs['dwt_level']} @ {padded}) → expected {want}")

    dig = basis_digest(
        wavelet=basis_attrs["wavelet"], dwt_level=basis_attrs["dwt_level"],
        dwt_mode=basis_attrs["dwt_mode"], n_ticks_raw=basis_attrs["n_ticks_raw"],
        pad=basis_attrs["pad"], band_lengths=band_lengths,
        removal_json=basis_attrs.get("removal_json", "{}"),
        threshold_json=basis_attrs.get("threshold_json", "{}"),
        sigma_norm=basis_attrs["sigma_norm"])
    given = basis_attrs.get("basis_digest")
    if given is not None and str(given) != dig:
        raise ValueError(
            f"basis_digest {str(given)!r} does not match the basis it describes "
            f"(hashes to {dig!r}). helix's reader recomputes this and refuses the "
            f"shard, so writing it would produce a file only pimm-data can read.")
    basis_attrs = {**basis_attrs, "basis_digest": dig}

    def cat(key, dt):
        parts = [np.asarray(e[key], dt) for e in events]
        return np.concatenate(parts) if parts else np.empty(0, dt)

    band = cat("band", np.uint8)
    plane_gid = cat("plane_gid", np.int32)          # int32: gid can exceed 255 (matches helix)
    wire = cat("wire", np.int32)
    tau = cat("tau", np.int32)
    value = cat("value", np.float32)
    counts = np.array([len(e["value"]) for e in events], np.int64)
    event_offset = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    sigma = np.stack([np.asarray(e["sigma_threshold"], np.float32) for e in events])
    digests = np.array([coord_digest(e["band"], e["plane_gid"], e["wire"], e["tau"])
                        for e in events], np.uint64)

    with h5py.File(path, "w") as f:
        cfg = f.create_group("config")
        cfg.attrs["n_events"] = len(events)
        cfg.attrs["dataset_name"] = dataset_name
        cfg.attrs["file_index"] = file_index
        cfg.attrs["global_event_offset"] = global_event_offset
        cfg.attrs["readout_type"] = "wire"
        for k, v in basis_attrs.items():
            cfg.attrs[k] = v
        cfg.attrs["has_coords"] = bool(coords)
        cfg.attrs["noise_json"] = json.dumps(noise or {}, sort_keys=True)
        if provenance is not None:
            cfg.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
        cfg.create_dataset("band_lengths", data=np.asarray(band_lengths, np.int32))
        cfg.create_dataset("gids", data=np.asarray(gids, np.int32))
        cfg.create_dataset("n_wires", data=np.asarray(n_wires, np.int32))
        if norm_sigma is not None:
            cfg.create_dataset("norm_sigma", data=np.asarray(norm_sigma, np.float32))

        coord = f.create_group("coord")
        if coords:
            _ds(coord, "band", band, compression)
            _ds(coord, "plane_gid", plane_gid, compression)
            _ds(coord, "wire", wire, compression)
            _ds(coord, "tau", tau, compression)
        coord.create_dataset("event_offset", data=event_offset)
        coord.create_dataset("coord_digest", data=digests)
        _ds(coord, "sigma_threshold", sigma, compression)

        _ds(f, "value", value, compression)

        ident = f.create_group("ident")
        ident.create_dataset("run", data=np.array([e.get("run", "") for e in events], dtype=_STR))
        ident.create_dataset("source_file", data=np.array([e.get("source_file", "") for e in events], dtype=_STR))
        ident.create_dataset("event", data=np.array([e.get("event", -1) for e in events], np.int64))
