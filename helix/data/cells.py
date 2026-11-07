"""The fine cell grid: simulation truth on it, one ``{name}_cells_<k>.h5`` per coefficient shard,
and the join from a cell to the tokens covering it.

The label every downstream task scores against -- the denoiser's target, the detection floor,
the along-wire probe and the particle labels -- built once from the simulation's ``hits`` (and
``step``, for deposited energy and 3D position, and pysupera's ``labl``, for each track's class),
separately from the corpus build: labels change more often than the corpus. Same events, same order as ``{name}_coeff_<k>.h5``; joined to it by identity
(``/ident``) as the dataset modality ``cells`` (``CoeffTPCDataset(modalities=("coeff",
"cells"))``). Pretraining never opens it (tests/test_noisy_training.py).

Grid: ``FW`` = 2 wires x ``FT`` = 16 raw ticks, finer than any patch under test, so models
with different tokenizers are scored on the same cells. Every quantity the evaluation uses is
decided per cell (every window and padding box is aligned to the grid) except two sub-cell
ones, stored at their own granularity: each particle's charge centroid per (plane, track),
and each cell's dominant hits group's 3D centroid (the along-wire probe's label).

Layout (four tables, each a per-event CSR with its own ``event_offset``)::

    /config       attrs: format ("helix-cells-3"), n_events, grid_fw, grid_ft, key,
                         label_json, provenance_json
    /cells        key int64 (fkey), q float32,                one row per cell, sorted by key
                  dom_frac float32, dom_y float32,            the cell's largest hits group:
                  dom_z float32                               its share of the cell's charge
                                                              and its deposits' charge-weighted
                                                              (y, z), mm
    /cell_track   cell int32, track int32, q float32          the cells' charge split by track
                                                              (rows of /cells, /tracks)
    /plane_track  plane_gid int32, track int32,               per (plane, track): its PIXEL
                  wire float64, tick float64                  charge-weighted centroid
    /tracks       volume int16, track_id int32, energy f32,   hits group_to_track ids, per volume;
                  motif int8, pid int8                        energy = step de summed, MeV; the
                                                              track's classes (MOTIF_NAMES,
                                                              PID_NAMES; -1 none) from labl
    /ident        run, source_file, event                     copied from the coeff shard

A per-track label is a column of ``/tracks``; a cell takes its dominant track's
(:func:`cell_class`). The classes follow pimm-data's walk (``segment_motif`` = group -> fragment
-> motif, ``segment_pid`` = group -> fragment -> instance -> pid) at the track: a track lies in
one fragment for all but ~1 in 10^6 (labl lists those groups as overrides, not read here).

``Q0`` is the charge scale every per-cell charge target and prediction is expressed in,
``log1p(q / Q0)``; this is its one definition.

numpy + h5py only (no pimm-data): ``helix.data.dataset.CellsReader`` serves it to the
dataset, everything else reads :func:`read_event` directly.
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np

FORMAT = "helix-cells-3"

FW, FT = 2, 16                       # fine cell, wires x raw ticks
_TOFF = 4096                         # fine-tick offset inside the packed key

#: The charge scale of every per-cell charge target and prediction: ``y = log1p(q / Q0)``.
#: The median positive cell charge (e-) over the probe-split training events of the first
#: denoising truth; a constant, so a prediction means the same number whatever the events.
Q0 = 2043.2578108909702

#: Plane order within a volume; ``plane_gid = volume * 3 + PLANES.index(p)``.
PLANES = ("U", "V", "Y")

#: (h5 group, column, record name, dtype). The record names are those of :func:`cell_truth`,
#: plus the track table's.
SCHEMA = (
    ("cells", "key", "key", np.int64),
    ("cells", "q", "q", np.float32),
    ("cells", "dom_frac", "dom_frac", np.float32),
    ("cells", "dom_y", "dom_y", np.float32),
    ("cells", "dom_z", "dom_z", np.float32),
    ("cell_track", "cell", "ct_cell", np.int32),
    ("cell_track", "track", "ct_track", np.int32),
    ("cell_track", "q", "ct_q", np.float32),
    ("plane_track", "plane_gid", "pt_gid", np.int32),
    ("plane_track", "track", "pt_track", np.int32),
    ("plane_track", "wire", "pt_wire", np.float64),
    ("plane_track", "tick", "pt_tick", np.float64),
    ("tracks", "volume", "trk_volume", np.int16),
    ("tracks", "track_id", "trk_id", np.int32),
    ("tracks", "energy", "trk_energy", np.float32),
    ("tracks", "motif", "trk_motif", np.int8),
    ("tracks", "pid", "trk_pid", np.int8),
)
GROUPS = tuple(dict.fromkeys(g for g, *_ in SCHEMA))

LABEL = dict(
    source="hits",
    charge="hits charges_u16 / 65535 * peak_charges, summed over every sample of every group in the cell (e-)",
    track="hits group_to_track[group], numbered per volume",
    energy="step de summed over the track's deposits (hits deposit_to_group), MeV",
    centroid="per (plane, track): charge-weighted pixel wire and tick",
    dominant="per cell: the hits group carrying most of its charge; dom_frac its share, (dom_y, dom_z) "
             "its deposits' step-charge-weighted position (mm: step positions * pos_step_mm + pos_origin)",
    motif="per track: labl track.fragment -> particle.fragment_sem_type, as MOTIF_NAMES; -1 unowned",
    pid="per track: labl track.fragment -> particle.fragment_instance -> (pdg, instance_sem_type), "
        "as PID_NAMES; -1 unowned or another PDG",
)

#: The label classes, pimm-data's PILArNet vocabulary (pimm_data/_resolve.py, docs/LABELS.md):
#: pysupera SemanticType (1 shower, 2 track, 3 delta, 4 michel, 5 le_scatter) -> motif; a
#: particle's sign-blind PDG -> pid, a low-energy instance -> le.
MOTIF_NAMES = ("shower", "track", "michel", "delta", "le")
PID_NAMES = ("photon", "electron", "muon", "pion", "proton", "le")
_MOTIF_OF_SEM = np.array([-1, 0, 1, 3, 2, 4, -1], np.int64)
_PID_OF_PDG = {22: 0, 11: 1, 13: 2, 211: 3, 2212: 4}
_SEM_LE = 5
_STR = h5py.string_dtype(encoding="utf-8")


# ------------------------------------------------------------------------------------------- grid
def fkey(g, fw, ft):
    """Pack (plane_gid, fine wire, fine tick) into one int64. Coordinates must be >= 0 (wire) /
    > -4096 (tick): a negative wire sets the plane bits."""
    return (np.asarray(g, np.int64) << 40) | (np.asarray(fw, np.int64) << 20) \
        | (np.asarray(ft, np.int64) + _TOFF)


def unkey(k):
    k = np.asarray(k, np.int64)
    return k >> 40, (k >> 20) & 0xFFFFF, (k & 0xFFFFF) - _TOFF


def _coeff_keys(band, plane_gid, wire, tau, cfg):
    """``(k, key)``: which coefficients are in the tokenized bands, and their fine cells."""
    from helix.model.tokenize import tick_of_tau
    band, plane_gid, wire, tau = (np.asarray(x) for x in (band, plane_gid, wire, tau))
    k = band < cfg.n_bands
    ct = tick_of_tau(tau[k], plane_gid[k], band[k], cfg)
    return k, fkey(plane_gid[k], wire[k] // FW, np.floor(ct / FT).astype(np.int64))


def coeff_cells(band, plane_gid, wire, tau, cfg):
    """Fine-cell keys (sorted, unique) of an event's kept coefficients in the tokenized bands."""
    return np.unique(_coeff_keys(band, plane_gid, wire, tau, cfg)[1])


def coeff_amplitude(c, cfg):
    """``(keys, amp)``: :func:`coeff_cells` of a coeff part ``c`` (``band, plane_gid, wire, tau,
    value`` and its ``_meta``) and, per cell, the largest ``|value| / sigma`` of the kept
    coefficients in it -- the compressor's own detection statistic, in the units the tokenizer
    normalises by (:func:`helix.model.tokenize.sigma_for_rows`)."""
    from helix.model.tokenize import sigma_for_rows
    k, key = _coeff_keys(c["band"], c["plane_gid"], c["wire"], c["tau"], cfg)
    m = c["_meta"]
    band, g = np.asarray(c["band"])[k], np.asarray(c["plane_gid"])[k]
    a = np.abs(np.asarray(c["value"], np.float64).reshape(-1)[k]) \
        / np.maximum(sigma_for_rows(g, band, m["gids"], m["norm_sigma"]), 1e-6)
    keys, inv = np.unique(key, return_inverse=True)
    amp = np.zeros(len(keys))
    np.maximum.at(amp, inv.ravel(), a)
    return keys, amp


def _fourier(x):
    ang = 2 * np.pi * x[..., None] * np.array([1, 2, 3], np.float32)
    return np.concatenate([np.sin(ang), np.cos(ang)], -1).reshape(x.shape[0], -1)


def cell_inputs(keys, cell_key, band_lengths, cfg):
    """Per fine cell: which token covers it in each band, and where inside it.

    ``keys``: fine-cell keys (:func:`fkey`). ``cell_key``: the tokenized event's per-token cell
    keys (``B["cell_key"]``). Returns ``idx`` (n, n_bands) int64 -- the covering token's row,
    -1 if none -- and ``aux`` (n, 13 * n_bands) float32: a presence bit per band, then the cell's
    continuous offset inside each band's token (wire, tau), Fourier-encoded. The frozen probes
    (:mod:`helix.eval`) and the trainable denoising head (:mod:`helix.model.denoise`) both read
    exactly this, so a frozen and a fine-tuned number differ only in what was trained.
    """
    from helix.model.tokenize import pixel_cells

    nb, pw, pt = cfg.n_bands, cfg.pw, cfg.pt
    g, fw, ft = unkey(np.asarray(keys, np.int64))
    w, t = fw * FW, ft * FT + FT // 2
    pc = pixel_cells(g, w, t, band_lengths, cfg)
    ck = np.asarray(cell_key, np.int64)
    order = np.argsort(ck); cks = ck[order]
    if len(cks):
        pos = np.clip(np.searchsorted(cks, pc), 0, len(cks) - 1)
        idx = np.where(cks[pos] == pc, order[pos], -1)
    else:
        idx = np.full(pc.shape, -1, np.int64)
    dec = (1 << np.asarray(cfg.lev)).astype(np.float64)
    toff = np.asarray(cfg.toff)[g % 3]
    offs = []
    for b in range(nb):
        tau = (t + toff) / dec[b] - cfg.delta[b]
        offs += [(w % pw + 0.5) / pw, (tau / pt) % 1.0]
    aux = np.concatenate([(idx >= 0).astype(np.float32), _fourier(np.stack(offs, 1).astype(np.float32))], 1)
    return idx.astype(np.int64), aux.astype(np.float32)


def cell_neighbors(keys, cell_key, band_lengths, cfg, radius=1):
    """Per fine cell and band: the tokens of the (2r+1) x (2r+1) patches around the one covering
    it, -> (n, n_bands, (2r+1)**2) int64, -1 where no token.

    Found by shifting the covering PATCH's block coordinates, so a cell with no token of its own
    in a band still sees the tokens beside it. Slot ``(2r+1)**2 // 2`` is the covering token, the
    same index :func:`cell_inputs` returns. Order: wire offset major, tick offset minor, each from
    -r to r.
    """
    from helix.model.tokenize import cell_key as pack, pixel_cells, unpack_cell_key

    g, fw, ft = unkey(np.asarray(keys, np.int64))
    pc = pixel_cells(g, fw * FW, ft * FT + FT // 2, band_lengths, cfg)          # (n, nb)
    pg, pb, wb, tb = unpack_cell_key(pc.ravel())
    ck = np.asarray(cell_key, np.int64)
    order = np.argsort(ck); cks = ck[order]
    offs = [(dw, dt) for dw in range(-radius, radius + 1) for dt in range(-radius, radius + 1)]
    out = np.full((pc.size, len(offs)), -1, np.int64)
    for j, (dw, dt) in enumerate(offs):
        w2, t2 = wb + dw, tb + dt
        ok = (w2 >= 0) & (t2 >= 0)
        if not ok.any() or not len(cks):
            continue
        k2 = pack(pg[ok], pb[ok], w2[ok] * cfg.pw, t2[ok] * cfg.pt, cfg)
        pos = np.clip(np.searchsorted(cks, k2), 0, len(cks) - 1)
        out[np.nonzero(ok)[0], j] = np.where(cks[pos] == k2, order[pos], -1)
    return out.reshape(pc.shape[0], pc.shape[1], len(offs))


# ------------------------------------------------------------------------------------------ truth
def decode_hits_plane(grp):
    """One ``hits`` plane group -> flat per-sample ``(wire, tick, q, group)``.

    CSR-style: each group owns ``group_sizes[i]`` consecutive samples, stored as offsets from its
    centre. ``charges_u16`` is quantised against the group's own peak, so it dequantises as
    ``q / 65535 * peak_charges[group]``.
    """
    cw = grp["center_wires"][:].astype(np.int64)
    ct = grp["center_times"][:].astype(np.int64)
    dw = grp["delta_wires"][:].astype(np.int64)
    dt = grp["delta_times"][:].astype(np.int64)
    qs = grp["charges_u16"][:].astype(np.float64)
    gid = grp["group_ids"][:].astype(np.int64)
    sz = grp["group_sizes"][:].astype(np.int64)
    pk = grp["peak_charges"][:].astype(np.float64)
    if sz.sum() != qs.shape[0]:
        raise ValueError(f"hits plane is inconsistent: sum(group_sizes)={sz.sum()} but "
                         f"{qs.shape[0]} charge samples")
    rep = np.repeat(np.arange(len(sz)), sz)          # sample -> its group's row
    return dict(wire=cw[rep] + dw, tick=ct[rep] + dt, q=qs / 65535.0 * pk[rep], group=gid[rep])


def group_centroids(pos, q, deposit_to_group, n_groups):
    """``(n_groups, 3)`` charge-weighted deposit position per group, NaN for a group with no
    deposit; a group whose deposits carry no charge falls back to their unweighted mean."""
    pos, q = np.asarray(pos, np.float64), np.asarray(q, np.float64)
    d2g = np.asarray(deposit_to_group, np.int64)
    if d2g.shape[0] != pos.shape[0]:
        raise ValueError(f"deposit count mismatch: deposit_to_group has {d2g.shape[0]} entries, "
                         f"step has {pos.shape[0]} positions")
    w = np.bincount(d2g, weights=q, minlength=n_groups)
    n = np.bincount(d2g, minlength=n_groups).astype(np.float64)
    out = np.full((n_groups, 3), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for a in range(3):
            sw = np.bincount(d2g, weights=q * pos[:, a], minlength=n_groups)
            sp = np.bincount(d2g, weights=pos[:, a], minlength=n_groups)
            out[:, a] = np.where(w > 1e-12, sw / np.maximum(w, 1e-12), np.where(n > 0, sp / np.maximum(n, 1), np.nan))
    return out


def cell_truth(g, w, t, q, trk, grp, grp_yz):
    """Per-pixel-sample truth -> the cell record (:data:`SCHEMA` names, without the track table).

    ``g`` (plane_gid), ``w``, ``t``, ``q``, ``trk`` (any integer particle id >= 0) and ``grp``
    (any integer hits group id, unique within the event) per sample; ``grp_yz``: per sample, its
    group's (y, z) centroid (mm). Returns

      key, q                     every fine cell a sample falls in (sorted) and its charge,
                                 summed in sample order (float64)
      dom_frac, dom_y, dom_z     the cell's largest group: its share of the cell's charge and
                                 its (y, z)
      ct_cell, ct_track, ct_q    the cells' charge split by particle: row of ``key``, ``trk``
                                 value, charge; sorted by (cell, track)
      pt_gid, pt_track,          per (plane, particle), sorted: its charge-weighted PIXEL
      pt_wire, pt_tick           centroid (wire, tick)
    """
    G, W, T = (np.asarray(x, np.int64) for x in (g, w, t))
    Q, TRK, GRP = np.asarray(q, np.float64), np.asarray(trk, np.int64), np.asarray(grp, np.int64)
    YZ = np.asarray(grp_yz, np.float64).reshape(-1, 2)
    key, inv = np.unique(fkey(G, W // FW, T // FT), return_inverse=True)
    inv = inv.astype(np.int64).ravel()
    qc = np.bincount(inv, weights=Q, minlength=len(key))
    tu, tinv = np.unique(TRK, return_inverse=True)
    nt = max(len(tu), 1)
    cu, cinv = np.unique(inv * nt + tinv.ravel(), return_inverse=True)
    pu, pinv = np.unique(G * nt + tinv.ravel(), return_inverse=True)
    pq = np.bincount(pinv.ravel(), weights=Q)
    # the dominant group per cell: charge per (cell, group); the largest is written last
    gu, ginv = np.unique(GRP, return_inverse=True)
    ng = max(len(gu), 1)
    cg, cginv = np.unique(inv * ng + ginv.ravel(), return_inverse=True)
    cginv = cginv.ravel()
    cgq = np.bincount(cginv, weights=Q)
    cell_of = cg // ng
    order = np.lexsort((cgq, cell_of))
    top = np.zeros(len(key), np.int64)
    top[cell_of[order]] = order
    first = np.zeros(len(cg), np.int64)
    first[cginv[::-1]] = np.arange(len(Q))[::-1]                     # one sample of each (cell, group)
    dyz = YZ[first[top]] if len(key) else np.zeros((0, 2))
    return dict(key=key, q=qc,
                dom_frac=cgq[top] / np.maximum(qc, 1e-12) if len(key) else np.zeros(0),
                dom_y=dyz[:, 0], dom_z=dyz[:, 1],
                ct_cell=cu // nt, ct_track=tu[cu % nt], ct_q=np.bincount(cinv.ravel(), weights=Q),
                pt_gid=pu // nt, pt_track=tu[pu % nt],
                pt_wire=np.bincount(pinv.ravel(), weights=Q * W) / np.maximum(pq, 1e-12),
                pt_tick=np.bincount(pinv.ravel(), weights=Q * T) / np.maximum(pq, 1e-12))


def track_classes(labl_ev, track_id):
    """Per Geant4 ``track_id``: its ``(motif, pid)`` class from a labl entry (pimm-data
    ``supera_labl``: ``track/{id, fragment}``, ``particle/...``); -1 for a track the entry does
    not list or that no fragment owns."""
    tid, frag = labl_ev["track"]["id"][:].astype(np.int64), labl_ev["track"]["fragment"][:].astype(np.int64)
    part = labl_ev["particle"]
    sem = part["fragment_sem_type"][:].astype(np.int64)
    inst = part["fragment_instance"][:].astype(np.int64)
    pdg, isem = np.abs(part["pdg"][:].astype(np.int64)), part["instance_sem_type"][:].astype(np.int64)
    track_id = np.asarray(track_id, np.int64)
    f = np.full(len(track_id), -1, np.int64)
    if len(tid):
        p = np.clip(np.searchsorted(tid, track_id), 0, len(tid) - 1)
        f = np.where(tid[p] == track_id, frag[p], -1)

    def take(col, rows):                                       # col[rows], -1 where rows is -1
        return np.where(rows >= 0, col[np.maximum(rows, 0)], -1) if len(col) else np.full(len(rows), -1)

    s_ = take(sem, f)
    motif = np.where((s_ >= 0) & (s_ < len(_MOTIF_OF_SEM)), _MOTIF_OF_SEM[np.clip(s_, 0, len(_MOTIF_OF_SEM) - 1)], -1)
    i = take(inst, f)
    pid = np.full(len(track_id), -1, np.int64)
    ap, ise = take(pdg, i), take(isem, i)
    for code, cls in _PID_OF_PDG.items():
        pid[ap == code] = cls
    pid[(i >= 0) & (ise == _SEM_LE)] = PID_NAMES.index("le")
    return motif, pid


def cell_class(rec, col):
    """Per cell of a record: the class (``col``: ``"trk_motif"`` / ``"trk_pid"``) of the track
    carrying most of its charge; -1 if that track has none."""
    c, t, q = (np.asarray(rec[k]) for k in ("ct_cell", "ct_track", "ct_q"))
    out = np.full(len(rec["key"]), -1, np.int64)
    o = np.lexsort((q, c))                                     # per cell, the largest share last
    out[c[o]] = np.asarray(rec[col], np.int64)[t[o]]
    return out


def require_format(cfg, where=""):
    """Refuse a file that is not :data:`FORMAT` on this grid."""
    got = cfg.attrs.get("format")
    got = got.decode() if isinstance(got, bytes) else got
    if got != FORMAT:
        raise ValueError(f"{where or 'cells shard'}: format {got!r}, not {FORMAT!r}; rebuild it (scripts/build_truth.py)")
    if (int(cfg.attrs["grid_fw"]), int(cfg.attrs["grid_ft"])) != (FW, FT):
        raise ValueError(f"{where}: grid {cfg.attrs['grid_fw']} x {cfg.attrs['grid_ft']}, helix scores on {FW} x {FT}")


def cells_path(coeff_shard, out_dir=None):
    """``{name}_coeff_<k>.h5`` -> ``{name}_cells_<k>.h5`` (beside it, or in ``out_dir``)."""
    d, b = os.path.split(str(coeff_shard))
    stem, k = b[:-3].rsplit("_coeff_", 1)
    if not k.isdigit():
        raise ValueError(f"{coeff_shard}: not a {{name}}_coeff_<k>.h5 shard")
    return os.path.join(out_dir or d, f"{stem}_cells_{k}.h5")


# ------------------------------------------------------------------------------------------ build
def step_positions(sv):
    """A ``step`` volume's deposit positions (mm): ``positions`` (uint16) * ``pos_step_mm`` +
    ``pos_origin_{x,y,z}``, as JAXTPC writes them (``production/save.py``, ``load_event_step``)."""
    origin = np.array([sv.attrs[f"pos_origin_{a}"] for a in "xyz"], np.float64)
    return sv["positions"][:].astype(np.float64) * float(sv.attrs["pos_step_mm"]) + origin


def event_record(hits_ev, step_ev, labl_ev):
    """One simulation event (its ``hits``, ``step`` and ``labl`` entry groups) -> a record
    (:data:`SCHEMA` names).

    Samples are concatenated volume by volume (sorted by name), plane by plane, in stored order,
    so each cell's charge is summed in that order.
    """
    G, W, T, Q, TRK, GRP, YZ, vol, tid, energy = [], [], [], [], [], [], [], [], [], []
    base, gbase = 0, 0
    for vk in sorted(x for x in hits_ev if x.startswith("volume_")):
        vi, hv, sv = int(vk.split("_")[1]), hits_ev[vk], step_ev[vk]
        if int(hv.attrs["n_actual"]) == 0:
            # Nothing deposited in this volume: the simulator writes its planes with no entries
            # and no per-deposit arrays (deposit_to_group, step de, ...). It adds no cell, no track.
            if int(sv.attrs["n_actual"]) != 0 or any(int(hv[pl].attrs["n_entries"]) for pl in PLANES if pl in hv):
                raise ValueError(f"{vk}: hits n_actual is 0 but step or a plane has entries")
            gbase += len(hv["group_to_track"])
            continue
        d2g = hv["deposit_to_group"][:].astype(np.int64)
        g2t = hv["group_to_track"][:].astype(np.int64)
        de = sv["de"][:].astype(np.float64)
        if len(de) != len(d2g):
            raise ValueError(f"{vk}: step has {len(de)} deposits, hits deposit_to_group {len(d2g)}")
        cen = group_centroids(step_positions(sv), sv["charge"][:], d2g, len(g2t))[:, 1:]          # (y, z)
        planes = [(pi, decode_hits_plane(hv[pl])) for pi, pl in enumerate(PLANES) if pl in hv]
        tv = np.unique(np.concatenate([g2t[d2g]] + [g2t[s["group"]] for _, s in planes]))
        energy.append(np.bincount(np.searchsorted(tv, g2t[d2g]), weights=de, minlength=len(tv)))
        vol.append(np.full(len(tv), vi)); tid.append(tv)
        for pi, s in planes:
            G.append(np.full(len(s["q"]), vi * 3 + pi)); W.append(s["wire"]); T.append(s["tick"])
            Q.append(np.asarray(s["q"], np.float64)); TRK.append(base + np.searchsorted(tv, g2t[s["group"]]))
            GRP.append(gbase + s["group"]); YZ.append(cen[s["group"]])
        base += len(tv)
        gbase += len(g2t)
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int64)
    rec = cell_truth(cat(G), cat(W), cat(T), cat(Q), cat(TRK), cat(GRP),
                     np.concatenate(YZ) if YZ else np.zeros((0, 2)))
    motif, pid = track_classes(labl_ev, cat(tid))
    rec.update(trk_volume=cat(vol), trk_id=cat(tid), trk_energy=cat(energy), trk_motif=motif, trk_pid=pid)
    return {name: np.asarray(rec[name]).astype(dt) for _, _, name, dt in SCHEMA}


def write_shard(path, ident, records, *, label, provenance):
    """Write ``records`` (one per event of ``ident``) to ``path``, whole or not at all."""
    if len(records) != len(ident["event"]):
        raise ValueError(f"{len(records)} records for {len(ident['event'])} events")
    part = str(path) + ".part"
    with h5py.File(part, "w") as f:
        cfg = f.create_group("config")
        cfg.attrs.update(format=FORMAT, n_events=len(records), grid_fw=FW, grid_ft=FT,
                         key="fkey(plane_gid, wire // grid_fw, tick // grid_ft) = g << 40 | fw << 20 | (ft + 4096)",
                         label_json=json.dumps(label, sort_keys=True),
                         provenance_json=json.dumps(provenance, sort_keys=True))
        for grp in GROUPS:
            cols = [(col, name, dt) for g, col, name, dt in SCHEMA if g == grp]
            n = [len(r[cols[0][1]]) for r in records]
            for r in records:
                if any(len(r[name]) != len(r[cols[0][1]]) for _, name, _ in cols):
                    raise ValueError(f"/{grp}: columns of unequal length")
            h = f.create_group(grp)
            h.create_dataset("event_offset", data=np.concatenate([[0], np.cumsum(n)]).astype(np.int64))
            for col, name, dt in cols:
                data = np.concatenate([np.asarray(r[name], dt) for r in records]) if records else np.zeros(0, dt)
                h.create_dataset(col, data=data, compression="gzip", compression_opts=4, shuffle=True)
        idg = f.create_group("ident")
        idg.create_dataset("run", data=np.array(ident["run"], dtype=_STR))
        idg.create_dataset("source_file", data=np.array(ident["source_file"], dtype=_STR))
        idg.create_dataset("event", data=np.asarray(ident["event"], np.int64))
    os.replace(part, path)


def _text(col):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in col]


def read_ident(f):
    """``/ident`` of an open coeff or cells file as ``{run, source_file, event}`` lists/arrays."""
    return dict(run=_text(f["ident"]["run"][:]), source_file=_text(f["ident"]["source_file"][:]),
                event=f["ident"]["event"][:].astype(np.int64))


def labl_path(hits_file):
    """The pysupera ``labl`` file (pimm-data ``supera_labl``) of a task, beside its ``hits``:
    ``<task>/labl/{name}_labl_<task>.h5``."""
    p = os.path.abspath(str(hits_file))
    task = os.path.dirname(os.path.dirname(p))
    return os.path.join(task, "labl", os.path.basename(p).replace("_hits_", "_labl_"))


def build_shard(coeff_shard, source_root, out_dir=None):
    """Write the cells shard of ``coeff_shard``; ``(path, message)``. An existing one is kept.

    Events come from the coeff shard's ``/ident`` (any number of source files): ``source_file`` of
    run (campaign group) ``run`` under the campaign root ``source_root``
    (:func:`helix.tpc.io.sensor_path`), whose task's ``hits``, ``step`` and ``labl``
    (:func:`labl_path`) files hold campaign event ``event``.
    """
    import hdf5plugin  # noqa: F401  (hits / step use a plugin filter)
    from helix.core.coeff_event import code_version
    from helix.tpc import io

    out = cells_path(coeff_shard, out_dir)
    if os.path.exists(out):
        return out, "exists"
    with h5py.File(coeff_shard, "r") as f:
        ident = read_ident(f)
    records, threshold, sim, files = [], set(), {}, {}
    try:
        for run, sf, ev in zip(ident["run"], ident["source_file"], ident["event"].tolist()):
            if (run, sf) not in files:
                paths = io.task_files(io.sensor_path(source_root, run, sf))
                fh, fs = (h5py.File(paths[kind], "r") for kind in ("hits", "step"))
                fl = h5py.File(labl_path(paths["hits"]), "r")
                files[(run, sf)] = fh, fs, fl
                for x in (fh, fs):
                    io.require_format(x)
                ids = io.event_ids(fh)
                if not (np.array_equal(ids, io.event_ids(fs)) and np.array_equal(ids, io.event_ids(fl))):
                    raise ValueError(f"{paths['hits']} and its step / labl files hold different events")
                files[(run, sf)] = fh, fs, fl, {int(e): i for i, e in enumerate(ids)}
                c = fh["config"].attrs
                sim[f"{run}/{sf}"] = {k: (c[k].item() if hasattr(c[k], "item") else str(c[k]))
                                      for k in ("production_version", "git_commit", "git_dirty") if k in c}
            fh, fs, fl, pos = files[(run, sf)]
            if ev not in pos:
                raise KeyError(f"{run}/{sf}: no event {ev}")
            hits_ev = io.entry(fh, pos[ev], ev)
            threshold.add(float(hits_ev.attrs["threshold"]))
            records.append(event_record(hits_ev, io.entry(fs, pos[ev], ev), io.entry(fl, pos[ev], ev)))
    finally:
        for handles in files.values():
            for x in handles[:3]:
                x.close()
    if len(threshold) > 1:
        raise ValueError(f"{coeff_shard}: hits written with different sample thresholds {sorted(threshold)}")
    label = dict(LABEL, hits_threshold=threshold.pop() if threshold else None)
    write_shard(out, ident, records, label=label,
                provenance=dict(code=code_version(), coeff_shard=os.path.basename(str(coeff_shard)), sim=sim))
    n = sum(len(r["key"]) for r in records)
    return out, f"{len(records)} events, {n:,} cells"


# ------------------------------------------------------------------------------------------- read
def read_event(f, pos):
    """Event at write position ``pos`` of an open cells file -> a record (:data:`SCHEMA` names)."""
    out = {}
    for grp in GROUPS:
        off = f[grp]["event_offset"]
        a, b = int(off[pos]), int(off[pos + 1])
        for g, col, name, dt in SCHEMA:
            if g == grp:
                out[name] = f[grp][col][a:b]
    return out


def open_event(path, pos):
    """``(record, ident)`` of event ``pos`` of a cells shard at ``path``, format-checked."""
    with h5py.File(path, "r") as f:
        require_format(f["config"], str(path))
        ident = read_ident(f)
        return read_event(f, pos), dict(run=ident["run"][pos], source_file=ident["source_file"][pos],
                                        event=int(ident["event"][pos]))
