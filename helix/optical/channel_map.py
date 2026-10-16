"""Optical detector channel map: readout channel -> optical detector (opdet), position, type.

The model needs, per optical waveform, where its detector is and what it sees (SBND:
coated PMTs see VUV + visible light, uncoated PMTs visible only, X-ARAPUCAs VUV or
visible). This module builds one table from whatever the experiment provides and maps
the readout channels stored in the shard (``/optical/<system>/channel``) onto it.

Inputs (any combination; later ones fill or override earlier ones):
  * ``geometry``: per opdet id the centre (cm) and the detector type -
      - HDF5 with ``/opdets/{opdet, center_xyz_cm, pd_type}`` (e.g. a geometry export);
      - CSV with a header ``opdet,x,y,z,pd_type[,tpc]``;
      - JSON: a list of objects with those keys.
  * ``pds_map``: sbndcode's ``sbnd_pds_mapping.json`` (a list indexed by opdet: pd_type,
    tpc, sensible_to_vuv, sensible_to_vis).
  * ``readout``: per system, readout channel -> opdet, for systems whose readout ids are
    not opdet ids (CSV ``readout,opdet`` or JSON ``{"readout": opdet}``). Without one,
    a system's readout channel IS the opdet id where the opdet's type belongs to that
    system (SBND PMTs); otherwise its channels are reported unknown (SBND X-ARAPUCA:
    the readout codes 700-1963 need the DAQ map).

    python -m helix.optical.channel_map --geometry geo.h5 --pds-map sbnd_pds_mapping.json \\
        [--readout xarapuca=xa_map.csv] --out map.npz [--check extracted.h5]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parent / "data"
SYSTEM_TYPES = {"pmt": ("pmt_coated", "pmt_uncoated"), "xarapuca": ("xarapuca_vuv", "xarapuca_vis")}
SEES = {"pmt_coated": (True, True), "pmt_uncoated": (False, True), "xarapuca_vuv": (True, False),
        "xarapuca_vis": (False, True)}


def _rows(path):
    p = Path(path)
    if p.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            g = f["opdets"]
            xyz = g["center_xyz_cm"][:]
            return [dict(opdet=int(o), x=float(a), y=float(b), z=float(c),
                         pd_type=t.decode() if isinstance(t, bytes) else str(t))
                    for o, (a, b, c), t in zip(g["opdet"][:], xyz, g["pd_type"][:])]
    if p.suffix == ".csv":
        with open(p) as fh:
            return [dict(r) for r in csv.DictReader(fh)]
    d = json.loads(p.read_text())
    if isinstance(d, dict):                                    # {"opdet": {...}}
        return [dict(v, opdet=int(k)) for k, v in d.items()]
    return [dict(r, opdet=int(r.get("opdet", r.get("channel", i)))) for i, r in enumerate(d)]


def build(geometry=None, pds_map=None, readout=None):
    """-> dict of arrays: opdet, xyz (cm, nan if unknown), pd_type, tpc (-1 unknown),
    vuv, vis; and per system ``readout_<s>``, ``opdet_<s>`` (explicit readout maps)."""
    rec = {}
    for src in (geometry, pds_map):
        if src is None:
            continue
        for r in _rows(src):
            o = int(r["opdet"]); e = rec.setdefault(o, {})
            for k in ("x", "y", "z"):
                if r.get(k) not in (None, ""):
                    e[k] = float(r[k])
            if r.get("pd_type"):
                e["pd_type"] = str(r["pd_type"])
            if r.get("tpc") not in (None, ""):
                e["tpc"] = int(r["tpc"])
            for k in ("sensible_to_vuv", "sensible_to_vis"):
                if k in r:
                    e[k] = bool(r[k]) if not isinstance(r[k], str) else r[k].lower() in ("1", "true")
    op = np.array(sorted(rec), np.int32)
    m = dict(opdet=op,
             xyz=np.array([[rec[o].get(k, np.nan) for k in ("x", "y", "z")] for o in op], np.float64).reshape(-1, 3),
             pd_type=np.array([rec[o].get("pd_type", "") for o in op]))
    tpc = [rec[o].get("tpc", -1) for o in op]
    m["tpc"] = np.array([t if t >= 0 or np.isnan(x) else int(x > 0) for t, x in zip(tpc, m["xyz"][:, 0])], np.int8)
    m["vuv"] = np.array([rec[o].get("sensible_to_vuv", SEES.get(t, (False, False))[0]) for o, t in zip(op, m["pd_type"])])
    m["vis"] = np.array([rec[o].get("sensible_to_vis", SEES.get(t, (False, False))[1]) for o, t in zip(op, m["pd_type"])])
    for s, src in (readout or {}).items():
        p = Path(src)
        if p.suffix == ".csv":
            with open(p) as fh:
                pairs = [(int(r["readout"]), int(r["opdet"])) for r in csv.DictReader(fh)]
        else:
            pairs = [(int(k), int(v)) for k, v in json.loads(p.read_text()).items()]
        m[f"readout_{s}"] = np.array([a for a, _ in pairs], np.int32)
        m[f"opdet_{s}"] = np.array([b for _, b in pairs], np.int32)
    return m


def save(m, path):
    np.savez(path, **m)


def load(path=None):
    """The packaged map (``data/sbnd_channel_map.npz``) or ``path``."""
    d = np.load(path or _DATA / "sbnd_channel_map.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def lookup(m, system, channels):
    """Readout channels of ``system`` -> dict(opdet, xyz, pd_type, tpc, vuv, vis, known)."""
    ch = np.asarray(channels, np.int64)
    opdet = ch                                     # default: the readout channel is the opdet id
    if f"readout_{system}" in m:
        r, o = m[f"readout_{system}"], m[f"opdet_{system}"]
        srt = np.argsort(r)
        k = srt[np.clip(np.searchsorted(r, ch, sorter=srt), 0, len(r) - 1)]
        opdet = np.where(r[k] == ch, o[k], -1)
    i = np.clip(np.searchsorted(m["opdet"], opdet), 0, len(m["opdet"]) - 1)
    known = (m["opdet"][i] == opdet) & np.isin(m["pd_type"][i], SYSTEM_TYPES.get(system, ()))
    nan = np.full(3, np.nan)
    return dict(opdet=np.where(known, opdet, -1), xyz=np.where(known[:, None], m["xyz"][i], nan),
                pd_type=np.where(known, m["pd_type"][i], ""), tpc=np.where(known, m["tpc"][i], -1),
                vuv=known & m["vuv"][i], vis=known & m["vis"][i], known=known)


def check(m, extracted):
    """Channel coverage of an extracted SBND file (``<system>/wf/channel``)."""
    import h5py
    out = {}
    with h5py.File(extracted, "r") as f:
        for s in SYSTEM_TYPES:
            if f"{s}/wf/channel" not in f:
                continue
            ch = np.unique(f[f"{s}/wf/channel"][:])
            L = lookup(m, s, ch)
            out[s] = dict(channels=int(len(ch)), known=int(L["known"].sum()),
                          types={t: int((L["pd_type"] == t).sum()) for t in SYSTEM_TYPES[s]},
                          unknown_examples=ch[~L["known"]][:5].tolist())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry"); ap.add_argument("--pds-map")
    ap.add_argument("--readout", action="append", default=[], help="SYSTEM=FILE (csv readout,opdet or json)")
    ap.add_argument("--out", required=True); ap.add_argument("--check", default=None)
    a = ap.parse_args(argv)
    m = build(a.geometry, a.pds_map, dict(r.split("=", 1) for r in a.readout))
    save(m, a.out)
    print(f"{len(m['opdet'])} optical detectors:", {t: int((m['pd_type'] == t).sum()) for t in np.unique(m["pd_type"])})
    if a.check:
        print(json.dumps(check(m, a.check), indent=1))


if __name__ == "__main__":
    main()
