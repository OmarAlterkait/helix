"""Optical channel map (helix.optical.channel_map): inputs in several formats, lookups."""
import json

import numpy as np

from helix.optical import channel_map as cm


def _inputs(tmp_path):
    geo = tmp_path / "geo.csv"
    geo.write_text("opdet,x,y,z,pd_type\n0,-200,0,10,xarapuca_vis\n1,-200,0,20,pmt_coated\n2,200,0,20,pmt_uncoated\n"
                   "3,200,5,30,xarapuca_vuv\n")
    pds = tmp_path / "pds.json"
    pds.write_text(json.dumps([{"channel": i, "pd_type": t, "tpc": int(i >= 2), "sensible_to_vuv": v, "sensible_to_vis": True}
                               for i, (t, v) in enumerate([("xarapuca_vis", False), ("pmt_coated", True),
                                                           ("pmt_uncoated", False), ("xarapuca_vuv", True)])]))
    xa = tmp_path / "xa.csv"
    xa.write_text("readout,opdet\n700,0\n701,3\n")
    return geo, pds, xa


def test_build_and_lookup(tmp_path):
    geo, pds, xa = _inputs(tmp_path)
    m = cm.build(geo, pds, {"xarapuca": xa})
    p = tmp_path / "m.npz"; cm.save(m, p); m = cm.load(p)
    L = cm.lookup(m, "pmt", [1, 2, 0, 99])                   # PMT readout = opdet id; 0 is an X-ARAPUCA, 99 unknown
    assert L["known"].tolist() == [True, True, False, False]
    assert L["pd_type"][:2].tolist() == ["pmt_coated", "pmt_uncoated"] and L["tpc"][:2].tolist() == [0, 1]
    assert L["vuv"][:2].tolist() == [True, False] and np.allclose(L["xyz"][1], [200, 0, 20])
    X = cm.lookup(m, "xarapuca", [701, 700, 702])
    assert X["opdet"].tolist() == [3, 0, -1] and X["pd_type"][:2].tolist() == ["xarapuca_vuv", "xarapuca_vis"]
    assert np.isnan(X["xyz"][2]).all()


def test_json_geometry_and_tpc_from_position(tmp_path):
    g = tmp_path / "g.json"
    g.write_text(json.dumps([{"opdet": 5, "x": -10, "y": 1, "z": 2, "pd_type": "pmt_coated"},
                             {"opdet": 6, "x": 10, "y": 1, "z": 2, "pd_type": "pmt_coated"}]))
    m = cm.build(g)
    assert m["tpc"].tolist() == [0, 1] and m["vuv"].all() and m["vis"].all()


def test_packaged_sbnd_map():
    m = cm.load()
    types, n = np.unique(m["pd_type"], return_counts=True)
    assert dict(zip(types.tolist(), n.tolist())) == {"pmt_coated": 96, "pmt_uncoated": 24, "xarapuca_vis": 96, "xarapuca_vuv": 96}
    assert np.isfinite(m["xyz"]).all() and set(m["tpc"].tolist()) == {0, 1}
