"""What helix and pimm-data duplicate, pinned so it cannot drift.

Some duplication between the two packages is FORCED by the boundary: helix.core
and helix.tpc must import with no pimm_data present (docs/ARCHITECTURE.md §1),
so they cannot import a shared helper, and pimm-data must run with no helix
present. Where both genuinely need the same thing, both carry it.

The old `tests/test_forward_mirror.py` held those copies together and was deleted
with the forward-model move -- correctly for the parts that stopped being
mirrored, WRONGLY for the parts that did not. Geometry never moved: it is still
duplicated, and from that deletion until this file it was pinned by nothing.

This file can exist because a TEST may import both packages even though
helix.tpc may not. That is the whole trick, and it is why the pin belongs here
rather than in either package.

What is deliberately NOT pinned: helix's plane-label parser
(helix.tpc.geometry._label_gid, over helix.tpc.readout.canonical_plane_gid) and
canonical_plane_id (pimm-data) have ALREADY diverged on purpose -- helix's is
wire-only and raises on `volume_N_Pixel`, pimm-data's handles it, which is why
test_canonical_plane_id_stable stayed in pimm-data during the move. A test
asserting they agree would be wrong.
"""

import hashlib
import os

import numpy as np
import pytest

pytest.importorskip("pimm_data")
pytest.importorskip("torch")

import torch  # noqa: E402
import pimm_data.geometry as pd_geom  # noqa: E402
from helix.tpc import geometry as hx_geom  # noqa: E402


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _bundled(mod, name="cubic_wireplane_geometry.json"):
    return os.path.join(os.path.dirname(mod.__file__), "data", name)


# ── the detector description ────────────────────────────────────────────────

def test_bundled_geometry_json_is_byte_identical():
    """Both packages ship the same 153 kB JAXTPC export.

    It is an exported artifact, authored by neither, and in both the bundled copy
    is only a fallback (`_resolve` takes any path). But two copies of a detector
    description drift as silently as two copies of code.
    """
    a, b = _bundled(hx_geom), _bundled(pd_geom)
    assert os.path.exists(a) and os.path.exists(b)
    assert _md5(a) == _md5(b), (
        "the bundled plane geometry differs between helix and pimm-data; "
        "densify would size grids from one and the DSP from the other")


def test_both_loaders_agree_on_every_plane():
    """Not just the file -- the parsed registries must match field by field."""
    h = hx_geom.load_plane_registry(_bundled(hx_geom))
    p = pd_geom.load_plane_registry(_bundled(pd_geom))
    assert set(h) == set(p), "different plane ids"
    for gid in sorted(h):
        for key in ("label", "n_wires", "n_ticks", "pedestal"):
            assert h[gid][key] == p[gid][key], f"plane {gid} disagrees on {key}"
        assert np.array_equal(np.asarray(h[gid]["wire_lengths"]),
                              np.asarray(p[gid]["wire_lengths"])), \
            f"plane {gid} disagrees on wire_lengths"


# ── the one duplicated function ─────────────────────────────────────────────



