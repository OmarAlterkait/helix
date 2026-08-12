"""The along-wire axis is a detector constant; this pins the fit and the guard."""
import numpy as np
import pytest

from helix.probe.alongwire import fit_alongwire, verify_alongwire, u_of

# Measured on three events from three shards, ~600k deposits: exactly the
# +-60/0 degree wire orientations, identical to four decimals throughout.
EXPECTED = np.array([[-0.5, +np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2], [-1.0, 0.0]] * 2)
PITCH = 1.51702                      # wires per mm


def _synthetic(n=4000, seed=0):
    """Deposits whose wire follows the real geometry exactly."""
    rng = np.random.default_rng(seed)
    gid = rng.integers(0, 6, n)
    y = rng.uniform(-1500, 1500, n)
    z = rng.uniform(-1500, 1500, n)
    # PITCH directions, derived from the measured along-wire axes: along-wire is
    # (-b, a)/|a,b| for pitch (a, b), so U (-0.5,+0.866) <- pitch 30 deg,
    # V (-0.5,-0.866) <- 150 deg, Y (-1,0) <- 90 deg.
    ang = np.array([np.pi / 6, 5 * np.pi / 6, np.pi / 2] * 2)[gid]
    wire = PITCH * (y * np.cos(ang) + z * np.sin(ang)) + 1000.0
    return gid, y, z, wire


def test_fit_recovers_the_wire_orientations():
    gid, y, z, wire = _synthetic()
    vecs, diag = fit_alongwire(gid, y, z, wire)
    assert set(vecs) == set(range(6))
    for g, v in vecs.items():
        # the fit determines an AXIS; either sign is the same geometry
        d = min(np.linalg.norm(v - EXPECTED[g]), np.linalg.norm(v + EXPECTED[g]))
        assert d < 1e-6, (g, v, EXPECTED[g])
        assert abs(diag[g]["pitch_wires_per_mm"] - PITCH) < 1e-6
        assert diag[g]["resid_rms_wires"] < 1e-6


def test_verify_rejects_a_moved_axis():
    """A silently moved axis redefines the target, so it must be loud."""
    gid, y, z, wire = _synthetic()
    stored = EXPECTED.copy()
    verify_alongwire(stored, gid, y, z, wire)          # unchanged: passes

    stored[2] = [0.9, np.sqrt(1 - 0.81)]
    with pytest.raises(ValueError, match="along-wire geometry moved"):
        verify_alongwire(stored, gid, y, z, wire)


def test_verify_accepts_a_sign_flip():
    """u and -u are the same axis; only a real rotation is an error."""
    gid, y, z, wire = _synthetic()
    verify_alongwire(-EXPECTED, gid, y, z, wire)


def test_fit_refuses_when_there_is_nothing_to_fit():
    with pytest.raises(ValueError, match="no plane had"):
        fit_alongwire(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))


def test_u_projects_onto_the_axis_in_metres():
    gid = np.array([2, 2])                       # Y: along-wire is (-1, 0)
    b1 = np.array([[0.0, 1000.0, 0.0], [0.0, -2000.0, 0.0]])
    np.testing.assert_allclose(u_of(gid, b1, EXPECTED), [-1.0, 2.0], atol=1e-6)
