"""Volume / pressure constraint tests."""
from __future__ import annotations

import numpy as np

from xpbd import Volume, System, build_mesh


def _octahedron():
    """Regular octahedron — closed mesh for volume constraint."""
    V = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0],
        [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ], dtype=np.float64)
    F = np.array([
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ], dtype=np.int64)
    return build_mesh(V, F)


def test_zero_correction_at_rest():
    """Volume at rest value → no correction."""
    mesh = _octahedron()
    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, 0.0, 0.0))
    vol = Volume.from_mesh(mesh, k_pressure=1.0, compliance=0.0)
    sys.add_constraint(vol)

    X_before = sys.X.copy()
    sys.step(dt=1e-2, iters=10)
    np.testing.assert_allclose(sys.X, X_before, atol=1e-10)


def test_inflation():
    """k_pressure > 1 inflates the mesh."""
    mesh = _octahedron()
    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, 0.0, 0.0))
    vol = Volume.from_mesh(mesh, k_pressure=1.5, compliance=0.0)
    sys.add_constraint(vol)

    # Initial mean radius
    r0 = np.linalg.norm(sys.X, axis=1).mean()
    for _ in range(50):
        sys.step(dt=1e-2, iters=10)
    r1 = np.linalg.norm(sys.X, axis=1).mean()

    assert r1 > r0, f"expected inflation: r0={r0}, r1={r1}"


def test_open_mesh_rejected():
    """Volume constraint raises on open mesh."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float64)
    F = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = build_mesh(V, F)
    try:
        Volume.from_mesh(mesh)
        assert False, "should have raised"
    except ValueError:
        pass
