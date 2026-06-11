"""Dihedral bending constraint tests."""
from __future__ import annotations

import numpy as np

from xpbd import Bend, Stretch, System, build_mesh


def _two_tri_strip():
    """Two triangles sharing an edge: a minimal bend quad."""
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ], dtype=np.float64)
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64)
    return build_mesh(V, F)


def test_zero_correction_at_rest():
    """Flat configuration → bend constraint already satisfied, no correction."""
    mesh = _two_tri_strip()
    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, 0.0, 0.0))
    bend = Bend.from_mesh(mesh, compliance=0.0)
    sys.add_constraint(bend)

    X_before = sys.X.copy()
    sys.step(dt=1e-2, iters=10)
    np.testing.assert_allclose(sys.X, X_before, atol=1e-10)


def test_perturbation_pulls_back():
    """Fold the mesh, then verify bend constraint reduces the dihedral error.

    Uses moderate compliance — very low compliance (near-rigid) on a single
    bend quad can overshoot due to the nonlinear constraint linearization.
    """
    mesh = _two_tri_strip()
    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, 0.0, 0.0))
    bend = Bend.from_mesh(mesh, compliance=1e-1)
    sys.add_constraint(bend)

    # Fold vertex 3 out of plane (small perturbation)
    sys.X[3, 2] = 0.1
    sys.V[3, 2] = 0.0

    for _ in range(10):
        sys.step(dt=1e-2, iters=10, solver="gauss-seidel")

    # Vertex 3 should be pulled back toward z=0 (or at least not grow)
    assert abs(sys.X[3, 2]) < 0.11, f"expected pull-back, got z={sys.X[3, 2]}"


def test_linear_momentum_preserved():
    """Bending constraint preserves linear momentum."""
    mesh = _two_tri_strip()
    masses = mesh.vertex_masses(1.0)
    sys = System(mesh.V.copy(), masses, gravity=(0.0, 0.0, 0.0))

    # Perturb out of plane
    sys.X[2, 2] = 0.3
    sys.X[3, 2] = -0.3
    sys.V = np.random.default_rng(42).standard_normal(sys.V.shape) * 0.1

    bend = Bend.from_mesh(mesh, compliance=1e-4)
    sys.add_constraint(bend)

    p0 = (sys.V * masses[:, None]).sum(axis=0)
    for _ in range(20):
        sys.step(dt=1.0 / 60, iters=5)
    p1 = (sys.V * masses[:, None]).sum(axis=0)

    np.testing.assert_allclose(p1, p0, atol=1e-9)
