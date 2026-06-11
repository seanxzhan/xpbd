"""Mesh topology tests."""
from __future__ import annotations

import numpy as np

from xpbd import Mesh, NonManifoldError, build_mesh


def test_single_triangle():
    """Single triangle: 3 edges, no interior edges, open mesh."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float64)
    F = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = build_mesh(V, F)
    assert mesh.edges.shape[0] == 3
    assert mesh.bend_quads.shape[0] == 0
    assert not mesh.is_closed
    assert mesh.boundary_mask.all()


def test_two_triangles():
    """Two triangles sharing an edge: 1 interior edge → 1 bend quad."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0]], dtype=np.float64)
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64)
    mesh = build_mesh(V, F)
    assert mesh.edges.shape[0] == 5
    assert mesh.bend_quads.shape[0] == 1
    assert not mesh.is_closed


def test_non_manifold_raises():
    """Three faces sharing an edge → NonManifoldError."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0], [0.5, 0, 1]],
                 dtype=np.float64)
    F = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    try:
        build_mesh(V, F)
        assert False, "should have raised"
    except NonManifoldError:
        pass


def test_vertex_masses():
    """Vertex masses sum to total mesh area × density."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    F = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = build_mesh(V, F)
    density = 2.0
    masses = mesh.vertex_masses(density)
    total_area = 0.5  # unit right triangle
    np.testing.assert_allclose(masses.sum(), total_area * density, atol=1e-12)
