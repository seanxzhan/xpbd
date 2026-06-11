"""Linear and angular momentum conservation under internal constraints."""
from __future__ import annotations

import numpy as np

from xpbd import Stretch, System


def _linear_momentum(V, masses):
    return (V * masses[:, None]).sum(axis=0)


def _angular_momentum(X, V, masses, origin):
    r = X - origin
    p = V * masses[:, None]
    return np.cross(r, p).sum(axis=0)


def test_linear_momentum_conserved():
    """Internal constraints conserve total linear momentum (no gravity)."""
    X = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    masses = np.array([1.0, 2.0, 1.0])
    sys = System(X, masses, gravity=(0.0, 0.0, 0.0))
    sys.V = np.array([[1.0, 0.0, 0.0], [-0.5, 0.5, 0.0], [0.0, -1.0, 0.0]])

    edges = np.array([[0, 1], [1, 2], [0, 2]])
    rest = np.linalg.norm(np.diff(X[edges], axis=1).squeeze(), axis=1)
    sys.add_constraint(Stretch(edges, rest, compliance=1e-4))

    p0 = _linear_momentum(sys.V, masses)
    for _ in range(50):
        sys.step(dt=1.0 / 60, iters=10)
    p1 = _linear_momentum(sys.V, masses)

    np.testing.assert_allclose(p1, p0, atol=1e-9)


def test_angular_momentum_conserved():
    """Internal constraints conserve angular momentum (no gravity)."""
    X = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    masses = np.array([1.0, 2.0, 1.0])
    sys = System(X, masses, gravity=(0.0, 0.0, 0.0))
    sys.V = np.array([[0.5, 0.3, 0.0], [-0.2, 0.4, 0.0], [0.1, -0.5, 0.0]])

    edges = np.array([[0, 1], [1, 2], [0, 2]])
    rest = np.linalg.norm(np.diff(X[edges], axis=1).squeeze(), axis=1)
    sys.add_constraint(Stretch(edges, rest, compliance=1e-4))

    origin = np.zeros(3)
    L0 = _angular_momentum(sys.X, sys.V, masses, origin)
    for _ in range(50):
        sys.step(dt=1.0 / 60, iters=10)
    L1 = _angular_momentum(sys.X, sys.V, masses, origin)

    np.testing.assert_allclose(L1, L0, atol=5e-3)
