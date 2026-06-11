"""Bit-identical regression (no randomness in solver)."""
from __future__ import annotations

import numpy as np

from xpbd import Stretch, System


def test_deterministic_across_runs():
    """Same initial conditions → bit-identical trajectory."""
    def _run():
        X = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.0, 0.0]])
        masses = np.array([1.0, 1.0, 1.0])
        sys = System(X, masses, gravity=(0.0, -9.81, 0.0))
        sys.V[1] = [0.0, 1.0, 0.0]
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        rest = np.linalg.norm(np.diff(X[edges], axis=1).squeeze(), axis=1)
        sys.add_constraint(Stretch(edges, rest, compliance=1e-3))
        for _ in range(60):
            sys.step(dt=1.0 / 60, iters=5)
        return sys.X.copy(), sys.V.copy()

    X1, V1 = _run()
    X2, V2 = _run()
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(V1, V2)
