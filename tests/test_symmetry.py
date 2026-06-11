"""Symmetry preservation under gravity and constraints."""
from __future__ import annotations

import numpy as np

from xpbd import Stretch, System


def test_symmetric_drop():
    """Two particles symmetric about the y-axis remain symmetric under gravity."""
    X = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])
    sys = System(X, masses, gravity=(0.0, -9.81, 0.0))
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([2.0]), compliance=1e-4))

    for _ in range(100):
        sys.step(dt=1.0 / 60, iters=10)

    # x-coordinates should remain symmetric about 0
    np.testing.assert_allclose(sys.X[0, 0], -sys.X[1, 0], atol=1e-10)
    # y-coordinates should be equal
    np.testing.assert_allclose(sys.X[0, 1], sys.X[1, 1], atol=1e-10)
    # z-coordinates should be equal (and zero)
    np.testing.assert_allclose(sys.X[0, 2], sys.X[1, 2], atol=1e-10)
