"""Free-fall trajectory matches symplectic Euler (no constraints active)."""
from __future__ import annotations

import numpy as np

from xpbd import System


def _symplectic_euler(x0, v0, g, dt, n_steps):
    x, v = x0.copy(), v0.copy()
    for _ in range(n_steps):
        v = v + dt * g
        x = x + dt * v
    return x, v


def test_freefall_matches_symplectic_euler():
    """Single particle in gravity, no constraints."""
    X = np.array([[1.0, 2.0, 3.0]])
    sys = System(X, np.array([1.0]), gravity=(0.0, -9.81, 0.0))
    sys.V = np.array([[0.5, 1.0, -0.3]])

    dt = 1.0 / 60
    n_steps = 100
    for _ in range(n_steps):
        sys.step(dt=dt, iters=1)

    x_ref, v_ref = _symplectic_euler(
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, 1.0, -0.3]),
        np.array([0.0, -9.81, 0.0]),
        dt, n_steps
    )
    np.testing.assert_allclose(sys.X[0], x_ref, atol=1e-10)
    np.testing.assert_allclose(sys.V[0], v_ref, atol=1e-10)


def test_pinned_particle_stays_put():
    """Pinned particle ignores gravity."""
    X = np.array([[0.0, 5.0, 0.0]])
    sys = System(X, np.array([1.0]), gravity=(0.0, -9.81, 0.0))
    sys.pin([0])
    for _ in range(100):
        sys.step(dt=1.0 / 60, iters=5)
    np.testing.assert_array_equal(sys.X[0], [0.0, 5.0, 0.0])
    np.testing.assert_array_equal(sys.V[0], [0.0, 0.0, 0.0])
