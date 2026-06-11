"""Constraint residual convergence tests."""
from __future__ import annotations

import numpy as np

from xpbd import Stretch, System


def test_chain_residual_decreases():
    """Multi-edge chain: residual drops monotonically with iterations."""
    N = 10
    spacing = 1.0
    X = np.zeros((N, 3))
    X[:, 0] = np.arange(N) * spacing
    # Perturb positions
    X[3:7, 0] += 0.5

    masses = np.ones(N)
    sys = System(X, masses, gravity=(0.0, 0.0, 0.0))
    sys.pin([0, N - 1])
    edges = np.array([[i, i + 1] for i in range(N - 1)])
    rest = np.full(N - 1, spacing)
    sys.add_constraint(Stretch(edges, rest, compliance=0.0))

    residuals = []
    for iters in (1, 5, 10, 20, 50):
        sys.X = X.copy()
        sys.V = np.zeros_like(X)
        sys.step(dt=1e-3, iters=iters)
        lengths = np.linalg.norm(np.diff(sys.X, axis=0), axis=1)
        residuals.append(float(np.max(np.abs(lengths - spacing))))

    for a, b in zip(residuals, residuals[1:]):
        assert b <= a + 1e-12, f"residual not monotone: {residuals}"


def test_gauss_seidel_converges():
    """Gauss-Seidel residual decreases with iteration count."""
    N = 20
    spacing = 0.5
    X = np.zeros((N, 3))
    X[:, 0] = np.arange(N) * spacing
    X[5:15, 0] += 0.3

    residuals = []
    for iters in (1, 5, 10, 20, 50):
        masses = np.ones(N)
        sys = System(X.copy(), masses, gravity=(0.0, 0.0, 0.0))
        sys.pin([0, N - 1])
        edges = np.array([[i, i + 1] for i in range(N - 1)])
        rest = np.full(N - 1, spacing)
        sys.add_constraint(Stretch(edges, rest, compliance=0.0))
        sys.step(dt=1e-3, iters=iters, solver="gauss-seidel")
        lengths = np.linalg.norm(np.diff(sys.X, axis=0), axis=1)
        residuals.append(float(np.max(np.abs(lengths - spacing))))

    for a, b in zip(residuals, residuals[1:]):
        assert b <= a + 1e-12, f"GS residual not monotone: {residuals}"
