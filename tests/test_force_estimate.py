"""XPBD-specific: Lagrange multiplier as constraint force estimate.

The total multiplier λ after solving gives the constraint force magnitude:
    F = λ / Δt²

For a chain of particles under gravity, the top constraint force should
approximate the total weight of the chain below it.
"""
from __future__ import annotations

import numpy as np

from xpbd import Stretch, System


def test_chain_support_force():
    """Chain of N particles; top fixed. Force at support ≈ (N-1)*m*g."""
    N = 10
    spacing = 0.1
    mass = 1.0
    compliance = 1e-8  # very stiff (nearly rigid)
    dt = 1.0 / 60
    iters = 100

    X = np.zeros((N, 3), dtype=np.float64)
    X[:, 1] = np.arange(N) * (-spacing)  # hanging down

    sys = System(X, np.full(N, mass), gravity=(0.0, -9.81, 0.0))
    sys.pin([0])

    edges = np.array([[i, i + 1] for i in range(N - 1)])
    rest = np.full(N - 1, spacing)
    stretch = Stretch(edges, rest, compliance=compliance)
    sys.add_constraint(stretch)

    # Run until settled (Gauss-Seidel for accurate force estimates)
    for _ in range(300):
        sys.step(dt=dt, iters=iters, solver="gauss-seidel")

    # Force at the top constraint (index 0): should support (N-1) particles
    # λ / Δt² gives the constraint force
    lam_top = stretch.lam[0]
    force_estimate = abs(lam_top) / (dt * dt)
    expected_force = (N - 1) * mass * 9.81

    rel_error = abs(force_estimate - expected_force) / expected_force
    assert rel_error < 0.15, (
        f"Force estimate {force_estimate:.2f} vs expected {expected_force:.2f}, "
        f"rel error {rel_error:.3f}"
    )


def test_force_improves_with_iterations():
    """More iterations → better force estimate convergence."""
    N = 5
    spacing = 0.2
    mass = 1.0
    compliance = 1e-8
    dt = 1.0 / 60

    errors = []
    for iters in (20, 50, 200):
        X = np.zeros((N, 3), dtype=np.float64)
        X[:, 1] = np.arange(N) * (-spacing)

        sys = System(X, np.full(N, mass), gravity=(0.0, -9.81, 0.0))
        sys.pin([0])

        edges = np.array([[i, i + 1] for i in range(N - 1)])
        rest = np.full(N - 1, spacing)
        stretch = Stretch(edges, rest, compliance=compliance)
        sys.add_constraint(stretch)

        for _ in range(300):
            sys.step(dt=dt, iters=iters)

        lam_top = stretch.lam[0]
        force_estimate = abs(lam_top) / (dt * dt)
        expected_force = (N - 1) * mass * 9.81
        errors.append(abs(force_estimate - expected_force) / expected_force)

    # Error should decrease (or stay flat) with more iterations
    for a, b in zip(errors, errors[1:]):
        assert b <= a + 0.01, f"error didn't improve: {errors}"
