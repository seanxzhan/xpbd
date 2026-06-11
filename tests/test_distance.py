"""Stretch (distance) constraint behaviour tests."""
from __future__ import annotations

import numpy as np

from xpbd import Stretch, System


def _two_particle_system(distance: float, masses=(1.0, 1.0), gravity=(0.0, 0.0, 0.0)):
    X = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    s = System(X, np.array(masses), gravity=gravity)
    return s


def test_residual_zero_at_rest():
    """Particles already at rest length: |C| stays at 0 after a step."""
    sys = _two_particle_system(distance=1.0)
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), compliance=0.0))
    sys.step(dt=1e-2, iters=1)
    L = np.linalg.norm(sys.X[0] - sys.X[1])
    assert abs(L - 1.0) < 1e-12


def test_one_iter_zero_compliance_kills_gap():
    """compliance=0, one iter → stretched pair pulled to rest length (same as PBD k=1)."""
    sys = _two_particle_system(distance=1.5)
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), compliance=0.0))
    sys.step(dt=1e-3, iters=1)
    L = np.linalg.norm(sys.X[0] - sys.X[1])
    assert abs(L - 1.0) < 1e-9, f"length={L}, expected 1.0"


def test_pinned_endpoint():
    """Particle 0 pinned at origin; rest length 1 → free particle settles to |x|=1."""
    sys = _two_particle_system(distance=1.5)
    sys.pin([0])
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), compliance=0.0))
    sys.step(dt=1e-3, iters=1)
    np.testing.assert_array_equal(sys.X[0], [0.0, 0.0, 0.0])
    L = np.linalg.norm(sys.X[1])
    assert abs(L - 1.0) < 1e-9


def test_both_endpoints_pinned_no_change():
    """Both pinned → no-op."""
    sys = _two_particle_system(distance=2.0)
    sys.pin([0, 1])
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), compliance=0.0))
    X_before = sys.X.copy()
    sys.step(dt=1e-3, iters=10)
    np.testing.assert_array_equal(sys.X, X_before)


def test_residual_monotone_in_iters():
    """Residual decreases with more iterations."""
    edges = np.array([[0, 1]])
    rest = np.array([1.0])
    residuals = []
    for n_iter in (1, 4, 16, 64):
        sys = _two_particle_system(distance=2.0)
        sys.add_constraint(Stretch(edges, rest, compliance=1e-4))
        sys.step(dt=1e-2, iters=n_iter)
        residuals.append(abs(np.linalg.norm(sys.X[0] - sys.X[1]) - 1.0))
    for a, b in zip(residuals, residuals[1:]):
        assert b <= a + 1e-12, f"residual rose: {residuals}"


def test_compliance_controls_equilibrium():
    """Higher compliance → larger equilibrium extension under gravity."""
    extensions = []
    for alpha in (1e-5, 1e-4, 1e-3):
        X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        sys = System(X, np.ones(2), gravity=(0.0, 0.0, -9.81))
        sys.pin([0])
        sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), compliance=alpha))
        for _ in range(500):
            sys.step(dt=1.0 / 60, iters=10)
        ext = float(np.linalg.norm(sys.X[1])) - 1.0
        extensions.append(ext)
    # Higher compliance = softer = more extension
    for a, b in zip(extensions, extensions[1:]):
        assert b > a, f"expected increasing extension, got {extensions}"
