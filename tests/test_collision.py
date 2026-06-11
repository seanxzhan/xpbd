"""Collision constraint tests."""
from __future__ import annotations

import numpy as np

from xpbd import Plane, Sphere, System


def test_particle_settles_on_plane():
    """Single particle dropped onto a floor plane."""
    X = np.array([[0.0, 1.0, 0.0]])
    sys = System(X, np.array([1.0]), gravity=(0.0, -9.81, 0.0))
    sys.add_collider(Plane(normal=(0.0, 1.0, 0.0), offset=0.0))

    for _ in range(200):
        sys.step(dt=1.0 / 60, iters=5)

    # Should be at or above the floor
    assert sys.X[0, 1] >= -1e-6, f"fell through floor: y={sys.X[0, 1]}"


def test_particle_stays_above_sphere():
    """Particle dropped onto a sphere remains above its surface."""
    X = np.array([[0.0, 2.0, 0.0]])
    sys = System(X, np.array([1.0]), gravity=(0.0, -9.81, 0.0))
    sys.add_collider(Sphere(center=np.array([0.0, 0.0, 0.0]), radius=1.0))

    for _ in range(200):
        sys.step(dt=1.0 / 60, iters=5)

    dist = np.linalg.norm(sys.X[0]) - 1.0
    assert dist >= -1e-4, f"penetrated sphere: dist={dist}"


def test_restitution():
    """Non-zero restitution causes bounce."""
    X = np.array([[0.0, 0.5, 0.0]])
    sys = System(X, np.array([1.0]), gravity=(0.0, -9.81, 0.0))
    sys.add_collider(Plane(normal=(0.0, 1.0, 0.0), offset=0.0))

    # Drop and check velocity after collision
    for _ in range(60):
        sys.step(dt=1.0 / 60, iters=5, restitution=0.8)

    # After bounce, should have moved back up
    assert sys.X[0, 1] > 0.01, f"no bounce: y={sys.X[0, 1]}"
