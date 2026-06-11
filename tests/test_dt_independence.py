"""XPBD-specific: proves timestep and iteration independence.

This is the headline XPBD result — the fix for PBD's central flaw.
Same physical scene at varying dt and iters should produce the same
equilibrium, unlike PBD where stiffness depends on both.
"""
from __future__ import annotations

import numpy as np

from xpbd import Stretch, System


def _spring_drop(dt: float, iters: int, total_t: float, compliance: float):
    """Drop a mass at (1, 0, 0) anchored to pinned (0,0,0) with rest=1.
    Run for total_t and return equilibrium extension."""
    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    sys = System(X, np.ones(2), gravity=(0.0, 0.0, -9.81))
    sys.pin([0])
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), compliance=compliance))

    n_steps = int(round(total_t / dt))
    for _ in range(n_steps):
        sys.step(dt=dt, iters=iters)
    return float(np.linalg.norm(sys.X[1])) - 1.0


def test_extension_independent_of_dt():
    """Same compliance at varying dt → same equilibrium extension.

    This is exactly what PBD gets wrong and XPBD fixes. We use explicit
    constraint damping to help the system reach a true steady state,
    eliminating the confound of different numerical damping at different dt.
    """
    total_t = 10.0
    iters = 50
    compliance = 5e-3
    damping = 10.0  # strong explicit damping to settle quickly

    extensions = []
    for dt in (1.0 / 60, 1.0 / 120, 1.0 / 240):
        X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        sys = System(X, np.ones(2), gravity=(0.0, 0.0, -9.81))
        sys.pin([0])
        sys.add_constraint(Stretch(
            np.array([[0, 1]]), np.array([1.0]),
            compliance=compliance, damping=damping,
        ))
        n_steps = int(round(total_t / dt))
        for _ in range(n_steps):
            sys.step(dt=dt, iters=iters)
        ext = float(np.linalg.norm(sys.X[1])) - 1.0
        extensions.append(ext)

    # XPBD equilibrium: extension ≈ m*g*α = 1.0 * 9.81 * 5e-3 ≈ 0.049
    # All timesteps should converge near this value.
    spread = max(extensions) - min(extensions)
    mean_ext = np.mean(extensions)
    assert spread / mean_ext < 0.30, (
        f"dt-dependent extension (XPBD should fix this): {extensions}, "
        f"spread/mean={spread/mean_ext:.3f}"
    )


def test_extension_independent_of_iters():
    """Same compliance at varying iters → same equilibrium extension."""
    dt = 1.0 / 60
    total_t = 2.0
    compliance = 1e-3

    extensions = []
    for iters in (10, 20, 40, 80):
        ext = _spring_drop(dt=dt, iters=iters, total_t=total_t, compliance=compliance)
        extensions.append(ext)

    spread = max(extensions) - min(extensions)
    mean_ext = np.mean(extensions)
    assert spread / mean_ext < 0.10, (
        f"iter-dependent extension (XPBD should fix this): {extensions}"
    )
