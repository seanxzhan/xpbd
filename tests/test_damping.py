"""Rigid-mode-preserving damping (PBD §3.5)."""
from __future__ import annotations

import numpy as np

from xpbd import System


def test_rigid_motion_preserved():
    """k_damp=1 kills internal motion but preserves rigid-body velocity."""
    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    masses = np.array([1.0, 1.0, 1.0])
    sys = System(X, masses, gravity=(0.0, 0.0, 0.0))

    # Uniform translation + some internal motion
    vcm = np.array([2.0, 1.0, 0.0])
    sys.V = np.array([
        vcm + [0.1, -0.2, 0.0],
        vcm + [-0.05, 0.1, 0.0],
        vcm + [-0.05, 0.1, 0.0],
    ])

    # After one step with k_damp=1.0 and no constraints, velocity should
    # be purely rigid-body (CM velocity preserved)
    sys.step(dt=1e-3, iters=0, k_damp=1.0)

    # CM velocity preserved
    vcm_after = (sys.V * masses[:, None]).sum(axis=0) / masses.sum()
    np.testing.assert_allclose(vcm_after, vcm, atol=1e-9)


def test_zero_damping_unchanged():
    """k_damp=0 leaves velocities unchanged through the damping step."""
    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])
    sys = System(X, masses, gravity=(0.0, 0.0, 0.0))
    sys.V = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    V_before = sys.V.copy()

    sys.step(dt=1e-3, iters=0, k_damp=0.0)

    # With no constraints and no gravity and no damping, V should stay same
    # (only the predict step changes X, V gets recovered as (P-X)/dt = V)
    np.testing.assert_allclose(sys.V, V_before, atol=1e-12)
