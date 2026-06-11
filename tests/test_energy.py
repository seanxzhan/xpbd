"""Energy dissipation (implicit damping settles the system)."""
from __future__ import annotations

import numpy as np

from xpbd import Stretch, System


def _kinetic_energy(V, masses):
    return 0.5 * float(np.sum(masses * np.sum(V * V, axis=1)))


def test_system_dissipates_to_rest():
    """XPBD implicit time integration dissipates energy — system settles.

    Note: unlike PBD where KE is monotone non-increasing, XPBD with compliance
    has elastic potential energy — KE can temporarily increase as elastic
    energy converts to kinetic. But overall the system must settle.
    """
    X = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])
    sys = System(X, masses, gravity=(0.0, 0.0, 0.0))
    sys.V = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([2.0]), compliance=1e-4))

    KE_initial = _kinetic_energy(sys.V, masses)
    for _ in range(500):
        sys.step(dt=1.0 / 60, iters=10)

    KE_final = _kinetic_energy(sys.V, masses)
    # After 500 steps, system should have dissipated significantly
    assert KE_final < 0.1 * KE_initial, (
        f"system didn't settle: KE_initial={KE_initial}, KE_final={KE_final}"
    )


def test_zero_compliance_ke_non_increasing():
    """With zero compliance (infinitely stiff) starting at rest, KE is non-increasing.

    Important: the initial configuration must already satisfy the constraint.
    If it doesn't, the first projection adds energy by moving particles to
    satisfy C=0 — that's correct constraint enforcement, not a bug.
    """
    X = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # already at rest length
    masses = np.array([1.0, 1.0])
    sys = System(X, masses, gravity=(0.0, 0.0, 0.0))
    # Velocity along the constraint direction (will stretch it)
    sys.V = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])

    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([2.0]), compliance=0.0))

    # After first step, the constraint snaps particles back — KE drops.
    # Then it should remain non-increasing.
    sys.step(dt=1.0 / 60, iters=10)
    KE_prev = _kinetic_energy(sys.V, masses)
    for _ in range(50):
        sys.step(dt=1.0 / 60, iters=10)
        KE = _kinetic_energy(sys.V, masses)
        assert KE <= KE_prev + 1e-9, f"KE increased: {KE_prev} → {KE}"
        KE_prev = KE
