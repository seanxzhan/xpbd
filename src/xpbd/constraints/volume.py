"""Closed-mesh volume / pressure constraint with XPBD compliance.

Single equality constraint over all N vertices:

    C(p_1, ..., p_N) = Σ_t (p_{t1} × p_{t2}) · p_{t3}  -  k_pressure · V_0
"""
from __future__ import annotations

import numpy as np

from xpbd.constraints.base import ConstraintGroup
from xpbd.mesh import Mesh


def _volume_sum(V: np.ndarray, F: np.ndarray) -> float:
    p1 = V[F[:, 0]]; p2 = V[F[:, 1]]; p3 = V[F[:, 2]]
    return float(np.sum(np.cross(p1, p2) * p3))


class Volume(ConstraintGroup):
    def __init__(
        self,
        F: np.ndarray,
        V0: float,
        k_pressure: float = 1.0,
        compliance: float = 0.0,
        damping: float | None = None,
    ):
        self.F = np.ascontiguousarray(F, dtype=np.int64)
        self.V0 = float(V0)
        self.k_pressure = float(k_pressure)
        # Single constraint → scalar compliance stored as 1-element array
        self.compliance = np.array([float(compliance)], dtype=np.float64)
        if damping is None:
            self.damping = None
        else:
            self.damping = np.array([float(damping)], dtype=np.float64)
        self.lam = np.zeros(1, dtype=np.float64)

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
        k_pressure: float = 1.0,
        compliance: float = 0.0,
        damping: float | None = None,
    ) -> "Volume":
        if not mesh.is_closed:
            raise ValueError("Volume constraint requires a closed mesh")
        V0 = _volume_sum(mesh.V, mesh.F)
        return cls(mesh.F, V0, k_pressure=k_pressure, compliance=compliance,
                   damping=damping)

    def project_batch(
        self,
        P: np.ndarray,
        W: np.ndarray,
        X_prev: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        F = self.F
        p1 = P[F[:, 0]]; p2 = P[F[:, 1]]; p3 = P[F[:, 2]]

        C = float(np.sum(np.cross(p1, p2) * p3)) - self.k_pressure * self.V0

        # Per-corner gradient contribution
        g1 = np.cross(p2, p3)
        g2 = np.cross(p3, p1)
        g3 = np.cross(p1, p2)
        grad = np.zeros_like(P)
        np.add.at(grad, F[:, 0], g1)
        np.add.at(grad, F[:, 1], g2)
        np.add.at(grad, F[:, 2], g3)

        # ∇C M⁻¹ ∇Cᵀ
        grad_inv_mass = float(np.sum(W * np.sum(grad * grad, axis=1)))

        alpha_tilde = float(self.compliance[0]) / (dt * dt)
        denom = grad_inv_mass + alpha_tilde

        if denom < 1e-20 or (abs(C) < 1e-20 and abs(self.lam[0]) < 1e-20):
            return np.zeros_like(P), np.zeros(1, dtype=np.float64)

        numerator = -C - alpha_tilde * self.lam[0]

        # Damping
        if self.damping is not None:
            gamma = alpha_tilde * float(self.damping[0]) / dt
            dx = P - X_prev
            grad_dot_dx = float(np.sum(grad * dx * W[:, None]))
            numerator -= gamma * grad_dot_dx
            denom = (1.0 + gamma) * grad_inv_mass + alpha_tilde

        d_lam_val = numerator / denom
        dP = d_lam_val * W[:, None] * grad
        return dP, np.array([d_lam_val], dtype=np.float64)
