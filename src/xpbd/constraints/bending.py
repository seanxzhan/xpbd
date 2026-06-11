"""Dihedral bending constraint with XPBD compliance.

For each interior edge (p1, p2) shared by two triangles (p1, p2, p3) and
(p1, p2, p4):

    C(p1, p2, p3, p4) = arccos(n1 · n2) - phi_0

Gradients via PBD paper Appendix A (eqs. 25-28).
"""
from __future__ import annotations

import numpy as np

from xpbd.coloring import greedy_color
from xpbd.constraints.base import ConstraintGroup
from xpbd.mesh import Mesh


def _initial_dihedrals(V: np.ndarray, quads: np.ndarray) -> np.ndarray:
    if quads.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    p1 = V[quads[:, 0]]; p2 = V[quads[:, 1]]
    p3 = V[quads[:, 2]]; p4 = V[quads[:, 3]]
    e2 = p2 - p1; e3 = p3 - p1; e4 = p4 - p1
    c23 = np.cross(e2, e3); l23 = np.linalg.norm(c23, axis=1, keepdims=True)
    c24 = np.cross(e2, e4); l24 = np.linalg.norm(c24, axis=1, keepdims=True)
    n1 = c23 / np.maximum(l23, 1e-20)
    n2 = c24 / np.maximum(l24, 1e-20)
    d = np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0)
    return np.arccos(d)


def _project_dihedrals(
    P: np.ndarray,
    W: np.ndarray,
    X_prev: np.ndarray,
    dt: float,
    lam: np.ndarray,
    quads: np.ndarray,
    rest: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized XPBD bend projection for a subset of dihedral 4-tuples."""
    dP = np.zeros_like(P)
    d_lam = np.zeros(quads.shape[0], dtype=np.float64)
    if quads.shape[0] == 0:
        return dP, d_lam

    i1 = quads[:, 0]; i2 = quads[:, 1]
    i3 = quads[:, 2]; i4 = quads[:, 3]

    p1 = P[i1]; p2 = P[i2]; p3 = P[i3]; p4 = P[i4]
    e2 = p2 - p1; e3 = p3 - p1; e4 = p4 - p1

    c23 = np.cross(e2, e3); l23 = np.linalg.norm(c23, axis=1)
    c24 = np.cross(e2, e4); l24 = np.linalg.norm(c24, axis=1)

    eps = 1e-12
    valid = (l23 > eps) & (l24 > eps)
    inv23 = np.where(valid, 1.0 / np.maximum(l23, eps), 0.0)
    inv24 = np.where(valid, 1.0 / np.maximum(l24, eps), 0.0)

    n1 = c23 * inv23[:, None]
    n2 = c24 * inv24[:, None]

    d = np.sum(n1 * n2, axis=1)
    d_cl = np.clip(d, -1.0 + eps, 1.0 - eps)

    # Use cosine-based constraint: C = d - cos(rest_phi)
    # This avoids the 1/sin(θ) singularity at flat configurations.
    # The gradients q1..q4 = ∂d/∂p_k are used directly.
    cos_rest = np.cos(rest)
    C = d_cl - cos_rest

    # Gradients of d = n1·n2 w.r.t. p1..p4 (Appendix A eqs. 25-28)
    d_col = d_cl[:, None]
    q3 = (np.cross(e2, n2) + np.cross(n1, e2) * d_col) * inv23[:, None]
    q4 = (np.cross(e2, n1) + np.cross(n2, e2) * d_col) * inv24[:, None]
    q2 = -((np.cross(e3, n2) + np.cross(n1, e3) * d_col) * inv23[:, None]
           + (np.cross(e4, n1) + np.cross(n2, e4) * d_col) * inv24[:, None])
    q1 = -q2 - q3 - q4

    w1 = W[i1]; w2 = W[i2]; w3 = W[i3]; w4 = W[i4]

    # ∇C M⁻¹ ∇Cᵀ = Σ w_k |q_k|²  (since ∇C = [q1, q2, q3, q4] directly)
    grad_inv_mass = (w1 * np.sum(q1 * q1, axis=1)
                     + w2 * np.sum(q2 * q2, axis=1)
                     + w3 * np.sum(q3 * q3, axis=1)
                     + w4 * np.sum(q4 * q4, axis=1))

    alpha_tilde = alpha / (dt * dt)
    denom = grad_inv_mass + alpha_tilde

    # Numerator: -C - α̃ λ
    numerator = -C - alpha_tilde * lam

    # Damping term: ∇C · (x - x_prev) = Σ q_k · dx_k
    if beta is not None:
        gamma = alpha_tilde * beta / dt
        dx1 = P[i1] - X_prev[i1]
        dx2 = P[i2] - X_prev[i2]
        dx3 = P[i3] - X_prev[i3]
        dx4 = P[i4] - X_prev[i4]
        grad_dot_dx = (np.einsum("ij,ij->i", q1, dx1)
                       + np.einsum("ij,ij->i", q2, dx2)
                       + np.einsum("ij,ij->i", q3, dx3)
                       + np.einsum("ij,ij->i", q4, dx4))
        numerator -= gamma * grad_dot_dx
        denom = (1.0 + gamma) * grad_inv_mass + alpha_tilde

    active = valid & (denom > eps) & (grad_inv_mass > eps)
    d_lam[active] = numerator[active] / denom[active]

    # Position update: Δx_k = w_k * q_k * Δλ
    np.add.at(dP, i1, (d_lam * w1)[:, None] * q1)
    np.add.at(dP, i2, (d_lam * w2)[:, None] * q2)
    np.add.at(dP, i3, (d_lam * w3)[:, None] * q3)
    np.add.at(dP, i4, (d_lam * w4)[:, None] * q4)
    return dP, d_lam


class Bend(ConstraintGroup):
    def __init__(
        self,
        quads: np.ndarray,
        rest_phi: np.ndarray,
        compliance: float | np.ndarray = 0.0,
        damping: float | np.ndarray | None = None,
    ):
        quads = np.ascontiguousarray(quads, dtype=np.int64)
        rest_phi = np.ascontiguousarray(rest_phi, dtype=np.float64)
        if quads.ndim != 2 or quads.shape[1] != 4:
            raise ValueError(f"quads must be (M, 4); got {quads.shape}")
        if rest_phi.shape != (quads.shape[0],):
            raise ValueError(f"rest_phi shape {rest_phi.shape} != ({quads.shape[0]},)")
        self.idx = quads
        self.rest = rest_phi
        M = quads.shape[0]
        if np.isscalar(compliance):
            self.compliance = np.full(M, float(compliance), dtype=np.float64)
        else:
            self.compliance = np.ascontiguousarray(compliance, dtype=np.float64)
        if damping is None:
            self.damping = None
        elif np.isscalar(damping):
            self.damping = np.full(M, float(damping), dtype=np.float64)
        else:
            self.damping = np.ascontiguousarray(damping, dtype=np.float64)
        self.lam = np.zeros(M, dtype=np.float64)
        self._colors: np.ndarray | None = None

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
        compliance: float | np.ndarray = 0.0,
        damping: float | np.ndarray | None = None,
    ) -> "Bend":
        rest = _initial_dihedrals(mesh.V, mesh.bend_quads)
        return cls(mesh.bend_quads, rest, compliance=compliance, damping=damping)

    # ------------------------------------------------------------- Jacobi

    def project_batch(
        self,
        P: np.ndarray,
        W: np.ndarray,
        X_prev: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _project_dihedrals(
            P, W, X_prev, dt, self.lam,
            self.idx, self.rest, self.compliance, self.damping,
        )

    # ------------------------------------------------------- Gauss–Seidel

    def _ensure_colors(self) -> np.ndarray:
        if self._colors is None:
            self._colors = greedy_color(self.idx)
        return self._colors

    @property
    def n_colors(self) -> int:
        c = self._ensure_colors()
        return int(c.max() + 1) if c.size else 1

    def project_color(
        self,
        P: np.ndarray,
        W: np.ndarray,
        X_prev: np.ndarray,
        dt: float,
        c: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        colors = self._ensure_colors()
        mask = colors == c
        dP, d_lam_sub = _project_dihedrals(
            P, W, X_prev, dt, self.lam[mask],
            self.idx[mask], self.rest[mask],
            self.compliance[mask],
            self.damping[mask] if self.damping is not None else None,
        )
        d_lam_full = np.zeros(self.idx.shape[0], dtype=np.float64)
        d_lam_full[mask] = d_lam_sub
        return dP, d_lam_full
