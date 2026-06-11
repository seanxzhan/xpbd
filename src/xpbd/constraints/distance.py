"""Stretch (distance) constraint with XPBD compliance.

C(p_i, p_j) = |p_i - p_j| - rest

XPBD Gauss-Seidel update (paper Eq. 18):
    Δλ_j = (-C_j - α̃_j λ_j - γ_j ∇C·(x-x_prev)) / ((w_i+w_j) + α̃_j)
where α̃ = α/Δt², γ = α̃β/Δt.
"""
from __future__ import annotations

import numpy as np

from xpbd.coloring import greedy_color
from xpbd.constraints.base import ConstraintGroup
from xpbd.mesh import Mesh


def _project_edges(
    P: np.ndarray,
    W: np.ndarray,
    X_prev: np.ndarray,
    dt: float,
    lam: np.ndarray,
    edges: np.ndarray,
    rest: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized XPBD distance projection for a subset of edges."""
    dP = np.zeros_like(P)
    d_lam = np.zeros(edges.shape[0], dtype=np.float64)
    if edges.shape[0] == 0:
        return dP, d_lam

    i = edges[:, 0]
    j = edges[:, 1]
    diff = P[i] - P[j]
    L = np.linalg.norm(diff, axis=1)
    ok = L > 1e-12
    n = np.zeros_like(diff)
    n[ok] = diff[ok] / L[ok, None]
    C = L - rest

    w1 = W[i]
    w2 = W[j]
    wsum = w1 + w2

    alpha_tilde = alpha / (dt * dt)

    # Denominator: ∇C M⁻¹ ∇Cᵀ + α̃
    # For distance constraint: |∇_{p_i} C|² = 1, |∇_{p_j} C|² = 1
    # so ∇C M⁻¹ ∇Cᵀ = w_i + w_j
    denom = wsum + alpha_tilde

    # Numerator: -C - α̃ λ
    numerator = -C - alpha_tilde * lam

    # Damping term (paper Eq. 26)
    if beta is not None:
        gamma = alpha_tilde * beta / dt
        # ∇C · (x_i - x_prev_i) for the distance constraint
        dx_i = P[i] - X_prev[i]
        dx_j = P[j] - X_prev[j]
        grad_dot_dx = np.einsum("ij,ij->i", n, dx_i) - np.einsum("ij,ij->i", n, dx_j)
        numerator -= gamma * grad_dot_dx
        denom = (1.0 + gamma) * wsum + alpha_tilde

    active = ok & (denom > 1e-20)
    d_lam[active] = numerator[active] / denom[active]

    # Position update: Δx = M⁻¹ ∇Cᵀ Δλ
    d1 = (d_lam * w1)[:, None] * n
    d2 = -(d_lam * w2)[:, None] * n
    np.add.at(dP, i, d1)
    np.add.at(dP, j, d2)
    return dP, d_lam


class Stretch(ConstraintGroup):
    def __init__(
        self,
        edges: np.ndarray,
        rest: np.ndarray,
        compliance: float | np.ndarray = 0.0,
        damping: float | np.ndarray | None = None,
    ):
        edges = np.ascontiguousarray(edges, dtype=np.int64)
        rest = np.ascontiguousarray(rest, dtype=np.float64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges must be (M, 2); got {edges.shape}")
        if rest.shape != (edges.shape[0],):
            raise ValueError(f"rest shape {rest.shape} != ({edges.shape[0]},)")
        self.idx = edges
        self.rest = rest
        M = edges.shape[0]
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
    ) -> "Stretch":
        e = mesh.edges
        d = mesh.V[e[:, 0]] - mesh.V[e[:, 1]]
        rest = np.linalg.norm(d, axis=1)
        return cls(e, rest, compliance=compliance, damping=damping)

    # ------------------------------------------------------------- Jacobi

    def project_batch(
        self,
        P: np.ndarray,
        W: np.ndarray,
        X_prev: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _project_edges(
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
        dP, d_lam_sub = _project_edges(
            P, W, X_prev, dt, self.lam[mask],
            self.idx[mask], self.rest[mask],
            self.compliance[mask],
            self.damping[mask] if self.damping is not None else None,
        )
        # Map sub-array d_lam back to full array
        d_lam_full = np.zeros(self.idx.shape[0], dtype=np.float64)
        d_lam_full[mask] = d_lam_sub
        return dP, d_lam_full
