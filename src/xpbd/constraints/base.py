"""XPBD constraint group interface.

Unlike PBD where stiffness k ∈ [0,1] is a unitless solver artifact,
XPBD uses a compliance α (inverse stiffness) with physical units.
Each constraint accumulates a Lagrange multiplier λ across solver
iterations, giving timestep- and iteration-count-independent behaviour.

The projection methods return (dP, dLam) — both the position correction
and the multiplier update — so the system can accumulate λ.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ConstraintGroup(ABC):
    """A vectorized batch of constraints with XPBD compliance."""

    compliance: np.ndarray   # (M,) per-constraint compliance α
    damping: np.ndarray | None  # (M,) per-constraint damping β, or None
    lam: np.ndarray          # (M,) accumulated Lagrange multiplier

    @abstractmethod
    def project_batch(
        self,
        P: np.ndarray,
        W: np.ndarray,
        X_prev: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute position and multiplier corrections for this batch.

        Parameters
        ----------
        P : (N, 3) float64
            Current predicted positions.
        W : (N,) float64
            Inverse masses; W[i] == 0 → pinned.
        X_prev : (N, 3) float64
            Positions at the start of the step (for damping term).
        dt : float
            Timestep.

        Returns
        -------
        dP : (N, 3) float64
            Position correction.
        dLam : (M,) float64
            Multiplier update (caller adds to self.lam).
        """

    # ------------------------------------------------------------- Gauss–Seidel

    @property
    def n_colors(self) -> int:
        """Number of color classes. Default 1: the whole group is one class."""
        return 1

    def project_color(
        self,
        P: np.ndarray,
        W: np.ndarray,
        X_prev: np.ndarray,
        dt: float,
        c: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project just the constraints of color c.

        Default: groups without coloring fold into a single class.
        """
        if c == 0:
            return self.project_batch(P, W, X_prev, dt)
        return np.zeros_like(P), np.zeros_like(self.lam)
