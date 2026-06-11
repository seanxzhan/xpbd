"""XPBD particle system + simulation step.

Extended Position-Based Dynamics (Macklin et al. 2016): constraints carry a
compliance α and accumulate a Lagrange multiplier λ across solver iterations,
giving timestep- and iteration-count-independent material stiffness.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from xpbd.constraints.base import ConstraintGroup
from xpbd.constraints.collision import generate_collision_constraints
from xpbd.mesh import Mesh
from xpbd.solver import damp_velocities


class System:
    """Particles + the XPBD step loop.

    Attributes
    ----------
    X : (N, 3) float64
        Current positions.
    V : (N, 3) float64
        Current velocities.
    W : (N,) float64
        Inverse masses. ``W[i] == 0`` means vertex i is pinned.
    """

    def __init__(
        self,
        X: np.ndarray,
        masses: np.ndarray,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
    ):
        X = np.ascontiguousarray(X, dtype=np.float64)
        masses = np.ascontiguousarray(masses, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"X must be (N, 3); got {X.shape}")
        if masses.shape != (X.shape[0],):
            raise ValueError(f"masses shape {masses.shape} != ({X.shape[0]},)")
        if (masses <= 0).any():
            raise ValueError("masses must be > 0; use pin() for fixed verts")

        self.X = X
        self.V = np.zeros_like(X)
        self.masses = masses
        self.W = 1.0 / masses
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.constraints: list[ConstraintGroup] = []
        self.colliders: list = []

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
        density: float,
        **kw,
    ) -> "System":
        masses = mesh.vertex_masses(density)
        return cls(mesh.V.copy(), masses, **kw)

    def pin(self, indices: Iterable[int]) -> None:
        """Make the listed vertices kinematic by setting their inverse mass to 0."""
        self.W[list(indices)] = 0.0
        self.V[self.W == 0.0] = 0.0

    def add_constraint(self, group: ConstraintGroup) -> None:
        self.constraints.append(group)

    def add_collider(self, collider) -> None:
        self.colliders.append(collider)

    # ------------------------------------------------------------------ step

    def step(
        self,
        dt: float,
        iters: int = 1,
        k_damp: float = 0.0,
        restitution: float = 0.0,
        friction: float = 0.0,
        solver: str = "jacobi",
        contact_skin: float = 0.0,
    ) -> None:
        """One XPBD step (Algorithm 1 in Macklin et al. 2016).

        Parameters
        ----------
        solver : {"jacobi", "gauss-seidel"}
            "jacobi" — each constraint group computed from the same P.
            "gauss-seidel" — graph-colored sequential processing.
        """
        if solver not in ("jacobi", "gauss-seidel"):
            raise ValueError(
                f"solver must be 'jacobi' or 'gauss-seidel', got {solver!r}"
            )
        free = self.W > 0.0

        # (1) Apply external forces (gravity)
        self.V[free] += dt * self.gravity

        # (2) Damping (PBD §3.5 rigid-mode preserving)
        if k_damp > 0.0:
            damp_velocities(self.X, self.V, self.W, k_damp)

        # Snapshot pre-step velocity for restitution/friction
        V_pre = self.V.copy()

        # (3) Predict position
        P = self.X + dt * self.V

        # (4) Generate per-step collision constraints
        coll = generate_collision_constraints(
            self.X, P, self.colliders, free, skin=contact_skin
        )

        # (5) Initialize multipliers to zero (XPBD line 4)
        for group in self.constraints:
            group.lam[:] = 0.0
        coll.lam[:] = 0.0

        # (6) Solver iterations — XPBD: no k' linearization needed
        X_prev = self.X  # for damping term in constraints

        if solver == "jacobi":
            for _ in range(iters):
                for group in self.constraints:
                    dP, d_lam = group.project_batch(P, self.W, X_prev, dt)
                    P += dP
                    group.lam += d_lam
                if coll.idx.shape[0] > 0:
                    dP, _ = coll.project_batch(P, self.W, X_prev, dt)
                    P += dP
        else:  # gauss-seidel
            for _ in range(iters):
                for group in self.constraints:
                    for c in range(group.n_colors):
                        dP, d_lam = group.project_color(P, self.W, X_prev, dt, c)
                        P += dP
                        group.lam += d_lam
                if coll.idx.shape[0] > 0:
                    for c in range(coll.n_colors):
                        dP, _ = coll.project_color(P, self.W, X_prev, dt, c)
                        P += dP

        # (7) Update velocity from positional change
        self.V = (P - self.X) / dt

        # (8) Commit positions
        self.X = P

        # (9) Collision velocity update (restitution + friction)
        if coll.idx.shape[0] > 0:
            self._collision_velocity_update(coll, V_pre, restitution, friction)

    def _collision_velocity_update(self, coll, V_pre, restitution, friction):
        """Apply restitution and Coulomb friction for collided particles."""
        i = coll.idx
        n = coll.normals
        v_pre = V_pre[i]
        v_post = self.V[i]

        vn_pre = np.einsum("ij,ij->i", v_pre, n)
        active = vn_pre < 0.0
        if not active.any():
            return

        vn_post = np.einsum("ij,ij->i", v_post[active], n[active])
        v_post[active] -= (vn_post[:, None] * n[active]
                           + restitution * vn_pre[active][:, None] * n[active])

        if friction > 0.0:
            vn_now = np.einsum("ij,ij->i", v_post[active], n[active])
            v_t = v_post[active] - vn_now[:, None] * n[active]
            vt_mag = np.linalg.norm(v_t, axis=1)
            cap = np.minimum(friction * np.abs(vn_pre[active]), vt_mag)
            ok = vt_mag > 1e-12
            t_hat = np.zeros_like(v_t)
            t_hat[ok] = v_t[ok] / vt_mag[ok, None]
            v_post[active] -= cap[:, None] * t_hat

        self.V[i] = v_post
