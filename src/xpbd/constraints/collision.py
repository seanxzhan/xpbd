"""Static colliders (plane, sphere, triangle mesh) + per-step collision constraints.

Collisions use zero compliance (infinite stiffness) — the constraint is
projected exactly as in PBD. No Lagrange multiplier storage needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xpbd.coloring import greedy_color
from xpbd.constraints.base import ConstraintGroup


# ---------------------------------------------------------------- colliders


@dataclass
class Plane:
    """Half-space {x : n · x ≥ d}."""
    normal: np.ndarray
    offset: float

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=np.float64)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            raise ValueError("plane normal must be nonzero")
        self.normal = n / nn
        self.offset = float(self.offset)

    def signed_distance(self, P: np.ndarray) -> np.ndarray:
        return P @ self.normal - self.offset

    def surface_normals(self, P: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.normal, P.shape).copy()

    def contact_anchors(self, X, P, free_mask, skin):
        sdf = self.signed_distance(P)
        hit = (sdf < skin) & free_mask
        if not hit.any():
            return None
        Phit = P[hit]
        n = self.surface_normals(Phit)
        offsets = np.einsum("ij,ij->i", Phit, n) - sdf[hit]
        return np.where(hit)[0], n, offsets


@dataclass
class Sphere:
    """Solid sphere obstacle."""
    center: np.ndarray
    radius: float

    def __post_init__(self):
        self.center = np.ascontiguousarray(self.center, dtype=np.float64)
        if self.center.shape != (3,):
            raise ValueError(f"sphere center must be (3,); got {self.center.shape}")
        if self.radius <= 0:
            raise ValueError("sphere radius must be > 0")
        self.radius = float(self.radius)

    def signed_distance(self, P: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(P - self.center, axis=1)
        return d - self.radius

    def surface_normals(self, P: np.ndarray) -> np.ndarray:
        d = P - self.center
        L = np.linalg.norm(d, axis=1, keepdims=True)
        ok = L > 1e-12
        n = np.zeros_like(d)
        n[ok[:, 0]] = d[ok[:, 0]] / L[ok[:, 0]]
        n[~ok[:, 0]] = np.array([1.0, 0.0, 0.0])
        return n

    def contact_anchors(self, X, P, free_mask, skin):
        sdf = self.signed_distance(P)
        hit = (sdf < skin) & free_mask
        if not hit.any():
            return None
        Phit = P[hit]
        n = self.surface_normals(Phit)
        offsets = np.einsum("ij,ij->i", Phit, n) - sdf[hit]
        return np.where(hit)[0], n, offsets


class TriangleMesh:
    """Static triangle-mesh obstacle with CCD + closest-point fallback."""

    def __init__(self, V: np.ndarray, F: np.ndarray):
        V = np.ascontiguousarray(V, dtype=np.float64)
        F = np.ascontiguousarray(F, dtype=np.int64)
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(f"V must be (Vn, 3); got {V.shape}")
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError(f"F must be (Fn, 3); got {F.shape}")
        self.V = V
        self.F = F
        self.V0 = V[F[:, 0]]
        self.V1 = V[F[:, 1]]
        self.V2 = V[F[:, 2]]
        self._e1 = self.V1 - self.V0
        self._e2 = self.V2 - self.V0
        n = np.cross(self._e1, self._e2)
        L = np.linalg.norm(n, axis=1, keepdims=True)
        if (L < 1e-20).any():
            raise ValueError("TriangleMesh has degenerate (zero-area) faces")
        self.face_normals = n / L
        self._e1e1 = np.einsum("fi,fi->f", self._e1, self._e1)
        self._e1e2 = np.einsum("fi,fi->f", self._e1, self._e2)
        self._e2e2 = np.einsum("fi,fi->f", self._e2, self._e2)
        self._det = self._e1e1 * self._e2e2 - self._e1e2 * self._e1e2

    def contact_anchors(self, X, P, free_mask, skin):
        active = np.where(free_mask)[0]
        if active.size == 0 or self.F.shape[0] == 0:
            return None

        o = X[active]
        p = P[active]
        d = p - o

        t_min, face_min = _ray_segment_min_t(
            o, d, self.V0, self._e1, self._e2, skin=skin
        )
        ccd_hit = np.isfinite(t_min)

        cp = _closest_point_on_tris(
            p, self.V0, self._e1, self._e2, self.face_normals,
            self._e1e1, self._e1e2, self._e2e2, self._det,
        )
        diff = p[:, None, :] - cp
        dist_sq = np.einsum("nfi,nfi->nf", diff, diff)
        cp_face_idx = np.argmin(dist_sq, axis=1)
        rows = np.arange(active.size)
        cp_best = cp[rows, cp_face_idx]
        n_cp_face = self.face_normals[cp_face_idx]
        sd = np.einsum("ni,ni->n", p - cp_best, n_cp_face)
        cp_hit = (sd < skin) & (~ccd_hit)

        any_hit = ccd_hit | cp_hit
        if not any_hit.any():
            return None

        local = np.where(any_hit)[0]
        normals_out = np.where(
            ccd_hit[local, None],
            self.face_normals[face_min[local]],
            self.face_normals[cp_face_idx[local]],
        )
        t_safe = np.where(ccd_hit[local], t_min[local], 0.0)
        q_ccd = o[local] + t_safe[:, None] * d[local]
        anchors = np.where(ccd_hit[local, None], q_ccd, cp_best[local])
        offsets = np.einsum("ni,ni->n", anchors, normals_out)
        return active[local], normals_out, offsets


def _ray_segment_min_t(o, d, V0, E1, E2, skin=0.0, eps=1e-12):
    """Vectorized Möller–Trumbore: smallest valid t per ray, or +inf."""
    N = o.shape[0]
    h = np.cross(d[:, None, :], E2[None, :, :])
    a = np.einsum("fi,nfi->nf", E1, h)
    parallel = np.abs(a) < eps
    inv_a = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, a))

    s = o[:, None, :] - V0[None, :, :]
    u = inv_a * np.einsum("nfi,nfi->nf", s, h)
    miss_u = (u < 0.0) | (u > 1.0)

    qv = np.cross(s, E1[None, :, :])
    v = inv_a * np.einsum("nfi,nfi->nf", d[:, None, :], qv)
    miss_v = (v < 0.0) | (u + v > 1.0)

    t = inv_a * np.einsum("fi,nfi->nf", E2, qv)
    d_len = np.linalg.norm(d, axis=1)
    upper = np.where(d_len > eps, 1.0 + skin / np.where(d_len > eps, d_len, 1.0), 0.0)
    miss_t = (t < 0.0) | (t > upper[:, None])

    n_face = np.cross(E1, E2)
    d_dot_n = np.einsum("ni,fi->nf", d, n_face)
    miss_dir = d_dot_n >= 0.0

    miss = parallel | miss_u | miss_v | miss_t | miss_dir
    t = np.where(miss, np.inf, t)
    face_min = np.argmin(t, axis=1)
    t_min = t[np.arange(N), face_min]
    return t_min, face_min


def _closest_point_on_tris(p, V0, E1, E2, n, e1e1, e1e2, e2e2, det):
    """Vectorized closest point on each triangle for each particle."""
    diff0 = p[:, None, :] - V0[None, :, :]
    dn = np.einsum("nfi,fi->nf", diff0, n)
    proj = p[:, None, :] - dn[..., None] * n[None, :, :]

    diff_proj = proj - V0[None, :, :]
    re1 = np.einsum("nfi,fi->nf", diff_proj, E1)
    re2 = np.einsum("nfi,fi->nf", diff_proj, E2)
    s = (e2e2[None, :] * re1 - e1e2[None, :] * re2) / det[None, :]
    t = (e1e1[None, :] * re2 - e1e2[None, :] * re1) / det[None, :]
    inside = (s >= 0.0) & (t >= 0.0) & (s + t <= 1.0)

    u1 = np.clip(np.einsum("nfi,fi->nf", diff0, E1) / e1e1[None, :], 0.0, 1.0)
    cp1 = V0[None, :, :] + u1[..., None] * E1[None, :, :]
    u2 = np.clip(np.einsum("nfi,fi->nf", diff0, E2) / e2e2[None, :], 0.0, 1.0)
    cp2 = V0[None, :, :] + u2[..., None] * E2[None, :, :]
    e3 = E2 - E1
    e33 = np.einsum("fi,fi->f", e3, e3)
    diff1 = p[:, None, :] - (V0 + E1)[None, :, :]
    u3 = np.clip(np.einsum("nfi,fi->nf", diff1, e3) / e33[None, :], 0.0, 1.0)
    cp3 = (V0 + E1)[None, :, :] + u3[..., None] * e3[None, :, :]

    pb = p[:, None, :]
    d1 = np.einsum("nfi,nfi->nf", cp1 - pb, cp1 - pb)
    d2 = np.einsum("nfi,nfi->nf", cp2 - pb, cp2 - pb)
    d3 = np.einsum("nfi,nfi->nf", cp3 - pb, cp3 - pb)
    edge_d = np.stack([d1, d2, d3], axis=-1)
    edge_cp = np.stack([cp1, cp2, cp3], axis=-2)
    e_idx = np.argmin(edge_d, axis=-1)
    cp_edge = np.take_along_axis(
        edge_cp, e_idx[..., None, None].repeat(3, axis=-1), axis=-2
    ).squeeze(-2)

    return np.where(inside[..., None], proj, cp_edge)


# ----------------------------------------------------- collision constraint


class CollisionGroup(ConstraintGroup):
    """One-shot inequality constraints generated at predict time.

    Zero compliance — projected exactly like PBD (no multiplier storage).
    """

    def __init__(self, idx, normals, offsets):
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        normals = np.ascontiguousarray(normals, dtype=np.float64)
        offsets = np.ascontiguousarray(offsets, dtype=np.float64)
        if idx.ndim != 1:
            raise ValueError(f"idx must be (M,); got {idx.shape}")
        self.idx = idx
        self.normals = normals
        self.offsets = offsets
        M = idx.shape[0]
        self.compliance = np.zeros(M, dtype=np.float64)
        self.damping = None
        self.lam = np.zeros(M, dtype=np.float64)
        self._colors: np.ndarray | None = None

    def _project(self, P, W, i, n, offsets):
        out = np.zeros_like(P)
        if i.shape[0] == 0:
            return out
        C = np.einsum("ij,ij->i", P[i], n) - offsets
        active = (C < 0.0) & (W[i] > 0.0)
        if not active.any():
            return out
        np.add.at(out, i[active], (-C[active])[:, None] * n[active])
        return out

    def project_batch(self, P, W, X_prev, dt):
        dP = self._project(P, W, self.idx, self.normals, self.offsets)
        return dP, np.zeros(self.idx.shape[0], dtype=np.float64)

    # ------------------------------------------------------- Gauss–Seidel

    def _ensure_colors(self):
        if self._colors is None:
            self._colors = greedy_color(self.idx.reshape(-1, 1))
        return self._colors

    @property
    def n_colors(self):
        c = self._ensure_colors()
        return int(c.max() + 1) if c.size else 1

    def project_color(self, P, W, X_prev, dt, c):
        colors = self._ensure_colors()
        mask = colors == c
        dP = self._project(P, W, self.idx[mask], self.normals[mask], self.offsets[mask])
        return dP, np.zeros(self.idx.shape[0], dtype=np.float64)


# ------------------------------------------- per-step constraint generation


def generate_collision_constraints(X, P, colliders, free_mask, skin=0.0):
    """Build a CollisionGroup from the X→P predicted move."""
    idx_list, n_list, off_list = [], [], []
    for c in colliders:
        out = c.contact_anchors(X, P, free_mask, skin)
        if out is None:
            continue
        idx, n, offsets = out
        idx_list.append(idx)
        n_list.append(n)
        off_list.append(offsets)

    if not idx_list:
        return CollisionGroup(
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    return CollisionGroup(
        np.concatenate(idx_list),
        np.concatenate(n_list, axis=0),
        np.concatenate(off_list),
    )
