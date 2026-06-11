"""Inflated balloon — XPBD version.

Volume constraint on a closed mesh (icosphere) with k_pressure > 1.
"""
from __future__ import annotations

import argparse

import numpy as np

from xpbd import Bend, Stretch, Volume, System, build_mesh
from xpbd.viz import Viewer


def make_icosphere(subdivisions: int = 2, radius: float = 0.5):
    """Generate an icosphere by subdividing an icosahedron."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    V = np.array(verts, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)

    # Normalize to unit sphere
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    # Subdivide
    for _ in range(subdivisions):
        V, F = _subdivide(V, F)

    V *= radius
    return V, F


def _subdivide(V, F):
    """Subdivide each triangle into 4 by splitting edges."""
    edge_midpoints = {}
    new_verts = list(V)

    def get_midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key in edge_midpoints:
            return edge_midpoints[key]
        mid = (V[a] + V[b]) / 2.0
        mid /= np.linalg.norm(mid)
        idx = len(new_verts)
        new_verts.append(mid)
        edge_midpoints[key] = idx
        return idx

    new_faces = []
    for tri in F:
        a, b, c = tri
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.extend([
            [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]
        ])

    return np.array(new_verts, dtype=np.float64), np.array(new_faces, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdivisions", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dt", type=float, default=1.0 / 60)
    ap.add_argument("--stretch-compliance", type=float, default=1e-5)
    ap.add_argument("--bend-compliance", type=float, default=1e-2)
    ap.add_argument("--volume-compliance", type=float, default=0.0)
    ap.add_argument("--k-pressure", type=float, default=1.5)
    ap.add_argument("--solver", choices=["jacobi", "gauss-seidel"], default="gauss-seidel")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-frames", type=int, default=120)
    args = ap.parse_args()

    V, F = make_icosphere(subdivisions=args.subdivisions)
    # Move up so it doesn't clip the floor
    V[:, 1] += 1.0
    mesh = build_mesh(V, F)

    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, 0.0, 0.0))
    sys.add_constraint(Stretch.from_mesh(mesh, compliance=args.stretch_compliance))
    sys.add_constraint(Bend.from_mesh(mesh, compliance=args.bend_compliance))
    sys.add_constraint(Volume.from_mesh(mesh, k_pressure=args.k_pressure,
                                        compliance=args.volume_compliance))

    if args.smoke:
        import time
        t0 = time.time()
        for _ in range(args.smoke_frames):
            sys.step(dt=args.dt, iters=args.iters, solver=args.solver)
        elapsed = time.time() - t0
        print(f"{args.smoke_frames} frames in {elapsed:.3f}s "
              f"({args.smoke_frames / elapsed:.1f} fps)")
        r = np.linalg.norm(sys.X - sys.X.mean(axis=0), axis=1).mean()
        print(f"mean radius: {r:.4f} (initial: 0.5)")
        return

    viewer = Viewer(sys, mesh.F, name="balloon")
    viewer.run(dt=args.dt, iters=args.iters, solver=args.solver)


if __name__ == "__main__":
    main()
