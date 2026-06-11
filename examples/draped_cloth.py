"""Cloth falling on a floor + sphere obstacle — XPBD version."""
from __future__ import annotations

import argparse

import numpy as np

from xpbd import Bend, Stretch, System, build_mesh
from xpbd.viz import Viewer


def make_grid_mesh(n: int = 32, side: float = 1.0, height: float = 2.0):
    xs = np.linspace(-side, side, n)
    zs = np.linspace(-side, side, n)
    XX, ZZ = np.meshgrid(xs, zs, indexing="xy")
    V = np.stack([XX.ravel(), np.full(n * n, height), ZZ.ravel()], axis=1)
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + (i + 1)
            c = (j + 1) * n + i
            d = (j + 1) * n + (i + 1)
            if (i + j) % 2 == 0:
                F.extend([[a, b, d], [a, d, c]])
            else:
                F.extend([[a, b, c], [b, d, c]])
    return build_mesh(V, np.array(F, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dt", type=float, default=1.0 / 60)
    ap.add_argument("--stretch-compliance", type=float, default=1e-4)
    ap.add_argument("--bend-compliance", type=float, default=1e-1)
    ap.add_argument("--solver", choices=["jacobi", "gauss-seidel"], default="gauss-seidel")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-frames", type=int, default=120)
    args = ap.parse_args()

    mesh = make_grid_mesh(args.n)
    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, -9.81, 0.0))
    sys.add_constraint(Stretch.from_mesh(mesh, compliance=args.stretch_compliance))
    sys.add_constraint(Bend.from_mesh(mesh, compliance=args.bend_compliance))

    if args.smoke:
        import time
        from xpbd import Plane, Sphere
        sys.add_collider(Plane(normal=(0.0, 1.0, 0.0), offset=0.0))
        sys.add_collider(Sphere(center=np.array([0.0, 0.8, 0.0]), radius=0.5))
        t0 = time.time()
        for _ in range(args.smoke_frames):
            sys.step(dt=args.dt, iters=args.iters, restitution=0.0,
                     friction=0.3, solver=args.solver)
        elapsed = time.time() - t0
        print(f"{args.smoke_frames} frames in {elapsed:.3f}s "
              f"({args.smoke_frames / elapsed:.1f} fps)")
        return

    viewer = Viewer(sys, mesh.F, name="cloth")
    viewer.add_floor(y=0.0)
    viewer.add_sphere_obstacle(center=(0.0, 0.8, 0.0), radius=0.5)
    viewer.run(dt=args.dt, iters=args.iters, restitution=0.0, friction=0.3,
               solver=args.solver)


if __name__ == "__main__":
    main()
