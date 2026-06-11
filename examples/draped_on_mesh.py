"""Cloth draped on a triangle-mesh obstacle — XPBD version."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from xpbd import Bend, Stretch, System, build_mesh
from xpbd.constraints.collision import TriangleMesh
from xpbd.io import fix_winding, load_obj
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
    ap.add_argument("--obj", type=str, default=None,
                    help="Path to OBJ obstacle (default: data/spot_simplified.obj)")
    ap.add_argument("--convex-hull", action="store_true",
                    help="Use convex hull of obstacle mesh")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dt", type=float, default=1.0 / 60)
    ap.add_argument("--stretch-compliance", type=float, default=1e-5)
    ap.add_argument("--bend-compliance", type=float, default=1e-2)
    ap.add_argument("--solver", choices=["jacobi", "gauss-seidel"], default="gauss-seidel")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-frames", type=int, default=120)
    args = ap.parse_args()

    # Load obstacle mesh
    if args.obj is None:
        obj_path = Path(__file__).parent.parent / "data" / "spot_simplified.obj"
    else:
        obj_path = Path(args.obj)

    obs_V, obs_F = load_obj(obj_path)
    obs_F = fix_winding(obs_V, obs_F)

    if args.convex_hull:
        from xpbd.io import convex_hull
        obs_V, obs_F = convex_hull(obs_V)

    # Scale and center obstacle
    obs_V -= obs_V.mean(axis=0)
    scale = 1.0 / max(obs_V.max() - obs_V.min())
    obs_V *= scale

    collider = TriangleMesh(obs_V, obs_F)

    # Cloth mesh
    mesh = make_grid_mesh(args.n, side=0.8, height=1.5)
    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, -9.81, 0.0))
    sys.add_constraint(Stretch.from_mesh(mesh, compliance=args.stretch_compliance))
    sys.add_constraint(Bend.from_mesh(mesh, compliance=args.bend_compliance))
    sys.add_collider(collider)

    if args.smoke:
        import time
        t0 = time.time()
        for _ in range(args.smoke_frames):
            sys.step(dt=args.dt, iters=args.iters, solver=args.solver,
                     friction=0.3, contact_skin=1e-3)
        elapsed = time.time() - t0
        print(f"{args.smoke_frames} frames in {elapsed:.3f}s "
              f"({args.smoke_frames / elapsed:.1f} fps)")
        return

    import polyscope as ps
    viewer = Viewer(sys, mesh.F, name="cloth")
    ps.register_surface_mesh("obstacle", obs_V, obs_F).set_color((0.6, 0.4, 0.3))
    viewer.add_floor(y=-0.8)
    viewer.run(dt=args.dt, iters=args.iters, solver=args.solver,
               friction=0.3, contact_skin=1e-3)


if __name__ == "__main__":
    main()
