"""Wind-blown cloth — XPBD version."""
from __future__ import annotations

import argparse

import numpy as np

from xpbd import Bend, Stretch, System, build_mesh
from xpbd.viz import Viewer


def make_grid_mesh(n: int = 32, side: float = 1.0):
    xs = np.linspace(-side, side, n)
    ys = np.linspace(-side, side, n)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    V = np.stack([XX.ravel(), YY.ravel(), np.zeros(n * n)], axis=1)
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
    return build_mesh(V, np.array(F, dtype=np.int64)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dt", type=float, default=1.0 / 60)
    ap.add_argument("--stretch-compliance", type=float, default=1e-5)
    ap.add_argument("--bend-compliance", type=float, default=1e-2)
    ap.add_argument("--k-damp", type=float, default=0.01)
    ap.add_argument("--solver", choices=["jacobi", "gauss-seidel"], default="gauss-seidel")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-frames", type=int, default=120)
    args = ap.parse_args()

    mesh, n = make_grid_mesh(args.n)
    sys = System.from_mesh(mesh, density=1.0, gravity=(0.0, -9.81, 0.0))
    sys.add_constraint(Stretch.from_mesh(mesh, compliance=args.stretch_compliance))
    sys.add_constraint(Bend.from_mesh(mesh, compliance=args.bend_compliance))

    # Pin the top 5% of vertices
    n_pin = max(1, int(0.05 * mesh.n_verts))
    top_idx = np.argsort(mesh.V[:, 1])[-n_pin:]
    sys.pin(top_idx)

    rng = np.random.default_rng(42)

    # Ornstein-Uhlenbeck wind
    wind = np.array([0.0, 0.0, 5.0])
    theta = 0.1  # mean reversion
    sigma = 2.0

    def wind_step():
        nonlocal wind
        wind += theta * (np.array([0.0, 0.0, 5.0]) - wind) * args.dt
        wind += sigma * np.sqrt(args.dt) * rng.standard_normal(3)
        free = sys.W > 0.0
        sys.V[free] += args.dt * wind * sys.W[free, None]

    if args.smoke:
        import time
        t0 = time.time()
        for _ in range(args.smoke_frames):
            wind_step()
            sys.step(dt=args.dt, iters=args.iters, k_damp=args.k_damp,
                     solver=args.solver)
        elapsed = time.time() - t0
        print(f"{args.smoke_frames} frames in {elapsed:.3f}s "
              f"({args.smoke_frames / elapsed:.1f} fps)")
        return

    def on_step(frame):
        wind_step()

    viewer = Viewer(sys, mesh.F, name="cloth")
    viewer.run(dt=args.dt, iters=args.iters, k_damp=args.k_damp,
               solver=args.solver, on_step=on_step)


if __name__ == "__main__":
    main()
