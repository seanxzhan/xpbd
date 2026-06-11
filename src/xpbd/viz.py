"""Polyscope adapter for live XPBD simulation.

Live UI knobs (in the imgui sidebar):
* Play / Reset buttons.
* iters slider
* |gravity| slider
* Per-constraint compliance slider (log-scale)
* k_damp slider for rigid-mode-preserving damping.
* friction μ and restitution e sliders
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from xpbd.system import System


class Viewer:
    """Tiny polyscope wrapper for cloth/soft-body XPBD scenes."""

    def __init__(
        self,
        sys: System,
        F: np.ndarray,
        name: str = "cloth",
    ):
        import polyscope as ps

        self._ps = ps
        if not getattr(ps, "_xpbd_initialized", False):
            ps.init()
            ps._xpbd_initialized = True
        ps.set_up_dir("y_up")
        ps.set_front_dir("z_front")
        ps.set_ground_plane_mode("none")

        self.sys = sys
        self.F = np.ascontiguousarray(F, dtype=np.int64)
        self.mesh = ps.register_surface_mesh(name, sys.X, self.F, edge_width=1.0)
        self.mesh.set_smooth_shade(True)

        self._X0 = sys.X.copy()
        self._V0 = sys.V.copy()
        self._W0 = sys.W.copy()
        self._gravity0 = sys.gravity.copy()

        g_mag = float(np.linalg.norm(self._gravity0))
        if g_mag > 1e-12:
            self._gravity_dir = self._gravity0 / g_mag
            self._gravity_mag = g_mag
        else:
            self._gravity_dir = None
            self._gravity_mag = 0.0

        self._dt: float = 1.0 / 60.0
        self._iters: int = 10
        self._k_damp: float = 0.0
        self._restitution: float = 0.0
        self._friction: float = 0.0
        self._solver: str = "jacobi"
        self._contact_skin: float = 0.0
        self._step_callback: Optional[Callable[[int], None]] = None
        self._ui_callback: Optional[Callable[[], None]] = None
        self._frame: int = 0
        self._playing: bool = True

    def add_floor(self, y: float = 0.0):
        from xpbd.constraints.collision import Plane

        self.sys.add_collider(Plane(normal=(0.0, 1.0, 0.0), offset=y))
        s = 5.0
        y_visual = y - 0.01
        floor_V = np.array([
            [-s, y_visual, -s], [s, y_visual, -s],
            [s, y_visual, s], [-s, y_visual, s],
        ])
        floor_F = np.array([[0, 1, 2], [0, 2, 3]])
        self._ps.register_surface_mesh("floor", floor_V, floor_F).set_color((0.5, 0.5, 0.5))

    def add_sphere_obstacle(self, center, radius: float, visual_inset: float = 0.0):
        from xpbd.constraints.collision import Sphere

        c = np.asarray(center, dtype=np.float64)
        self.sys.add_collider(Sphere(center=c, radius=radius))
        pc = self._ps.register_point_cloud(f"obstacle_{len(self.sys.colliders)}",
                                            c.reshape(1, 3))
        pc.set_radius(max(radius - visual_inset, 1e-6), relative=False)
        pc.set_color((0.7, 0.3, 0.3))

    def reset(self):
        self.sys.X[:] = self._X0
        self.sys.V[:] = self._V0
        self.sys.W[:] = self._W0
        self.sys.masses = 1.0 / np.where(self.sys.W > 0.0, self.sys.W, 1.0)
        self.sys.gravity = self._gravity0.copy()
        if self._gravity_dir is not None:
            self._gravity_mag = float(np.linalg.norm(self._gravity0))
        self._frame = 0
        self.mesh.update_vertex_positions(self.sys.X)

    def run(
        self,
        dt: float = 1.0 / 60,
        iters: int = 10,
        k_damp: float = 0.0,
        restitution: float = 0.0,
        friction: float = 0.0,
        solver: str = "jacobi",
        contact_skin: float = 0.0,
        on_step: Optional[Callable[[int], None]] = None,
        on_ui: Optional[Callable[[], None]] = None,
    ):
        if solver not in ("jacobi", "gauss-seidel"):
            raise ValueError(
                f"solver must be 'jacobi' or 'gauss-seidel', got {solver!r}"
            )
        self._dt = dt
        self._iters = iters
        self._k_damp = k_damp
        self._restitution = restitution
        self._friction = friction
        self._solver = solver
        self._contact_skin = contact_skin
        self._step_callback = on_step
        self._ui_callback = on_ui
        self._ps.set_user_callback(self._tick)
        self._ps.show()

    def _tick(self):
        if self._playing:
            self.sys.step(
                dt=self._dt,
                iters=self._iters,
                k_damp=self._k_damp,
                restitution=self._restitution,
                friction=self._friction,
                solver=self._solver,
                contact_skin=self._contact_skin,
            )
            self.mesh.update_vertex_positions(self.sys.X)
            self._frame += 1
            if self._step_callback is not None:
                self._step_callback(self._frame)

        self._draw_ui()

    def _draw_ui(self):
        import polyscope.imgui as psim

        _, self._playing = psim.Checkbox("Play", self._playing)
        psim.SameLine()
        if psim.Button("Reset"):
            self.reset()

        psim.Separator()
        if psim.RadioButton("Jacobi", self._solver == "jacobi"):
            self._solver = "jacobi"
        psim.SameLine()
        if psim.RadioButton("Gauss-Seidel", self._solver == "gauss-seidel"):
            self._solver = "gauss-seidel"

        _, self._iters = psim.SliderInt("iters", int(self._iters), 1, 100)
        _, self._k_damp = psim.SliderFloat("k_damp", float(self._k_damp), 0.0, 0.5)

        if self._gravity_dir is not None:
            changed, self._gravity_mag = psim.SliderFloat(
                "|gravity|", float(self._gravity_mag), 0.0, 20.0
            )
            if changed:
                self.sys.gravity = self._gravity_dir * self._gravity_mag

        if self.sys.colliders:
            psim.Separator()
            _, self._restitution = psim.SliderFloat(
                "restitution e", float(self._restitution), 0.0, 1.0
            )
            _, self._friction = psim.SliderFloat(
                "friction μ", float(self._friction), 0.0, 1.0
            )
            _, self._contact_skin = psim.SliderFloat(
                "contact skin", float(self._contact_skin), 0.0, 0.05
            )

        if self._ui_callback is not None:
            psim.Separator()
            self._ui_callback()

        psim.Separator()
        psim.Text(f"frame {self._frame}    dt {self._dt:.4f}")
