"""XPBD-specific: simple harmonic oscillator (paper Fig. 2).

A distance constraint with compliance α, mass m, rest=1, initial position x0=1.5.
The system should oscillate with period T = 2π√(mα) regardless of iteration count.
"""
from __future__ import annotations

import numpy as np

from xpbd import Stretch, System


def _run_oscillator(alpha: float, mass: float, x0: float, dt: float, iters: int, n_steps: int):
    """Run a 1-D oscillator and return the position trajectory."""
    X = np.array([[0.0, 0.0, 0.0], [x0, 0.0, 0.0]])
    sys = System(X, np.array([mass, mass]), gravity=(0.0, 0.0, 0.0))
    sys.pin([0])
    sys.add_constraint(Stretch(np.array([[0, 1]]), np.array([1.0]), compliance=alpha))

    positions = []
    for _ in range(n_steps):
        sys.step(dt=dt, iters=iters)
        positions.append(float(np.linalg.norm(sys.X[1])))
    return np.array(positions)


def _measure_period(traj: np.ndarray, dt: float) -> float:
    """Measure oscillation period from zero-crossings of (traj - mean)."""
    centered = traj - traj.mean()
    crossings = np.where(np.diff(np.sign(centered)))[0]
    if len(crossings) < 4:
        return float("inf")
    # Each pair of consecutive crossings is half a period
    half_periods = np.diff(crossings) * dt
    # Take full periods (pairs of half-periods), truncate to equal length
    n_pairs = min(len(half_periods[::2]), len(half_periods[1::2]))
    if n_pairs == 0:
        return float(np.median(half_periods)) * 2
    full_periods = half_periods[:2*n_pairs:2] + half_periods[1:2*n_pairs:2]
    return float(np.median(full_periods))


def test_period_matches_analytic():
    """XPBD oscillation period ≈ 2π√(mα) for moderate compliance."""
    alpha = 0.001
    mass = 1.0
    dt = 1.0 / 120
    iters = 50
    n_steps = 2000

    T_analytic = 2.0 * np.pi * np.sqrt(mass * alpha)
    traj = _run_oscillator(alpha, mass, 1.5, dt, iters, n_steps)
    T_measured = _measure_period(traj, dt)

    # Allow 10% tolerance due to implicit damping
    assert abs(T_measured - T_analytic) / T_analytic < 0.10, (
        f"T_measured={T_measured:.4f}, T_analytic={T_analytic:.4f}"
    )


def test_period_independent_of_iterations():
    """Period should be roughly the same regardless of iteration count."""
    alpha = 0.001
    mass = 1.0
    dt = 1.0 / 120
    n_steps = 2000

    periods = []
    for iters in (10, 30, 50, 100):
        traj = _run_oscillator(alpha, mass, 1.5, dt, iters, n_steps)
        T = _measure_period(traj, dt)
        periods.append(T)

    spread = max(periods) - min(periods)
    mean_T = np.mean(periods)
    # Spread should be <15% of mean (PBD would have >50% spread)
    assert spread / mean_T < 0.15, (
        f"iteration-dependent period: {periods}, spread={spread:.4f}"
    )
