"""Baseline controller: geometric pure pursuit.

Pure pursuit is deliberately simple: pick a point on the path a fixed
`lookahead` distance ahead of the vehicle, steer so that the rear axle follows
a circular arc through that point, and hold the speed with a proportional
acceleration law.  It uses no model prediction and no optimisation, which is
exactly why it is a useful reference for the MPC experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import BicycleParams, bicycle_step, wrap_angle
from .sim import ClosedLoopResult


@dataclass
class PurePursuitConfig:
    wheelbase: float = 2.5
    lookahead: float = 4.0      # arc-length advance along the path [m]
    speed: float = 2.0          # target speed [m/s]
    k_speed: float = 2.0        # [1/s], P gain of the speed loop
    a_max: float = 2.0          # [m/s^2]
    delta_max: float = 0.5236   # [rad], the same limit the MPC gets
    ddelta_max: float = 0.05    # [rad per step], same rate limit as the MPC


def pure_pursuit_control(
    state, path, cfg: PurePursuitConfig, delta_prev: float = 0.0
) -> np.ndarray:
    """One pure-pursuit control `[a, delta]` for the given state and path.

    The steering command is clipped to `delta_max` and slewed at
    `ddelta_max` so that the baseline faces exactly the same actuator limits
    as the MPC in Phase 3.
    """
    x, y, theta, v = state
    s_now, _ = path.project(np.array([x, y]))
    s_target = min(s_now + cfg.lookahead, path.length)
    # path.at works with arrays; take the single sample back to a float
    xt, yt, _ = (float(v[0]) for v in path.at(np.array([s_target])))

    alpha = float(wrap_angle(np.arctan2(yt - y, xt - x) - theta))
    # The target is selected by path arc length, but the circle-through-target
    # geometry uses its actual straight-line distance from the rear axle.
    target_distance = float(np.hypot(xt - x, yt - y))
    delta = (np.arctan2(2.0 * cfg.wheelbase * np.sin(alpha), target_distance)
             if target_distance > 1e-12 else 0.0)
    delta = float(np.clip(delta, -cfg.delta_max, cfg.delta_max))
    delta = float(np.clip(delta, delta_prev - cfg.ddelta_max, delta_prev + cfg.ddelta_max))

    acc = float(np.clip(cfg.k_speed * (cfg.speed - v), -cfg.a_max, cfg.a_max))
    return np.array([acc, delta])


def simulate_pure_pursuit(
    path, state0, n_steps: int, dt: float, cfg: PurePursuitConfig | None = None
) -> ClosedLoopResult:
    """Closed loop of the pure-pursuit baseline, same result type as the MPC."""
    cfg = cfg or PurePursuitConfig()
    params = BicycleParams(wheelbase=cfg.wheelbase)
    state = np.asarray(state0, dtype=float).reshape(4)

    states = np.empty((n_steps + 1, 4))
    controls = np.empty((n_steps, 2))
    states[0] = state
    delta_prev = 0.0
    for k in range(n_steps):
        u = pure_pursuit_control(state, path, cfg, delta_prev)
        delta_prev = u[1]
        controls[k] = u
        state = bicycle_step(state, u, dt, params)
        states[k + 1] = state

    return ClosedLoopResult(
        t=dt * np.arange(n_steps + 1),
        states=states,
        controls=controls,
        solve_times=np.zeros(n_steps),
        statuses=["pure_pursuit"] * n_steps,
        objectives=np.full(n_steps, np.nan),
    )
