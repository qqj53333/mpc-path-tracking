"""Helpers shared by the experiment scripts.

Imported after the script's own `sys.path` bootstrap (see D-001), so
`mpc_pt` is already importable here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpc_pt.models import BicycleParams, bicycle_step
from mpc_pt.evaluation import constraint_violations
from mpc_pt.paths import reference_from_path
from mpc_pt.sim import SimulationError, simulate_closed_loop

DEFAULT_PARAMS = BicycleParams(wheelbase=2.5)


@dataclass
class TrackingRun:
    """Output of one closed-loop tracking run plus its path metrics."""

    result: object          # mpc_pt.sim.ClosedLoopResult
    s: np.ndarray           # arc length of the closest path point
    lateral: np.ndarray     # signed lateral error [m], + = left of the path
    heading_error: np.ndarray

    @property
    def t(self):
        return self.result.t


def run_tracking(path, controller, x0, n_steps, dt, speed, params=DEFAULT_PARAMS):
    """Close the loop on `path` for `n_steps` samples.

    The reference is built a whole horizon longer than the run so that the
    last MPC window never gets clamped to a standing target.
    """
    reference = reference_from_path(
        path, speed=speed, dt=dt, n_steps=n_steps + controller.horizon,
        wheelbase=params.wheelbase,
    )
    result = simulate_closed_loop(
        controller, np.asarray(x0, dtype=float), reference, n_steps, dt,
        plant_step=lambda x, u: bicycle_step(x, u, dt, params),
    )
    if controller.cfg.constraints:
        c = controller.cfg
        violations = constraint_violations(
            result.states, result.controls, dt, v_min=c.v_min, v_max=c.v_max,
            a_min=c.a_min, a_max=c.a_max, delta_max=c.delta_max,
            delta_rate_max=c.ddelta_max / dt,
        )
        if max(violations.values()) > 1e-6:
            raise SimulationError(n_steps, result.states[-1], f"actual constraint violations: {violations}")
        print(f"validity: {n_steps}/{n_steps} solves accepted; violations {violations}")
    else:
        print(f"validity: {n_steps}/{n_steps} solves accepted; actuator limits disabled, model domain enforced")
    s = np.empty(n_steps + 1)
    lateral = np.empty(n_steps + 1)
    for k, state in enumerate(result.states):
        s[k], lateral[k] = path.project(state[:2])
    heading = path.at(s)[2]
    heading_error = np.arctan2(
        np.sin(result.states[:, 2] - heading), np.cos(result.states[:, 2] - heading)
    )
    return TrackingRun(result=result, s=s, lateral=lateral, heading_error=heading_error)


def save_tracking_csv(file_path, run: TrackingRun) -> None:
    """Write t, x, y, theta, v, a, delta, s, lateral_error to CSV."""
    controls = run.result.controls
    table = np.column_stack([
        run.t,
        run.result.states,
        np.vstack([controls, np.full((1, controls.shape[1]), np.nan)]),
        run.s,
        run.lateral,
    ])
    np.savetxt(
        file_path, table, delimiter=",",
        header="t,x,y,theta,v,a,delta,s,lateral", comments="", fmt="%.6f",
    )


def describe(name: str, run: TrackingRun, dt: float) -> str:
    """One-line summary used by the experiment scripts."""
    result = run.result
    lateral = run.lateral
    return (
        f"{name:<10s} |lat| max {np.abs(lateral).max():.3f} m, "
        f"end {abs(lateral[-1]):.4f} m, rms {np.sqrt(np.mean(lateral**2)):.4f} m | "
        f"max|delta| {np.abs(result.controls[:, 1]).max():.3f} rad, "
        f"max|a| {np.abs(result.controls[:, 0]).max():.2f} m/s^2 | "
        f"solve mean {result.solve_times.mean()*1e3:.1f} ms, "
        f"max {result.solve_times.max()*1e3:.1f} ms | "
        f"status {sorted(set(result.statuses))}"
    )
