"""Simulation loops for the MPC path-tracking project.

Two loops:

* :func:`simulate_open_loop` - a fixed sequence of `(a, delta)` inputs
  applied to the bicycle model (Phase 0).
* :func:`simulate_closed_loop` - the receding-horizon loop: the controller
  solves, only the first input is applied, the plant advances one sample
  (Phase 1-3).  The plant step is passed in, so the same loop drives the
  double integrator and the bicycle model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CONTROL_DIM, DEFAULT_PARAMS, STATE_DIM, BicycleParams, bicycle_step


@dataclass
class OpenLoopResult:
    """Container for one open-loop simulation.

    Attributes:
        t: time stamps, shape `(N+1,)`, starting at 0.
        states: states, shape `(N+1, 4)`, `states[k] = X(k)`.
        controls: applied inputs, shape `(N, 2)`, `controls[k]` acts on
            `[t[k], t[k+1])`.
        dt: sampling time [s].
    """

    t: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    dt: float


def simulate_open_loop(
    state0: np.ndarray,
    controls: np.ndarray,
    dt: float,
    params: BicycleParams = DEFAULT_PARAMS,
) -> OpenLoopResult:
    """Apply a fixed input sequence to the bicycle model.

    Args:
        state0: initial state `[x, y, theta, v]`.
        controls: `(N, 2)` array of `[a, delta]` held over each step.
        dt: sampling time T [s].
        params: fixed vehicle parameters.

    Returns:
        An :class:`OpenLoopResult` with `N + 1` states.
    """
    state0 = np.asarray(state0, dtype=float).reshape(STATE_DIM)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != CONTROL_DIM:
        raise ValueError(
            f"controls must have shape (N, 2), got {controls.shape}"
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be a positive finite number, got {dt}")

    n_steps = controls.shape[0]
    states = np.empty((n_steps + 1, STATE_DIM))
    states[0] = state0
    for k in range(n_steps):
        states[k + 1] = bicycle_step(states[k], controls[k], dt, params)

    t = dt * np.arange(n_steps + 1)
    return OpenLoopResult(t=t, states=states, controls=controls, dt=dt)

# --------------------------------------------------------------------------
# Closed loop
# --------------------------------------------------------------------------

@dataclass
class ClosedLoopResult:
    """One MPC closed-loop run."""

    t: np.ndarray              # (n_steps + 1,)
    states: np.ndarray         # (n_steps + 1, state_dim)
    controls: np.ndarray       # (n_steps, control_dim)
    solve_times: np.ndarray    # (n_steps,)
    statuses: list             # per-step solver status
    objectives: np.ndarray     # (n_steps,)


class SimulationError(RuntimeError):
    """Abort an invalid step; solver failures stop before plant application."""

    def __init__(self, step, state, status):
        self.step = int(step)
        self.state = np.asarray(state, dtype=float).copy()
        self.status = str(status)
        super().__init__(
            f"simulation stopped at step {step}, state={self.state.tolist()}: {status}"
        )


def simulate_closed_loop(controller, state0, reference, n_steps, dt, plant_step):
    """Roll the MPC out for `n_steps` samples.

    Args:
        controller: object with `horizon`, `control_dim` and
            `solve(x0, ref_states, ref_controls, u_prev) -> MpcResult`.
        state0: initial state.
        reference: :class:`mpc_pt.paths.Reference` for the whole run.
        n_steps: number of control intervals.
        dt: sampling time.
        plant_step: `f(state, control) -> next state` of the *plant*
            (same model as the controller here, but kept separate on purpose).
    """
    state0 = np.asarray(state0, dtype=float).reshape(-1)
    if not np.all(np.isfinite(state0)):
        raise ValueError("initial state must be finite")
    if not np.isfinite(dt) or dt <= 0 or not np.isclose(dt, reference.dt):
        raise ValueError("dt must be positive and match the reference sampling time")
    n_u = controller.control_dim
    states = np.empty((n_steps + 1, state0.size))
    controls = np.empty((n_steps, n_u))
    solve_times = np.empty(n_steps)
    objectives = np.empty(n_steps)
    statuses: list = []
    states[0] = state0

    u_prev = np.zeros(n_u)
    for k in range(n_steps):
        window = reference.window(k, controller.horizon)
        result = controller.solve(states[k], window.states, window.controls, u_prev)
        if result.status != "optimal":
            raise SimulationError(k, states[k], result.status)
        if (not all(np.all(np.isfinite(value)) for value in
                    (result.u0, result.states, result.controls, result.objective))
                or not np.isfinite(result.constraint_violation)
                or result.constraint_violation > 1e-6):
            raise SimulationError(k, states[k], "invalid or infeasible solver output")
        u = np.asarray(result.u0, dtype=float).reshape(n_u)
        controls[k] = u
        solve_times[k] = result.solve_time
        objectives[k] = result.objective
        statuses.append(result.status)
        u_prev = u
        states[k + 1] = plant_step(states[k], u)
        if not np.all(np.isfinite(states[k + 1])):
            raise SimulationError(k, states[k], "non-finite plant state")

    return ClosedLoopResult(
        t=dt * np.arange(n_steps + 1),
        states=states,
        controls=controls,
        solve_times=solve_times,
        statuses=statuses,
        objectives=objectives,
    )
