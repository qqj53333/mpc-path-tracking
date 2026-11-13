"""Dynamics models for the MPC path-tracking project.

The model and reference-material provenance are recorded in `docs/model.md`.
This module implements the kinematic bicycle model with the reference point
at the rear-axle centre:

    state  X = [x, y, theta, v]^T    (m, m, rad, m/s)
    input  u = [a, delta]^T          (m/s^2, rad)

Continuous-time dynamics

    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = (v / L) * tan(delta)
    v_dot     = a

discretised with the forward-Euler rule

    X(k+1) = X(k) + T * f(X(k), u(k))

Conventions that must stay identical across the whole project:
  * theta is the heading measured counter-clockwise from the world +x axis.
  * delta > 0 steers left, i.e. it increases theta.
  * The reference scenarios drive along +x, so a straight run has theta = 0
    (option A in the project model document).
  * SI units everywhere: m, rad, s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATE_DIM = 4
CONTROL_DIM = 2

IDX_X, IDX_Y, IDX_THETA, IDX_V = 0, 1, 2, 3
IDX_A, IDX_DELTA = 0, 1

# This plant accepts only the principal steering branch below 90 degrees.
_DELTA_SINGULAR = 0.5 * np.pi
# Optimisation always stays on the principal branch, even without actuator
# limits. The margin avoids the singularity; it is not a realistic actuator bound.
STEERING_DOMAIN_MAX = _DELTA_SINGULAR - 1e-3


@dataclass(frozen=True)
class BicycleParams:
    """Fixed vehicle parameters.

    Attributes:
        wheelbase: distance between front and rear axle, L [m].
    """

    wheelbase: float = 2.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.wheelbase) or self.wheelbase <= 0.0:
            raise ValueError(f"wheelbase must be positive, got {self.wheelbase}")


DEFAULT_PARAMS = BicycleParams()


def bicycle_rhs(
    state: np.ndarray,
    control: np.ndarray,
    params: BicycleParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Continuous-time right-hand side f(X, u) of the bicycle model.

    Args:
        state: `[x, y, theta, v]`.
        control: `[a, delta]`.
        params: fixed vehicle parameters.

    Returns:
        `[x_dot, y_dot, theta_dot, v_dot]`, same shape as `state`.
    """
    state = np.asarray(state, dtype=float).reshape(STATE_DIM)
    control = np.asarray(control, dtype=float).reshape(CONTROL_DIM)

    _, _, theta, v = state
    a, delta = control
    if abs(delta) >= _DELTA_SINGULAR:
        raise ValueError(
            f"|delta| must stay below 90 deg (got {delta:.6f} rad); "
            "tan(delta) is singular there"
        )

    return np.array(
        [
            v * np.cos(theta),
            v * np.sin(theta),
            (v / params.wheelbase) * np.tan(delta),
            a,
        ]
    )


def bicycle_step(
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
    params: BicycleParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """One forward-Euler step of the bicycle model: X(k+1) = X(k) + T*f(X(k), u(k)).

    The control is held constant over the step (zero-order hold); it is the
    *start-of-step* state that is used to evaluate the derivative, which is what
    makes the circle in the open-loop test a regular polygon rather than an
    exact circle.

    Args:
        state: `[x, y, theta, v]` at the beginning of the step.
        control: `[a, delta]` held over the step.
        dt: sampling time T [s], strictly positive.
        params: fixed vehicle parameters.

    Returns:
        state at the end of the step.
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be a positive finite number, got {dt}")

    state = np.asarray(state, dtype=float).reshape(STATE_DIM)
    return state + dt * bicycle_rhs(state, control, params)

# --------------------------------------------------------------------------
# Double integrator (Phase 1)
# --------------------------------------------------------------------------

DI_STATE_DIM = 2  # state [p, v]
DI_INPUT_DIM = 1  # input a


def di_matrices(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation of the double integrator.

    p(k+1) = p(k) + T*v(k) + T^2/2*a(k)
    v(k+1) = v(k) + T*a(k)
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be a positive finite number, got {dt}")
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt], [dt]])
    return A, B


def di_step(state: np.ndarray, acc: float, dt: float) -> np.ndarray:
    """One step of the double integrator with a scalar acceleration input."""
    A, B = di_matrices(dt)
    state = np.asarray(state, dtype=float).reshape(DI_STATE_DIM)
    return A @ state + B.ravel() * float(acc)


def wrap_angle(angle):
    """Fold an angle (or array) into [-pi, pi], with atan2's branch-cut sign.

    Using sin/cos handles angles accumulated over many turns, subject to
    floating-point precision.
    """
    return np.arctan2(np.sin(angle), np.cos(angle))
