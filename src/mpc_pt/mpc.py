"""MPC controllers.

Two controllers live here:

* :class:`LinearMPC` - Phase 1, double integrator, solved as a QP with cvxpy
  and the OSQP backend.  The linear model and the quadratic cost are written
  out explicitly, so the problem is exactly the one in the task document.
* :class:`NonlinearMPC` - Phase 2/3, kinematic bicycle model, solved as a
  nonlinear program with CasADi and IPOPT.  Constraints and the input-change
  penalty are switched on through :class:`NMPCConfig`.

Both expose the same interface used by :mod:`mpc_pt.sim`:
`solve(x0, ref_states, ref_controls, u_prev) -> MpcResult`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import cvxpy as cp
import numpy as np

from .models import CONTROL_DIM, STATE_DIM, STEERING_DOMAIN_MAX, di_matrices


@dataclass
class MpcResult:
    """What one MPC solve returns."""

    u0: np.ndarray            # first control, shape (control_dim,)
    states: np.ndarray        # predicted states, (horizon + 1, state_dim)
    controls: np.ndarray      # predicted inputs, (horizon, control_dim)
    solve_time: float         # wall clock, seconds
    status: str               # "optimal" or the failure reason
    objective: float
    constraint_violation: float = 0.0
    solver_status: str = ""


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _weights(name, value, size):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must contain {size} finite nonnegative weights")


def _failed(n, nx, nu, elapsed, status):
    # Failed iterates are never valid controls and must not become warm starts.
    return MpcResult(np.full(nu, np.nan), np.full((n + 1, nx), np.nan),
                     np.full((n, nu), np.nan), elapsed, str(status), float("nan"),
                     float("inf"), str(status))


# ==========================================================================
# Phase 1 - linear MPC for the double integrator
# ==========================================================================

@dataclass
class LinearMPCConfig:
    """Tuning of the Phase 1 problem (values follow the model template)."""

    horizon: int = 20
    dt: float = 0.1
    q_position: float = 1.0
    q_velocity: float = 0.1
    r_acc: float = 0.1
    p_scale: float = 1.0      # terminal weight P = p_scale * Q
    v_max: float = 1.0        # |v| <= v_max  [m/s]
    a_max: float = 0.5        # |a| <= a_max  [m/s^2]

    def __post_init__(self):
        _positive_integer("horizon", self.horizon)
        for name in ("dt", "v_max", "a_max"):
            _positive(name, getattr(self, name))
        _weights("cost", [self.q_position, self.q_velocity, self.r_acc, self.p_scale], 4)


class LinearMPC:
    """MPC for `p' = v, v' = a` with box constraints on `v` and `a`.

    The cvxpy problem is built once; every call only updates the parameters
    (current state, reference window) and re-solves, which keeps the per-step
    time close to what the solver actually needs.
    """

    control_dim = 1
    state_dim = 2

    def __init__(self, config: LinearMPCConfig | None = None):
        self.cfg = config or LinearMPCConfig()
        n = self.cfg.horizon
        A, B = di_matrices(self.cfg.dt)

        self.Q = np.diag([self.cfg.q_position, self.cfg.q_velocity])
        self.P = self.cfg.p_scale * self.Q
        self.R = float(self.cfg.r_acc)
        # square roots of the diagonal weights: writing the quadratic form as
        # sum_squares keeps the cvxpy problem DPP, so OSQP is only set up once
        self._sqrt_q = np.sqrt(np.diag(self.Q))
        self._sqrt_p = np.sqrt(np.diag(self.P))

        self._x0 = cp.Parameter(2)
        self._ref = cp.Parameter((2, n + 1))
        X = cp.Variable((2, n + 1))
        U = cp.Variable((1, n))

        cost = 0.0
        constraints = [X[:, 0] == self._x0]
        for k in range(n):
            dx = X[:, k] - self._ref[:, k]
            cost += cp.sum_squares(cp.multiply(self._sqrt_q, dx))
            cost += self.R * cp.sum_squares(U[:, k])
            constraints.append(X[:, k + 1] == A @ X[:, k] + B @ U[:, k])
        dxN = X[:, n] - self._ref[:, n]
        cost += cp.sum_squares(cp.multiply(self._sqrt_p, dxN))

        constraints += [
            X[1, :] <= self.cfg.v_max,
            X[1, :] >= -self.cfg.v_max,
            U[0, :] <= self.cfg.a_max,
            U[0, :] >= -self.cfg.a_max,
        ]

        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        self._X, self._U = X, U

    @property
    def horizon(self) -> int:
        return self.cfg.horizon

    def solve(self, x0, ref_states, ref_controls=None, u_prev=None) -> MpcResult:
        n = self.cfg.horizon
        ref_states = np.asarray(ref_states, dtype=float)
        if ref_states.shape != (n + 1, 2):
            raise ValueError(f"ref_states must be {(n + 1, 2)}, got {ref_states.shape}")

        self._x0.value = np.asarray(x0, dtype=float).reshape(2)
        self._ref.value = ref_states.T

        tic = time.perf_counter()
        try:
            self.problem.solve(
                solver=cp.OSQP, warm_start=True,
                eps_abs=1e-9, eps_rel=1e-9, max_iter=20000, polishing=True,
            )
        except cp.error.SolverError:
            return _failed(n, 2, 1, time.perf_counter() - tic, "solver_error")
        elapsed = time.perf_counter() - tic

        if (self.problem.status not in ("optimal", "optimal_inaccurate")
                or self._X.value is None or self._U.value is None):
            return _failed(n, 2, 1, elapsed, self.problem.status)

        X = np.asarray(self._X.value).reshape(2, n + 1).T
        U = np.asarray(self._U.value).reshape(1, n).T
        violation = max(float(np.max(c.violation())) for c in self.problem.constraints)
        if (not np.all(np.isfinite(X)) or not np.all(np.isfinite(U))
                or not np.isfinite(self.problem.value) or not np.isfinite(violation)
                or violation > 1e-6):
            return _failed(n, 2, 1, elapsed, "invalid_solution")
        return MpcResult(U[0].copy(), X, U, elapsed, "optimal",
                         float(self.problem.value), violation, self.problem.status)


# ==========================================================================
# Phase 2/3 - nonlinear MPC for the kinematic bicycle model
# ==========================================================================

@dataclass
class NMPCConfig:
    """Tuning of the bicycle MPC.  Defaults follow the project model document."""

    horizon: int = 20
    dt: float = 0.1
    wheelbase: float = 2.5

    # cost weights (model.md section 5)
    q: tuple = (4.0, 4.0, 32.8, 1.0)      # [x, y, theta, v]
    r: tuple = (0.25, 3.64)                # [a, delta]
    s_change: tuple = (0.0, 0.0)           # input-change penalty, Phase 3
    p_scale: float = 3.0                   # terminal weight P = p_scale * Q

    # constraints (model.md section 6); ignored when `constraints=False`
    constraints: bool = True
    v_min: float = 0.0
    v_max: float = 8.0
    a_min: float = -2.0
    a_max: float = 2.0
    delta_max: float = 0.5236              # ~30 deg
    ddelta_max: float = 0.05               # per step, i.e. 0.5 rad/s at T=0.1

    # solver
    max_iter: int = 300
    tol: float = 1e-6

    def __post_init__(self):
        _positive_integer("horizon", self.horizon)
        _positive_integer("max_iter", self.max_iter)
        for name in ("dt", "wheelbase", "tol", "ddelta_max"):
            _positive(name, getattr(self, name))
        _weights("q", self.q, 4)
        _weights("r", self.r, 2)
        _weights("s_change", self.s_change, 2)
        _weights("p_scale", [self.p_scale], 1)
        if not np.isfinite(self.delta_max) or not 0 < self.delta_max < STEERING_DOMAIN_MAX:
            raise ValueError("delta_max must lie inside the steering model domain")
        for lo, hi in ((self.v_min, self.v_max), (self.a_min, self.a_max)):
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                raise ValueError("constraint bounds must be finite and increasing")


class NonlinearMPC:
    """Direct nonlinear MPC for the kinematic bicycle model (IPOPT)."""

    control_dim = CONTROL_DIM
    state_dim = STATE_DIM

    def __init__(self, config: NMPCConfig | None = None):
        self.cfg = c = config or NMPCConfig()
        n = c.horizon
        L, dt = c.wheelbase, c.dt

        opti = ca.Opti()
        X = opti.variable(STATE_DIM, n + 1)
        U = opti.variable(CONTROL_DIM, n)
        p_x0 = opti.parameter(STATE_DIM)
        p_ref = opti.parameter(STATE_DIM, n + 1)
        p_uref = opti.parameter(CONTROL_DIM, n)
        p_uprev = opti.parameter(CONTROL_DIM)

        Q = np.diag(np.asarray(c.q, dtype=float))
        R = np.diag(np.asarray(c.r, dtype=float))
        S = np.diag(np.asarray(c.s_change, dtype=float))
        P = c.p_scale * Q

        def wrap(v):
            return ca.atan2(ca.sin(v), ca.cos(v))

        cost = 0
        opti.subject_to(X[:, 0] == p_x0)
        for k in range(n):
            x_k = X[:, k]
            u_k = U[:, k]
            # forward-Euler bicycle model, same one as mpc_pt.models
            opti.subject_to(X[0, k + 1] == x_k[0] + dt * x_k[3] * ca.cos(x_k[2]))
            opti.subject_to(X[1, k + 1] == x_k[1] + dt * x_k[3] * ca.sin(x_k[2]))
            opti.subject_to(X[2, k + 1] == x_k[2] + dt * (x_k[3] / L) * ca.tan(u_k[1]))
            opti.subject_to(X[3, k + 1] == x_k[3] + dt * u_k[0])

            e = X[:, k] - p_ref[:, k]
            e = ca.vertcat(e[0], e[1], wrap(e[2]), e[3])
            cost += ca.mtimes([e.T, Q, e])
            du = u_k - p_uref[:, k]
            cost += ca.mtimes([du.T, R, du])
            prev = p_uprev if k == 0 else U[:, k - 1]
            ddu = u_k - prev
            cost += ca.mtimes([ddu.T, S, ddu])

        eN = X[:, n] - p_ref[:, n]
        eN = ca.vertcat(eN[0], eN[1], wrap(eN[2]), eN[3])
        cost += ca.mtimes([eN.T, P, eN])
        opti.minimize(cost)

        # The tan(delta) model domain is mandatory. Physical limits are optional.
        opti.subject_to(opti.bounded(-STEERING_DOMAIN_MAX, U[1, :], STEERING_DOMAIN_MAX))
        if c.constraints:
            opti.subject_to(opti.bounded(c.v_min, X[3, 1:], c.v_max))
            opti.subject_to(opti.bounded(c.a_min, U[0, :], c.a_max))
            opti.subject_to(opti.bounded(-c.delta_max, U[1, :], c.delta_max))
            # previous steering angle for every horizon step: u_prev at k = 0,
            # then the previous decision variable
            prev_delta = ca.horzcat(p_uprev[1], U[1, :-1])
            opti.subject_to(
                opti.bounded(-c.ddelta_max, U[1, :] - prev_delta, c.ddelta_max)
            )

        opti.solver(
            "ipopt",
            {
                "print_time": False,
                "ipopt": {
                    "print_level": 0,
                    "max_iter": c.max_iter,
                    "tol": c.tol,
                    "acceptable_tol": 1e-6,
                    "constr_viol_tol": 1e-7,
                    "bound_relax_factor": 0.0,
                    "warm_start_init_point": "yes",
                },
            },
        )

        self._opti = opti
        self._X, self._U = X, U
        self._p_x0, self._p_ref, self._p_uref, self._p_uprev = p_x0, p_ref, p_uref, p_uprev

    @property
    def horizon(self) -> int:
        return self.cfg.horizon

    def solve(self, x0, ref_states, ref_controls, u_prev=None) -> MpcResult:
        n = self.cfg.horizon
        ref_states = np.asarray(ref_states, dtype=float)
        ref_controls = np.asarray(ref_controls, dtype=float)
        u_prev = np.zeros(CONTROL_DIM) if u_prev is None else np.asarray(u_prev, dtype=float)

        x0 = np.asarray(x0, dtype=float)
        for name, value, shape in (("x0", x0, (STATE_DIM,)),
                                   ("ref_states", ref_states, (n + 1, STATE_DIM)),
                                   ("ref_controls", ref_controls, (n, CONTROL_DIM)),
                                   ("u_prev", u_prev, (CONTROL_DIM,))):
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}")
        self._opti.set_value(self._p_x0, x0)
        self._opti.set_value(self._p_ref, ref_states.T)
        self._opti.set_value(self._p_uref, ref_controls.T)
        self._opti.set_value(self._p_uprev, u_prev)

        tic = time.perf_counter()
        try:
            sol = self._opti.solve()
        except RuntimeError:
            status = str(self._opti.stats().get("return_status", "solver_error"))
            return _failed(n, STATE_DIM, CONTROL_DIM, time.perf_counter() - tic, status)

        X = np.asarray(sol.value(self._X)).reshape(STATE_DIM, n + 1).T
        U = np.asarray(sol.value(self._U)).reshape(CONTROL_DIM, n).T
        objective = float(sol.value(self._opti.f))
        g = np.asarray(sol.value(self._opti.g)).ravel()
        lower = np.asarray(sol.value(self._opti.lbg)).ravel()
        upper = np.asarray(sol.value(self._opti.ubg)).ravel()
        violation = float(max(0.0, np.max(lower - g), np.max(g - upper)))
        elapsed = time.perf_counter() - tic
        if (not np.all(np.isfinite(X)) or not np.all(np.isfinite(U))
                or not np.all(np.isfinite(g)) or not np.isfinite(objective)
                or not np.isfinite(violation) or violation > 1e-6):
            return _failed(n, STATE_DIM, CONTROL_DIM, elapsed, "invalid_solution")

        solver_status = str(sol.stats()["return_status"])
        self._opti.set_initial(self._X, X.T)
        self._opti.set_initial(self._U, U.T)
        return MpcResult(U[0].copy(), X, U, elapsed, "optimal", objective,
                         violation, solver_status)
