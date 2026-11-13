"""Phase 1 tests: the double-integrator MPC must be feasible and stay inside
the box constraints.

The solver tolerance is 1e-9 (OSQP eps_abs/eps_rel) with polishing; the
assertions below allow 1e-6, which is looser than the observed violation
(~1e-11) but far tighter than any physically meaningful bound.
"""

from __future__ import annotations

import numpy as np

from mpc_pt.models import di_step
from mpc_pt.mpc import LinearMPC, LinearMPCConfig
from mpc_pt.paths import Reference
from mpc_pt.sim import simulate_closed_loop

DT = 0.1
P_REF = 5.0
TOL = 1e-6


def _controller(v_max=1.0, a_max=0.5, horizon=20):
    return LinearMPC(LinearMPCConfig(horizon=horizon, dt=DT, v_max=v_max, a_max=a_max))


def test_single_solve_is_optimal_and_within_bounds() -> None:
    controller = _controller()
    ref = np.tile([P_REF, 0.0], (controller.horizon + 1, 1))
    result = controller.solve(np.array([0.0, 0.0]), ref)

    assert result.status == "optimal"
    assert result.states.shape == (controller.horizon + 1, 2)
    assert result.controls.shape == (controller.horizon, 1)
    assert np.all(np.abs(result.states[:, 1]) <= 1.0 + TOL)
    assert np.all(np.abs(result.controls[:, 0]) <= 0.5 + TOL)
    # the first predicted state must be the state we started from
    np.testing.assert_allclose(result.states[0], [0.0, 0.0], atol=1e-9)


def test_closed_loop_reaches_target_and_stops() -> None:
    controller = _controller()
    reference = Reference(
        states=np.tile([P_REF, 0.0], (151, 1)), controls=np.zeros((150, 1)), dt=DT
    )
    result = simulate_closed_loop(
        controller, np.array([0.0, 0.0]), reference, 150, DT,
        plant_step=lambda x, u: di_step(x, u[0], DT),
    )

    assert set(result.statuses) == {"optimal"}
    assert abs(result.states[-1, 0] - P_REF) < 1e-3
    assert abs(result.states[-1, 1]) < 1e-4
    # The approach is *not* monotone.  With Q = diag(1, 0.1) the velocity
    # error is cheap, so the optimiser prefers to keep the speed up and
    # overshoot to p = 5.1155 m before coming back; raising q_velocity to 0.5
    # removes the overshoot (peak 5.0003 m, see docs/results.md).  This bound
    # only guards against a regression that makes the overshoot much worse.
    assert result.states[:, 0].max() < P_REF + 0.15


def test_closed_loop_respects_constraints() -> None:
    controller = _controller(v_max=0.8, a_max=0.4)
    reference = Reference(
        states=np.tile([P_REF, 0.0], (151, 1)), controls=np.zeros((150, 1)), dt=DT
    )
    result = simulate_closed_loop(
        controller, np.array([0.0, 0.0]), reference, 150, DT,
        plant_step=lambda x, u: di_step(x, u[0], DT),
    )

    assert np.abs(result.states[:, 1]).max() <= 0.8 + TOL
    assert np.abs(result.controls[:, 0]).max() <= 0.4 + TOL
    # the limits are actually reached, otherwise the test would be vacuous
    assert np.abs(result.controls[:, 0]).max() > 0.4 - 1e-3
    assert np.abs(result.states[:, 1]).max() > 0.8 - 1e-3
