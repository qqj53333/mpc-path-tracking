"""Phase 3 tests: limits are respected, and they are not vacuous.

The constrained MPC must stay inside the box and rate limits; the
unconstrained one must visibly exceed them on the same scenario, otherwise the
comparison experiment would have nothing to show.
"""

from __future__ import annotations

import numpy as np
import pytest

from mpc_pt.baselines import PurePursuitConfig, simulate_pure_pursuit
from mpc_pt.models import BicycleParams, bicycle_step
from mpc_pt.mpc import NMPCConfig, NonlinearMPC
from mpc_pt.paths import ArcPath, reference_from_path
from mpc_pt.sim import simulate_closed_loop

DT = 0.1
SPEED = 2.0
N_STEPS = 120
X0 = np.array([0.0, -0.5, 0.0, SPEED])
PARAMS = BicycleParams(wheelbase=2.5)

DELTA_MAX = 0.5236
DDELTA_MAX = 0.05
V_MAX = 8.0
A_MAX = 2.0
TOL = 1e-6

PATH = ArcPath(radius=8.0, length=40.0, direction=+1.0)


def _run(config: NMPCConfig):
    controller = NonlinearMPC(config)
    reference = reference_from_path(
        PATH, speed=SPEED, dt=DT, n_steps=N_STEPS + controller.horizon
    )
    result = simulate_closed_loop(
        controller, X0, reference, N_STEPS, DT,
        plant_step=lambda x, u: bicycle_step(x, u, DT, PARAMS),
    )
    return result


@pytest.fixture(scope="module")
def constrained_run():
    return _run(NMPCConfig(horizon=20, dt=DT, constraints=True, s_change=(0.1, 0.5)))


@pytest.fixture(scope="module")
def unconstrained_run():
    return _run(NMPCConfig(horizon=20, dt=DT, constraints=False))


def test_constrained_mpc_respects_every_limit(constrained_run) -> None:
    controls = constrained_run.controls
    states = constrained_run.states

    assert set(constrained_run.statuses) == {"optimal"}
    assert np.all(states[:, 3] >= -TOL)
    assert np.all(states[:, 3] <= V_MAX + TOL)
    assert np.all(np.abs(controls[:, 0]) <= A_MAX + TOL)
    assert np.all(np.abs(controls[:, 1]) <= DELTA_MAX + TOL)
    # the rate limit only exists for k >= 1: the first step is measured
    # against the initial (zero) steering angle
    delta_prev = np.concatenate([[0.0], controls[:-1, 1]])
    assert np.all(np.abs(controls[:, 1] - delta_prev) <= DDELTA_MAX + TOL)


def test_unconstrained_mpc_exceeds_the_steering_limit(unconstrained_run) -> None:
    assert np.abs(unconstrained_run.controls[:, 1]).max() > DELTA_MAX + 0.1


def test_limits_are_active_not_vacuous(constrained_run) -> None:
    delta = constrained_run.controls[:, 1]
    # the command saturates at the steering bound for part of the transient
    assert np.max(np.abs(delta)) > DELTA_MAX - 1e-4


def test_pure_pursuit_respects_the_same_limits() -> None:
    result = simulate_pure_pursuit(
        PATH, X0, N_STEPS, DT, PurePursuitConfig(lookahead=4.0, speed=SPEED)
    )
    delta = result.controls[:, 1]
    delta_prev = np.concatenate([[0.0], delta[:-1]])
    assert np.all(np.abs(delta) <= DELTA_MAX + TOL)
    assert np.all(np.abs(delta - delta_prev) <= DDELTA_MAX + TOL)


def test_longer_horizon_tracks_better_on_this_arc() -> None:
    short = _run(NMPCConfig(horizon=5, dt=DT, constraints=True, s_change=(0.1, 0.5)))
    long = _run(NMPCConfig(horizon=20, dt=DT, constraints=True, s_change=(0.1, 0.5)))

    def final_lateral(run):
        return abs(PATH.project(run.states[-1, :2])[1])

    assert final_lateral(long) < final_lateral(short)
