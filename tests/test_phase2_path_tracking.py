"""Phase 2 tests: path geometry, reference construction and closed-loop tracking.

The MPC here is unconstrained and uses the time-indexed tracking cost from the
project model document, so the assertions are about convergence to the path,
not about limits.
"""

from __future__ import annotations

import numpy as np
import pytest

from mpc_pt.models import bicycle_step, BicycleParams
from mpc_pt.mpc import NMPCConfig, NonlinearMPC
from mpc_pt.paths import ArcPath, StraightPath, reference_from_path
from mpc_pt.sim import simulate_closed_loop

DT = 0.1
SPEED = 2.0
PARAMS = BicycleParams(wheelbase=2.5)


@pytest.fixture(scope="module")
def controller() -> NonlinearMPC:
    return NonlinearMPC(NMPCConfig(horizon=20, dt=DT, constraints=False))


def test_straight_path_geometry() -> None:
    path = StraightPath(start=(1.0, 2.0), theta=0.3, length=10.0)
    x, y, theta = path.at(np.array([0.0, 5.0, 10.0]))
    np.testing.assert_allclose(x, [1.0, 1.0 + 5 * np.cos(0.3), 1.0 + 10 * np.cos(0.3)])
    np.testing.assert_allclose(y, [2.0, 2.0 + 5 * np.sin(0.3), 2.0 + 10 * np.sin(0.3)])
    np.testing.assert_allclose(theta, [0.3, 0.3, 0.3])

    # a point 0.4 m to the left of the path has lateral error +0.4
    base = np.array([1.0 + 5.0 * np.cos(0.3), 2.0 + 5.0 * np.sin(0.3)])
    left = base + 0.4 * np.array([-np.sin(0.3), np.cos(0.3)])
    s, lateral = path.project(left)
    assert s == pytest.approx(5.0, abs=1e-9)
    assert lateral == pytest.approx(0.4, abs=1e-9)


def test_arc_path_geometry() -> None:
    radius, length = 8.0, 12.0
    path = ArcPath(start=(0.0, 0.0), theta=0.0, radius=radius, length=length, direction=+1.0)
    # a left arc starting at the origin and heading +x has its centre at (0, R)
    np.testing.assert_allclose(path.center, [0.0, radius], atol=1e-12)

    s = np.linspace(0.0, length, 25)
    x, y, theta = path.at(s)
    np.testing.assert_allclose(np.hypot(x - path.center[0], y - path.center[1]), radius, atol=1e-12)
    np.testing.assert_allclose(theta, s / radius, atol=1e-12)

    # projection round-trip: sample the path, push a point 0.3 m inwards
    for s_i in (1.0, 4.0, 9.0):
        px, py, _ = path.at(s_i)
        radial = np.array([px - path.center[0], py - path.center[1]])
        radial /= np.linalg.norm(radial)
        probe = np.array([px, py]) - 0.3 * radial   # 0.3 m towards the centre = left
        s_out, lateral = path.project(probe)
        assert s_out == pytest.approx(s_i, abs=1e-9)
        assert lateral == pytest.approx(0.3, abs=1e-9)


def test_reference_feedforward_steering() -> None:
    straight = reference_from_path(StraightPath(length=10.0), 2.0, DT, 10)
    np.testing.assert_allclose(straight.controls[:, 1], 0.0, atol=1e-12)

    radius = 8.0
    arc_path = ArcPath(radius=radius, length=10.0)
    arc = reference_from_path(arc_path, 2.0, DT, 10)
    np.testing.assert_allclose(arc.controls[:, 1], np.arctan(2.5 / radius), atol=1e-12)
    np.testing.assert_allclose(arc.states[:, 3], 2.0, atol=1e-12)


def _run(path, x0, n_steps, controller):
    reference = reference_from_path(
        path, speed=SPEED, dt=DT, n_steps=n_steps + controller.horizon
    )
    result = simulate_closed_loop(
        controller, np.asarray(x0, float), reference, n_steps, DT,
        plant_step=lambda x, u: bicycle_step(x, u, DT, PARAMS),
    )
    lateral = np.array([path.project(state[:2])[1] for state in result.states])
    return result, lateral


def test_straight_tracking_converges(controller) -> None:
    path = StraightPath(length=60.0)
    result, lateral = _run(path, [0.0, 0.5, 0.0, SPEED], 200, controller)

    assert set(result.statuses) == {"optimal"}
    assert abs(lateral[-1]) < 0.01
    assert np.abs(lateral[100:]).max() < 0.005      # settled after ~10 s
    assert np.abs(lateral[0]) > 0.4                 # the test would be vacuous otherwise


def test_arc_tracking_converges(controller) -> None:
    path = ArcPath(radius=8.0, length=40.0, direction=+1.0)
    result, lateral = _run(path, [0.0, -0.5, 0.0, SPEED], 140, controller)

    assert set(result.statuses) == {"optimal"}
    assert abs(lateral[-1]) < 0.05
    assert np.abs(lateral[70:]).max() < 0.05
    assert np.abs(lateral[0]) > 0.4
