"""Geometry regressions independent of the original short-arc scenarios."""

import numpy as np
import pytest

from mpc_pt.baselines import PurePursuitConfig, pure_pursuit_control, simulate_pure_pursuit
from mpc_pt.paths import ArcPath


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("theta", [0.0, 2.7])
def test_arc_projection_round_trip_beyond_half_circle(direction, theta):
    path = ArcPath(start=(3, -2), theta=theta, radius=8, length=40, direction=direction)
    for s in (0, 1, 24, 26, 30, 40):
        x, y, heading = path.at(s)
        normal = np.array([-np.sin(heading), np.cos(heading)])
        for offset in (-0.3, 0.0, 0.3):
            actual_s, lateral = path.project(np.array([x, y]) + offset * normal)
            assert actual_s == pytest.approx(s, abs=1e-9)
            assert lateral == pytest.approx(offset, abs=1e-9)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_arc_projection_chooses_nearest_endpoint(direction):
    path = ArcPath(radius=8, length=40, direction=direction)
    for angle, expected in ((-0.1, 0), (5.1, 40)):
        phi = path.phi0 + direction * angle
        point = path.center + 8 * np.array([np.cos(phi), np.sin(phi)])
        assert path.project(point)[0] == pytest.approx(expected, abs=1e-9)


def test_arc_rejects_ambiguous_full_circle():
    with pytest.raises(ValueError, match="full circle"):
        ArcPath(radius=8, length=2 * np.pi * 8)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("s", [0.0, 26.0, 39.0])
def test_pure_pursuit_preserves_circle_curvature(direction, s):
    path = ArcPath(radius=8, length=40, direction=direction)
    x, y, theta = path.at(s)
    expected = np.arctan(direction * 2.5 / 8)
    control = pure_pursuit_control(
        [x, y, theta, 2], path, PurePursuitConfig(), delta_prev=expected
    )
    # Circle geometry fixes the steering independently of target arc advance,
    # including a target clamped to the endpoint only one metre ahead.
    assert control[1] == pytest.approx(expected, abs=1e-10)


def test_pure_pursuit_coincident_target_is_finite():
    path = ArcPath(radius=8, length=40)
    x, y, theta = path.at(40)
    control = pure_pursuit_control([x, y, theta, 2], path, PurePursuitConfig())
    np.testing.assert_allclose(control, [0, 0], atol=1e-12)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_pure_pursuit_tracks_beyond_half_circle(direction):
    path = ArcPath(radius=8, length=40, direction=direction)
    result = simulate_pure_pursuit(path, [0, -0.5 * direction, 0, 2], 180, 0.1)
    projected = np.array([path.project(state[:2]) for state in result.states])
    assert projected[-1, 0] > 35
    assert np.all(np.diff(projected[:, 0]) >= -1e-9)
    assert np.max(np.abs(projected[140:, 1])) < 0.1
