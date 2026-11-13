"""Phase 0 sanity tests for the kinematic bicycle model.

These tests do not check optimality (there is no optimiser yet); they check that
one step of the discrete model does what the geometry says it should.

Tolerances are motivated in the docstrings.  The circle test deliberately does
*not* compare the Euler polygon with the analytic circle point by point: forward
Euler traces a regular polygon whose vertices lie on a circle of radius
`R * (dtheta/2) / sin(dtheta/2)` around a centre shifted by `v*T/2` from the
analytic one.  The test therefore checks what actually holds: constant distance
to the fitted centre, and that the fitted radius is close to `L / tan(delta)`.
"""

from __future__ import annotations

import numpy as np
import pytest

from mpc_pt.models import BicycleParams, bicycle_step
from mpc_pt.sim import simulate_open_loop

L = 2.5
PARAMS = BicycleParams(wheelbase=L)


def fit_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic (Kasa) least-squares circle fit; exact for noise-free points."""
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(x))])
    b = x**2 + y**2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([cx, cy]), float(np.sqrt(c + cx**2 + cy**2))


def test_one_step_matches_handwritten_euler() -> None:
    """X(k+1) = X(k) + T * [v cos theta, v sin theta, (v/L) tan delta, a]."""
    state = np.array([1.0, 2.0, 0.3, 4.0])
    control = np.array([0.7, -0.2])
    T = 0.05

    expected = np.array(
        [
            1.0 + T * 4.0 * np.cos(0.3),
            2.0 + T * 4.0 * np.sin(0.3),
            0.3 + T * (4.0 / L) * np.tan(-0.2),
            4.0 + T * 0.7,
        ]
    )
    np.testing.assert_allclose(bicycle_step(state, control, T, PARAMS), expected, atol=1e-12)


def test_straight_line_when_steering_is_zero() -> None:
    """a = 0 and delta = 0: the car keeps its heading and moves straight.

    With delta = 0 the heading is constant, so every Euler step adds exactly
    `T * v` along the same direction.  The cross-track distance from the
    initial ray must therefore vanish to machine precision (tolerance 1e-12 m).
    """
    v = 2.0
    theta = 0.7
    T = 0.1
    controls = np.zeros((60, 2))
    result = simulate_open_loop(np.array([0.0, 0.0, theta, v]), controls, T, PARAMS)

    positions = result.states[:, :2]
    direction = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-direction[1], direction[0]])
    cross_track = positions @ normal

    n = np.arange(len(positions))
    expected = np.outer(n * T * v, direction)
    np.testing.assert_allclose(positions, expected, atol=1e-12)
    assert np.max(np.abs(cross_track)) < 1e-12


def test_constant_steering_traces_circle() -> None:
    """Constant v and delta, a = 0: the Euler polygon is a regular polygon.

    Checks
    ------
    1. every vertex is equidistant from the fitted centre (tolerance 1e-9 m;
       the algebraic fit is exact on these data, observed spread ~3e-13 m);
    2. the fitted radius is close to the geometric turning radius
       `R = L / tan(delta)`.  Forward Euler overshoots by
       `(dtheta/2)^2 / 6` in relative terms, i.e. ~`dtheta^2/24`, with
       `dtheta = v*T/R`; the tolerance `(v*T/R)^2 / 12` is twice that
       first-order estimate and therefore scales with T, as it must;
    3. the fitted centre is shifted by `v*T/2` from the analytic centre
       (1e-3 relative tolerance), which is the discretisation artefact the
       work order asks us to write down instead of pretending the two match.
    """
    v = 2.0
    delta = 0.3
    T = 0.1
    n_steps = 300  # 30 s > one full period 2*pi*R/v ~ 25.4 s
    theta0 = 0.0  # heading +x, the project convention

    controls = np.tile([0.0, delta], (n_steps, 1))
    result = simulate_open_loop(np.array([0.0, 0.0, theta0, v]), controls, T, PARAMS)
    points = result.states[:, :2]

    centre, radius = fit_circle(points)
    distances = np.linalg.norm(points - centre, axis=1)
    assert distances.max() - distances.min() < 1e-9

    R_analytic = L / np.tan(delta)
    rel_error = abs(radius - R_analytic) / R_analytic
    tolerance = (v * T / R_analytic) ** 2 / 12.0
    assert rel_error < tolerance, f"rel_error={rel_error:.2e} > {tolerance:.2e}"

    # The analytic centre lies R_analytic to the left of the start point for a
    # positive (left) steering angle; "left of +x" is +y.
    analytic_centre = R_analytic * np.array([-np.sin(theta0), np.cos(theta0)])
    offset = float(np.linalg.norm(centre - analytic_centre))
    assert abs(offset - 0.5 * v * T) < 1e-3 * 0.5 * v * T, (
        f"centre offset {offset:.6f} m, expected ~{0.5 * v * T:.6f} m"
    )


def test_positive_steering_turns_left() -> None:
    """delta > 0 must increase theta (counter-clockwise), the documented convention."""
    T = 0.1
    state0 = np.array([0.0, 0.0, 0.0, 1.0])
    left = simulate_open_loop(state0, np.tile([0.0, 0.3], (10, 1)), T, PARAMS)
    right = simulate_open_loop(state0, np.tile([0.0, -0.3], (10, 1)), T, PARAMS)
    assert left.states[-1, 2] > 0.0
    assert right.states[-1, 2] < 0.0
    # curvature sign check: a left turn curves towards +y when starting along +x
    assert left.states[-1, 1] > 0.0


def test_braking_turn_lowers_heading() -> None:
    """Phase 4 of the open-loop demo: a = -0.4, delta = -0.30 from v = 2 m/s.

    With delta < 0 and v > 0 the heading rate (v/L) tan(delta) is negative, so
    theta must decrease.  While a is constant the speed is affine in k, which
    makes the discrete heading change exact:
        dtheta = T * (1/L) * tan(delta) * sum(v_k).
    The continuous value is T-independent, (tan(delta)/L) * integral(v dt), and
    is slightly *smaller* in magnitude here because forward Euler uses the
    start-of-step (larger) speed throughout the braking interval.
    """
    v0, a, delta, T, n = 2.0, -0.4, -0.30, 0.1, 50
    state0 = np.array([0.0, 0.0, 1.0, v0])
    result = simulate_open_loop(state0, np.tile([a, delta], (n, 1)), T, PARAMS)

    d_theta = result.states[-1, 2] - state0[2]
    expected = T * (1.0 / L) * np.tan(delta) * result.states[:-1, 3].sum()
    continuous = (np.tan(delta) / L) * (v0 * n * T + 0.5 * a * (n * T) ** 2)

    assert d_theta < 0.0
    np.testing.assert_allclose(d_theta, expected, atol=1e-12)
    assert expected < continuous < 0.0  # Euler over-turns while braking
    assert abs(result.states[-1, 3]) < 1e-12  # v reaches exactly zero


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        bicycle_step(np.zeros(4), np.array([0.0, 0.0]), 0.0, PARAMS)
    with pytest.raises(ValueError):
        bicycle_step(np.zeros(4), np.array([0.0, np.pi / 2]), 0.1, PARAMS)
    with pytest.raises(ValueError):
        simulate_open_loop(np.zeros(4), np.zeros((5, 3)), 0.1, PARAMS)
