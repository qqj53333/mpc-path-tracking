"""Reference paths and the time-indexed reference trajectories they produce.

Two geometric paths are provided:

* :class:`StraightPath` - straight line along a heading.
* :class:`ArcPath`      - circular arc with constant curvature, left or right.

A path answers three questions:

* `at(s)`    - position and tangent angle at arc length `s`,
* `curvature(s)` - 1/R at that point (0 for the straight line),
* `project(p)`   - arc length of the closest point and the signed lateral
  error of a point `p` (positive to the *left* of the path direction).

:func:`reference_from_path` samples a path at a constant speed and returns the
reference states `[x, y, theta, v]` plus the feed-forward input
`u_ref = [a_ref, atan(L * kappa)]` used by the MPC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_WHEELBASE = 2.5


class StraightPath:
    """Straight reference path starting at `start` with heading `theta`."""

    def __init__(self, start=(0.0, 0.0), theta: float = 0.0, length: float = 60.0):
        self.start = np.asarray(start, dtype=float).reshape(2)
        self.theta = float(theta)
        self.length = float(length)
        if (not np.all(np.isfinite(self.start)) or not np.isfinite(self.theta)
                or not np.isfinite(self.length) or self.length < 0):
            raise ValueError("straight path coordinates must be finite and length nonnegative")

    def at(self, s):
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        direction = np.array([np.cos(self.theta), np.sin(self.theta)])
        x = self.start[0] + direction[0] * s
        y = self.start[1] + direction[1] * s
        return x, y, np.full_like(s, self.theta)

    def curvature(self, s):
        return np.zeros_like(np.asarray(s, dtype=float))

    def project(self, p):
        d = np.asarray(p, dtype=float).reshape(2) - self.start
        direction = np.array([np.cos(self.theta), np.sin(self.theta)])
        normal = np.array([-np.sin(self.theta), np.cos(self.theta)])  # left normal
        s = float(np.clip(d @ direction, 0.0, self.length))
        return s, float(d @ normal)


class ArcPath:
    """Circular arc of constant radius.

    `direction = +1` turns left (counter-clockwise), `-1` turns right.
    `start` is the point at `s = 0` and `theta` the tangent angle there.
    The arc must be shorter than a full circle: position alone cannot
    distinguish repeated visits to the same point on a multi-turn path.
    """

    def __init__(
        self,
        start=(0.0, 0.0),
        theta: float = 0.0,
        radius: float = 8.0,
        length: float = 25.0,
        direction: float = 1.0,
    ):
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive and finite")
        if not np.isfinite(length) or not 0.0 <= length < 2.0 * np.pi * radius:
            raise ValueError("length must be nonnegative and shorter than a full circle")
        if direction not in (-1.0, 1.0):
            raise ValueError("direction must be +1 (left) or -1 (right)")
        self.start = np.asarray(start, dtype=float).reshape(2)
        self.theta = float(theta)
        self.radius = float(radius)
        self.length = float(length)
        self.direction = float(direction)
        normal = np.array([-np.sin(self.theta), np.cos(self.theta)])
        self.center = self.start + self.direction * self.radius * normal
        self.phi0 = float(np.arctan2(self.start[1] - self.center[1],
                                     self.start[0] - self.center[0]))

    def _phi(self, s):
        return self.phi0 + self.direction * np.asarray(s, dtype=float) / self.radius

    def at(self, s):
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        phi = self._phi(s)
        x = self.center[0] + self.radius * np.cos(phi)
        y = self.center[1] + self.radius * np.sin(phi)
        theta = self.theta + self.direction * s / self.radius
        return x, y, theta

    def curvature(self, s):
        return np.full_like(np.asarray(s, dtype=float), self.direction / self.radius)

    def project(self, p):
        """Nearest point on the finite arc and error along its left normal."""
        p = np.asarray(p, dtype=float).reshape(2)
        phi_p = np.arctan2(p[1] - self.center[1], p[0] - self.center[0])
        angle = np.mod(self.direction * (phi_p - self.phi0), 2.0 * np.pi)
        s = float(self.radius * angle)
        if s > self.length:
            # The radial projection lies outside the finite arc. Choose the
            # nearer endpoint, including points immediately before the start.
            ex, ey, _ = self.at(self.length)
            end = np.array([ex, ey])
            s = 0.0 if np.linalg.norm(p - self.start) <= np.linalg.norm(p - end) else self.length
        px, py, heading = self.at(s)
        normal = np.array([-np.sin(heading), np.cos(heading)])
        lateral = (p - np.array([px, py])) @ normal
        return float(s), float(lateral)


@dataclass
class Reference:
    """Time-indexed reference: `states[k]` and `controls[k]` for step k."""

    states: np.ndarray   # (M+1, 4) = [x, y, theta, v]
    controls: np.ndarray  # (M, 2)   = [a_ref, delta_ref]
    dt: float

    def window(self, k: int, horizon: int) -> "Reference":
        """Rows k ... k+horizon, clamped to the available data."""
        last_state = len(self.states) - 1
        last_control = len(self.controls) - 1
        sidx = np.clip(np.arange(k, k + horizon + 1), 0, last_state)
        uidx = np.clip(np.arange(k, k + horizon), 0, last_control)
        return Reference(self.states[sidx], self.controls[uidx], self.dt)


def reference_from_path(
    path,
    speed: float,
    dt: float,
    n_steps: int,
    s_start: float = 0.0,
    wheelbase: float = DEFAULT_WHEELBASE,
) -> Reference:
    """Sample `path` at constant `speed` into a time-indexed reference.

    The feed-forward steering follows the path curvature,
    `delta_ref = atan(L * kappa)`, so that `R` penalises the deviation from the
    reference input rather than forcing the steering angle to zero.
    """
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    s = s_start + speed * dt * np.arange(n_steps + 1)
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(speed) or speed < 0:
        raise ValueError("reference dt must be positive and speed nonnegative")
    if not np.all(np.isfinite(s)) or s.min() < 0 or s.max() > path.length + 1e-9:
        raise ValueError("reference exceeds the finite path; endpoint stopping is not implemented")
    x, y, theta = path.at(s)
    states = np.column_stack([x, y, theta, np.full(n_steps + 1, speed)])
    delta_ref = np.arctan(wheelbase * path.curvature(s))
    controls = np.column_stack([np.zeros(n_steps), delta_ref[:n_steps]])
    return Reference(states=states, controls=controls, dt=dt)
