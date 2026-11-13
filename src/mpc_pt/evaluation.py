"""Shared, explicit validity and tracking metrics for reproducible experiments."""

import numpy as np


def constraint_violations(states, controls, dt, *, v_min=0.0, v_max=8.0,
                          a_min=-2.0, a_max=2.0, delta_max=0.5236,
                          delta_rate_max=0.5):
    """Positive excess in each quantity's own units; includes the first input."""
    states, controls = np.asarray(states), np.asarray(controls)
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
        raise ValueError("cannot score non-finite simulation data")
    rate = np.diff(np.r_[0.0, controls[:, 1]]) / dt
    return {
        "speed_mps": float(max(0, np.max(v_min - states[:, 3]), np.max(states[:, 3] - v_max))),
        "acceleration_mps2": float(max(0, np.max(a_min - controls[:, 0]),
                                       np.max(controls[:, 0] - a_max))),
        "steering_rad": float(max(0, np.max(np.abs(controls[:, 1])) - delta_max)),
        "steering_rate_radps": float(max(0, np.max(np.abs(rate)) - delta_rate_max)),
    }


def tracking_metrics(path, states, controls, dt, decision_times):
    """Timing is measured per decision, including Python, excluding plant/plots."""
    states, controls = np.asarray(states), np.asarray(controls)
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
        raise ValueError("cannot score non-finite simulation data")
    lateral = np.array([path.project(x[:2])[1] for x in states])
    steering_rate = np.diff(np.r_[0.0, controls[:, 1]]) / dt
    ms = 1000 * np.asarray(decision_times)
    return {
        "lateral_rms_m": float(np.sqrt(np.mean(lateral**2))),
        "lateral_final_m": float(abs(lateral[-1])),
        "lateral_max_m": float(np.max(np.abs(lateral))),
        "steering_rate_rms_radps": float(np.sqrt(np.mean(steering_rate**2))),
        "steering_rate_mean_abs_radps": float(np.mean(np.abs(steering_rate))),
        "decision_mean_ms": float(np.mean(ms)),
        "decision_p95_ms": float(np.percentile(ms, 95)),
        "decision_max_ms": float(np.max(ms)),
        "deadline_misses": int(np.sum(ms > dt * 1000)),
    }
