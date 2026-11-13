"""Checks for independent constraint metrics and held-out selection integrity."""
import sys
from pathlib import Path

import numpy as np
import pytest

from mpc_pt.evaluation import constraint_violations


def test_constraint_metric_includes_initial_steering_jump_and_speed():
    states = np.zeros((3, 4))
    states[:, 3] = [2, 2, 8.2]
    controls = np.array([[0, 0.1], [0, 0.1]])
    violations = constraint_violations(states, controls, 0.1)
    assert violations["speed_mps"] == pytest.approx(0.2)
    assert violations["steering_rate_radps"] == pytest.approx(0.5)


def test_nonfinite_data_cannot_be_scored():
    with pytest.raises(ValueError, match="non-finite"):
        constraint_violations(np.full((2, 4), np.nan), np.zeros((1, 2)), 0.1)


def test_failed_candidate_cannot_win_training_selection():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from _benchmark import MPC_CANDIDATES, TRAIN, TEST, select_parameter
    assert not {s.name for s in TRAIN} & {s.name for s in TEST}
    rows = []
    for parameter in MPC_CANDIDATES:
        for i, scenario in enumerate(TRAIN):
            rows.append(dict(controller="mpc", parameter=parameter,
                             valid=not (parameter == 10 and i == 0),
                             metrics={"lateral_rms_m": parameter / 100.0}))
    chosen, scores = select_parameter(rows, "mpc")
    assert chosen == 20
    assert not scores[0]["eligible"]
