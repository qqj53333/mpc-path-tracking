"""Failure, domain and shape checks outside the original nominal experiments."""

from dataclasses import replace

import numpy as np
import pytest

from mpc_pt.models import STEERING_DOMAIN_MAX, bicycle_step, di_step
from mpc_pt.mpc import LinearMPC, LinearMPCConfig, NMPCConfig, NonlinearMPC
from mpc_pt.paths import ArcPath, Reference, StraightPath, reference_from_path
from mpc_pt.sim import SimulationError, simulate_closed_loop


def test_failure_stops_before_previous_acceleration_is_repeated():
    real = LinearMPC()

    class FailAfterOne:
        horizon, control_dim = 20, 1
        last = None

        def solve(self, *args):
            if self.last is None:
                self.last = real.solve(*args)
                return self.last
            return replace(self.last, status="injected_failure")

    applied = []

    def plant(x, u):
        applied.append(u.copy())
        return di_step(x, u[0], 0.1)

    ref = Reference(np.tile([100, 0], (30, 1)), np.zeros((29, 1)), 0.1)
    with pytest.raises(SimulationError, match="injected_failure") as exc:
        simulate_closed_loop(FailAfterOne(), [0, 0.95], ref, 5, 0.1, plant)
    assert exc.value.step == 1
    assert exc.value.state[1] == pytest.approx(1.0, abs=1e-6)
    assert len(applied) == 1  # the failed step never reaches the plant


def test_actual_iteration_limit_is_not_a_valid_simulation():
    c = NonlinearMPC(NMPCConfig(max_iter=1))
    ref = reference_from_path(ArcPath(length=40), 2, 0.1, 25)
    with pytest.raises(SimulationError, match="Maximum_Iterations_Exceeded") as exc:
        simulate_closed_loop(c, [0, -0.5, 0, 2], ref, 5, 0.1,
                             lambda x, u: bicycle_step(x, u, 0.1))
    assert exc.value.step == 0


@pytest.mark.parametrize("bad_field", ["u0", "objective", "constraint_violation"])
def test_success_status_does_not_override_invalid_numeric_result(bad_field):
    real = LinearMPC()

    class InvalidOutput:
        horizon, control_dim = 20, 1

        def solve(self, *args):
            result = real.solve(*args)
            value = np.array([np.nan]) if bad_field == "u0" else float("nan")
            return replace(result, **{bad_field: value})

    ref = Reference(np.tile([5, 0], (22, 1)), np.zeros((21, 1)), 0.1)
    with pytest.raises(SimulationError, match="invalid or infeasible"):
        simulate_closed_loop(InvalidOutput(), [0, 0], ref, 1, 0.1,
                             lambda *_: pytest.fail("invalid control reached plant"))


@pytest.mark.parametrize("constrained", [False, True])
def test_one_step_nmpc_preserves_shapes_in_closed_loop(constrained):
    c = NonlinearMPC(NMPCConfig(horizon=1, constraints=constrained))
    ref = reference_from_path(ArcPath(length=40), 2, 0.1, 3)
    result = c.solve([0, -0.5, 0, 2], ref.states[:2], ref.controls[:1])
    assert result.status == "optimal"
    assert result.controls.shape == (1, 2)
    assert result.u0.shape == (2,)
    run = simulate_closed_loop(c, [0, -0.5, 0, 2], ref, 2, 0.1,
                               lambda x, u: bicycle_step(x, u, 0.1))
    assert run.controls.shape == (2, 2)


@pytest.mark.parametrize("offset", [-2.0, 2.0])
def test_unconstrained_model_stays_on_principal_steering_branch(offset):
    c = NonlinearMPC(NMPCConfig(constraints=False))
    ref = reference_from_path(StraightPath(), 2, 0.1, 25)
    r = c.solve([0, offset, 0, 2], ref.states[:21], ref.controls[:20])
    assert r.status == "optimal"
    assert np.max(np.abs(r.controls[:, 1])) <= STEERING_DOMAIN_MAX + 1e-6
    assert np.all(np.isfinite(bicycle_step([0, offset, 0, 2], r.u0, 0.1)))
    simulate_closed_loop(c, [0, offset, 0, 2], ref, 5, 0.1,
                        lambda x, u: bicycle_step(x, u, 0.1))


@pytest.mark.parametrize("config", [NMPCConfig, LinearMPCConfig])
@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_invalid_horizons_are_rejected(config, horizon):
    with pytest.raises(ValueError, match="positive integer"):
        config(horizon=horizon)


def test_scalar_straight_path_matches_array_interface():
    path = StraightPath(start=(1, -2), theta=0.3)
    scalar = path.at(3.0)
    array = path.at(np.array([3.0]))
    np.testing.assert_allclose(scalar, [a[0] for a in array])


def test_reference_refuses_inconsistent_endpoint_motion():
    with pytest.raises(ValueError, match="exceeds the finite path"):
        reference_from_path(StraightPath(length=1), 2, 0.1, 20)


def test_linear_infeasible_initial_state_is_reported():
    r = LinearMPC().solve([0, 2], np.tile([5, 0], (21, 1)))
    assert r.status != "optimal"
    assert r.controls.shape == (20, 1)
    assert np.isnan(r.u0).all()


@pytest.mark.parametrize("script", ["03_constraints.py", "04_horizon.py", "05_mpc_vs_baseline.py"])
def test_comparison_scripts_do_not_publish_failed_solver_runs(script, tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    directory = Path(__file__).resolve().parents[1] / "experiments"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("comparison_under_test", directory / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def low_iteration_controller(config):
        return NonlinearMPC(replace(config, max_iter=1))

    monkeypatch.setattr(module, "NonlinearMPC", low_iteration_controller)
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    with pytest.raises(SimulationError, match="Maximum_Iterations_Exceeded"):
        module.main()
    assert not list(tmp_path.iterdir())
