"""Predeclared scenarios and a common plant/measurement loop for both controllers."""
from dataclasses import dataclass
import time

import numpy as np

from mpc_pt.baselines import PurePursuitConfig, pure_pursuit_control
from mpc_pt.evaluation import constraint_violations, tracking_metrics
from mpc_pt.models import BicycleParams, bicycle_step
from mpc_pt.mpc import NMPCConfig, NonlinearMPC
from mpc_pt.paths import ArcPath, StraightPath, reference_from_path


@dataclass(frozen=True)
class Scenario:
    name: str
    radius: float | None
    direction: int
    speed: float
    offset: float
    heading: float = 0.0
    duration: float = 8.0

    def path(self):
        # All scenarios include room for the largest tested 4 s prediction horizon.
        length = self.speed * (self.duration + 4.0) + 2.0
        if self.radius is None:
            return StraightPath(length=length)
        return ArcPath(radius=self.radius, length=length, direction=self.direction)


TRAIN = (
    Scenario("train_straight", None, 1, 2.0, 0.5),
    Scenario("train_left_R8", 8.0, 1, 2.0, -0.5),
    Scenario("train_right_R12", 12.0, -1, 2.5, 0.5),
)
TEST = (
    Scenario("test_left_R6_slow", 6.0, 1, 1.5, -0.8, 0.15),
    Scenario("test_left_R12_fast", 12.0, 1, 3.0, 0.8, -0.15),
    Scenario("test_right_R10", 10.0, -1, 2.5, 0.8, 0.2),
    Scenario("test_straight_fast", None, 1, 3.0, -1.0, 0.2),
)
MPC_CANDIDATES = (10, 20, 30, 40)
PP_CANDIDATES = (1.5, 2.0, 3.0, 4.0)
NOISE_STD = np.array([0.02, 0.02, 0.005, 0.02])  # m, m, rad, m/s


def run_case(scenario, controller_kind, parameter, *, dt=0.1, plant_wheelbase=2.5,
             noisy=False, seed=7, delay_steps=0, scale_cost_with_dt=False):
    """No retries or retuning; a failed step ends this run and remains in the report."""
    path = scenario.path()
    steps = round(scenario.duration / dt)
    horizon = int(parameter) if controller_kind == "mpc" else 1
    reference = reference_from_path(path, scenario.speed, dt, steps + horizon)
    config = NMPCConfig(horizon=horizon, dt=dt, ddelta_max=0.5 * dt, s_change=(0.1, 0.5))
    if scale_cost_with_dt:
        ratio = dt / 0.1
        config.q = tuple(np.asarray(config.q) * ratio)
        config.r = tuple(np.asarray(config.r) * ratio)
        config.s_change = tuple(np.asarray(config.s_change) / ratio)
        config.p_scale /= ratio  # fixed terminal cost, scaled stage integrals
    controller = NonlinearMPC(config) if controller_kind == "mpc" else None
    pp = PurePursuitConfig(lookahead=float(parameter), speed=scenario.speed,
                          ddelta_max=0.5 * dt)
    plant = BicycleParams(wheelbase=plant_wheelbase)
    state = np.array([0.0, scenario.offset, scenario.heading, scenario.speed])
    states, commands, applied, times = [state.copy()], [], [], []
    noise = np.random.default_rng(seed).normal(size=(steps, 4)) * NOISE_STD if noisy else np.zeros((steps, 4))
    row = dict(scenario=scenario.name, controller=controller_kind, parameter=parameter,
               dt=dt, plant_wheelbase=plant_wheelbase, noisy=noisy,
               seed=seed if noisy else None, delay_steps=delay_steps,
               scale_cost_with_dt=scale_cost_with_dt,
               planned_steps=steps, accepted_decisions=0, solver_failures=0,
               valid=False, reason=None, metrics=None, max_prediction_violation=0.0)
    previous = np.zeros(2)
    for k in range(steps):
        measured = state + noise[k]
        tic = time.perf_counter()
        if controller is not None:
            window = reference.window(k, horizon)
            solution = controller.solve(measured, window.states, window.controls, previous)
            if solution.status != "optimal":
                row.update(reason=solution.status, solver_failures=1, failure_step=k)
                return row
            command = solution.u0
            row["max_prediction_violation"] = max(row["max_prediction_violation"], solution.constraint_violation)
        else:
            command = pure_pursuit_control(measured, path, pp, previous[1])
        times.append(time.perf_counter() - tic)
        if not np.all(np.isfinite(command)):
            row.update(reason="non-finite control", failure_step=k)
            return row
        row["accepted_decisions"] += 1
        commands.append(command.copy())
        previous = command.copy()
        actual = commands[k - delay_steps] if k >= delay_steps else np.zeros(2)
        state = bicycle_step(state, actual, dt, plant)
        states.append(state.copy())
        applied.append(actual.copy())
    states, commands, applied = map(np.asarray, (states, commands, applied))
    violations = constraint_violations(states, commands, dt)
    actual_violations = constraint_violations(states, applied, dt)
    row["command_violations"] = violations
    row["applied_violations"] = actual_violations
    row["metrics"] = tracking_metrics(path, states, applied, dt, times)
    row["valid"] = max(*violations.values(), *actual_violations.values()) <= 1e-6
    row["reason"] = "ok" if row["valid"] else "constraint_violation"
    return row


def select_parameter(rows, kind):
    """Minimum mean training RMS, with any failed scenario disqualifying a candidate."""
    candidates = MPC_CANDIDATES if kind == "mpc" else PP_CANDIDATES
    scores = []
    for parameter in candidates:
        subset = [r for r in rows if r["controller"] == kind and r["parameter"] == parameter]
        valid = len(subset) == len(TRAIN) and all(r["valid"] for r in subset)
        score = float(np.mean([r["metrics"]["lateral_rms_m"] for r in subset])) if valid else None
        scores.append(dict(parameter=parameter, eligible=valid, training_mean_rms_m=score))
    eligible = [r for r in scores if r["eligible"]]
    if not eligible:
        raise RuntimeError(f"no valid training candidate for {kind}")
    return min(eligible, key=lambda r: (r["training_mean_rms_m"], r["parameter"]))["parameter"], scores
