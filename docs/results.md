# Results

The tables and figures below were regenerated from the eight experiment scripts
on 2026-11-14 after the solver-validity and boundary fixes. Timings describe this run, not a
performance guarantee. [Verification details](verification.md) and the captured
[experiment output](experiment-results.json) are included.

Reproduce everything from the repository root:

    python experiments/00_open_loop.py
    python experiments/01_double_integrator_mpc.py
    python experiments/02_path_tracking.py
    python experiments/03_constraints.py
    python experiments/04_horizon.py
    python experiments/05_mpc_vs_baseline.py
    python experiments/06_benchmark.py
    python experiments/07_sampling.py
    python -m pytest

## Phase 0 - open-loop kinematic bicycle model

Script: [`experiments/00_open_loop.py`](../experiments/00_open_loop.py) ·
Figure: [`outputs/00_open_loop.png`](../outputs/00_open_loop.png) ·
Data: [`outputs/00_open_loop.csv`](../outputs/00_open_loop.csv)

### Setup

| Symbol | Value | Meaning |
|---|---|---|
| `L` | 2.5 m | wheelbase |
| `T` | 0.1 s | sampling time |
| `X(0)` | `[0, 0, 0, 0]` | origin, heading +x, standing still |
| `N` | 200 steps | 20 s |

Four piecewise-constant input phases, 5 s each:

| Phase | `a` [m/s^2] | `delta` [rad] | what happens |
|---|---|---|---|
| 1 | +0.4 | 0 | accelerate straight from 0 to 2 m/s |
| 2 | 0 | 0 | keep 2 m/s, straight |
| 3 | 0 | +0.30 | left turn, radius `L/tan(delta) = 8.08 m` |
| 4 | -0.4 | -0.30 | brake while steering back |

### What the numbers show

- **The straight legs are exactly straight.** During phase 2 the `y`
  coordinate varies by 0.0 m and `theta` stays at 0.0 rad.
- **Held steering produces a constant turning rate.** Phase 3 changes the
  heading by +1.237345 rad; the closed form `(v/L) tan(delta) * 5 s` gives the
  same number to six decimals.
- **Braking with negative steering lowers the heading, as it should.** In
  phase 4 (`delta = -0.30`, `v` falling from 2 to 0) the heading drops by
  0.631046 rad, exactly the discrete sum `T (1/L) tan(delta) sum(v_k)` that
  the script prints. The continuous integral over the same interval is
  -0.618715 rad: forward Euler uses the start-of-step (larger) speed, so it
  over-turns by about 2 % while braking. The curve flattens as `v -> 0`
  because the turning rate is proportional to `v`.
- **Forward Euler is visible in the distance travelled.** With
  `a = +0.4 m/s^2` for 5 s the continuous solution covers 5.0 m; the simulation
  reaches 4.9 m because each step uses the speed at its start. In the braking
  phase the same effect runs the other way (5.1 m instead of 5.0 m), so the
  20 s total of 30.0 m matches only by coincidence of the symmetric input.
- **Final state** at 20 s: `x = 25.600 m`, `y = 9.361 m`, `theta = 0.606 rad`,
  `v = 0`.

### Coordinate convention check

`docs/model.md` chooses option A: the vehicle points along +x, a straight
reference has `theta_ref = 0`, and `delta > 0` steers left. The first version
of this experiment started at `theta0 = pi/2` (heading +y), which is a valid
initial condition but contradicts the project convention; it was changed to
`theta0 = 0`. The model code did not change - only the scenario did.

## Phase 1 - linear MPC for the double integrator

Script: [`experiments/01_double_integrator_mpc.py`](../experiments/01_double_integrator_mpc.py) ·
Figures: [`outputs/01_double_integrator_single.png`](../outputs/01_double_integrator_single.png),
[`outputs/01_double_integrator_closed.png`](../outputs/01_double_integrator_closed.png)

Double integrator `p' = v`, `v' = a`, exact ZOH discretisation, solved as a
QP with cvxpy/OSQP. `N = 20` (2 s), `|v| <= 1.0 m/s`, `|a| <= 0.5 m/s^2`,
target `p_ref = 5 m`, `Q = diag(1, 0.1)`, `R = 0.1`, `P = Q`.

| quantity | value |
|---|---|
| final position | 4.999999 m |
| final velocity | -3.3e-06 m/s |
| max abs(v) | 1.000000000 m/s (limit 1.0) |
| max abs(a) | 0.500000000 m/s^2 (limit 0.5) |
| steps with `a` at the limit | 38 of 150 |
| steps with `v` at the limit | 31 of 150 |
| solve time | mean 1.98 ms, max 35.02 ms, total 0.30 s |
| solver status | `optimal` on all 150 steps |

The first solve of the run is the slow one (~35 ms, cvxpy/OSQP setup); after
that the warm-started QP takes ~2 ms.

**The approach is not monotone.** With `q_velocity = 0.1` the car overshoots
to `p = 5.1155 m` at `t = 7.1 s` and then comes back to 5.0. The velocity
error is cheap compared with the position error, so the optimiser keeps the
speed up and accepts a small overshoot. Changing the velocity weight would be a separate tuning experiment.
The published scripts use the defaults recorded in `docs/model.md`.

## Phase 2 - bicycle MPC path tracking

Script: [`experiments/02_path_tracking.py`](../experiments/02_path_tracking.py) ·
Figure: [`outputs/02_path_tracking.png`](../outputs/02_path_tracking.png) ·
Data: `outputs/02_straight.csv`, `outputs/02_arc.csv`

`N = 20`, `T = 0.1 s`, `v_ref = 2 m/s`, weights from `docs/model.md`
(`Q = diag(4, 4, 32.8, 1)`, `R = diag(0.25, 3.64)`, `P = 3Q`), **no**
physical actuator/state limits and no input-change penalty. The mandatory
steering guard `abs(delta) <= pi/2 - 0.001` keeps the model on its principal
branch even in this phase. Direct nonlinear MPC (CasADi/IPOPT),
forward-Euler prediction, heading error wrapped with `atan2(sin, cos)`.

| path | max abs(e_lat) | final abs(e_lat) | rms e_lat | max abs(delta) | solve time (mean) |
|---|---|---|---|---|---|
| straight line, 0.5 m offset | 0.500 m | 0.0000 m | 0.1254 m | 0.423 rad | 2.9 ms |
| arc R = 8 m, 0.5 m offset | 0.502 m | 0.0200 m | 0.1586 m | 0.815 rad | 2.6 ms |

Both converge. The unconstrained controller asks for **0.815 rad (47 deg)** of
steering at the first step of the arc, exceeding this project's 30 deg limit. That
is the unconstrained optimum and it is exactly what Phase 3.1 quantifies.

## Phase 3 - constraints, horizon, baseline

### 3.1 Constraints vs no constraints

Script: [`experiments/03_constraints.py`](../experiments/03_constraints.py) ·
Figure: [`outputs/03_constraints.png`](../outputs/03_constraints.png)

Arc R = 8 m, 0.5 m offset, v_ref = 2 m/s, 12 s. `limits` means
`0 <= v <= 8`, `|a| <= 2`, `|delta| <= 0.524`, `|Delta delta| <= 0.05`;
`limits + S` adds `S = diag(0.1, 0.5)`.

| variant | max abs(delta) | mean abs(Delta delta) | max abs(e_lat) | rms e_lat | final abs(e_lat) | solve ms |
|---|---|---|---|---|---|---|
| no constraints | 0.8154 | 0.0052 | 0.5024 | 0.1586 | 0.0200 | 2.99 |
| limits | 0.5236 | 0.0070 | 0.6046 | 0.2536 | 0.0202 | 4.15 |
| limits + S | 0.5236 | 0.0070 | 0.6047 | 0.2538 | 0.0204 | 4.33 |

Two observations, both only for this scenario:

1. **The limits make the transient worse, not better.** The unconstrained run
   is feasible-looking but asks for 47 deg of steering; once the optimisation enforces
   the 30 deg bound and 0.05 rad/step change limit, the vehicle drifts to
   0.605 m before recovering. It still ends on the path (0.020 m). This is the
   usual trade: feasibility is bought with transient tracking error.
2. **`S` does almost nothing here, and that is a result too.** The steering
   rate limit (0.05 rad per step) is active and the measured changes are small: rms lateral 0.2536 -> 0.2538 m, mean `|Delta delta|`
   0.0070 -> 0.0070. `S` is implemented and switchable
   (`NMPCConfig.s_change`) but in these settings the constraint, not the
   penalty, is what smooths the steering.

### 3.2 Prediction horizon

Script: [`experiments/04_horizon.py`](../experiments/04_horizon.py) ·
Figure: [`outputs/04_horizon.png`](../outputs/04_horizon.png)

Same arc and constraints, `N in {5, 10, 20, 40}`.

| N | look-ahead | rms e_lat | final abs(e_lat) | solve time (mean) |
|---|---|---|---|---|
| 5 | 0.5 s | 0.3449 m | 0.1302 m | 2.35 ms |
| 10 | 1.0 s | 0.2862 m | 0.0596 m | 3.15 ms |
| 20 | 2.0 s | 0.2538 m | 0.0204 m | 4.24 ms |
| 40 | 4.0 s | 0.2454 m | 0.0038 m | 23.02 ms |

Within this scenario, lengthening the horizon improves tracking monotonically.
The N=40 run uses about 5.4x the mean solve time of N=20
for about a 3.3 % rms improvement. This is specific to this scenario and run.

### 3.3 MPC vs pure pursuit

Script: [`experiments/05_mpc_vs_baseline.py`](../experiments/05_mpc_vs_baseline.py) ·
Figure: [`outputs/05_mpc_vs_baseline.png`](../outputs/05_mpc_vs_baseline.png)

Same arc, same initial state, **same limits**: the baseline gets the 30 deg
steering bound and the same 0.05 rad/step slew rate, otherwise the comparison
would be unfair.

| controller | rms e_lat | max abs(e_lat) | final abs(e_lat) | max abs(delta) | mean abs(Delta delta) |
|---|---|---|---|---|---|
| MPC (N = 20, constrained) | 0.2538 m | 0.6047 m | 0.0204 m | 0.524 rad | 0.00699 |
| pure pursuit (arc advance = 4 m) | 0.2834 m | 0.6038 m | 0.0492 m | 0.450 rad | 0.00531 |

In this run the MPC ends closer to the path (0.020 m vs 0.049 m) and has lower
rms error. Pure pursuit has lower mean steering increments and a smaller peak
steering angle. These metrics do not establish general controller superiority.

The target is selected 4 m ahead in path arc length. The steering formula uses
the actual rear-axle-to-target distance, including near an endpoint. The previous
implementation incorrectly used the arc advance as that distance; these results
replace the old baseline numbers. Residual error in this discrete simulation
should not be described as an inherent property of ideal pure pursuit. Only one
look-ahead and one scenario were compared.

## What is reliable and what is not

Reliable (checked by tests or by closed-form algebra):

- the discrete bicycle model, its sign conventions and the Euler artefacts
  (Phase 0 tests);
- the MPC feasibility and constraint satisfaction (Phase 1/3 tests);
- the path geometry and the projection used for the lateral error
  (Phase 2 tests);
- the qualitative comparisons above, because both controllers run on the same
  plant, the same path and the same limits.

Not established by this project:

- anything about a real vehicle: the model has no tyre slip, no actuator lag,
  no delay and no sensor noise;
- general statements such as "MPC beats pure pursuit" - only this arc, this
  offset, these weights and this look-ahead were measured;
- stability guarantees: no terminal constraint or terminal set is used, and
  `P = 3Q` is a practical weight, not a Riccati solution;
- real-time feasibility on embedded hardware: timings are from one Windows
  laptop with the Python overhead included.

## Verification and expanded comparison

`python experiments/run_all.py` ran 61 tests and all eight experiments in the
locked Windows/Python 3.12 environment. The complete command statuses and logs
are in `outputs/run_manifest.json`; any failed command leaves `complete=false`
and returns a nonzero exit code. Numerical solver failure stops the original
comparison scripts before they publish new results.

The original comparison above uses N=20 and a 4 m baseline look-ahead for
teaching. It does not select the best baseline. The separate
[train/test benchmark](benchmark.md) searches four candidates for each controller,
freezes the selected parameters, and evaluates nominal and perturbed held-out
scenarios. It also reports timing distributions, deadline misses and a
sampling-time study. See [verification](verification.md) for exact scope.

## Machine

- Intel Core i9-13980HX
- Windows 11, CPython 3.12.6 (64-bit)
- numpy 2.3.2, matplotlib 3.10.5
- cvxpy 1.9.3 with OSQP 1.1.3 (Phase 1 QP)
- CasADi 3.8.2 with bundled IPOPT 3.14.19 (Phase 2/3 NLP)
- pytest 9.1.1