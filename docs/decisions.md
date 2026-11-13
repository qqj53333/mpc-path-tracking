# Design decisions

One entry per decision, with the reason and the status. Items marked
*provisional* were chosen so that the project could run end to end; they are
the owner's call to overturn.

## D-001 - Repository layout and how the package is imported

*Status: done (Phase 0).*

`src/mpc_pt/` is a normal package, but the scripts do **not** rely on
`pip install -e .`. `experiments/*.py` and `tests/conftest.py` prepend
`<repo>/src` to `sys.path` (three lines at the top of each file).

Reason: every experiment must run with a single command from a fresh clone,
and the required repository structure has no `pyproject.toml`. A `sys.path`
bootstrap keeps that promise with no installation step.

Trade-off: the package cannot be imported from an unrelated directory without
the bootstrap. If that ever matters, a minimal `pyproject.toml` is a one-file
change.

## D-002 - Discretisation: forward Euler

*Status: done (Phase 0).*

`X(k+1) = X(k) + T f(X(k), u(k))`, as required by the task order.

Measured consequence: with constant `v` and `delta` the discrete trajectory
is a regular polygon whose vertices lie on a circle of radius
`R (dtheta/2)/sin(dtheta/2)`, with its centre shifted by `vT/2` from the
analytic circle. For `v = 2 m/s`, `T = 0.1 s`, `delta = 0.3 rad` the measured
offset is 0.100 m. While braking, Euler also over-turns (phase 4 of the open
loop: -0.631046 rad discrete vs -0.618715 rad continuous).

## D-003 - Tolerance of the "circle" test

*Status: done (Phase 0).*

The relative radius error of the Euler polygon is `(vT/R)^2/24` to leading
order; the test allows twice that, `(vT/R)^2/12`, so the tolerance tightens
automatically when `T` is reduced. Absolute tolerances elsewhere: 1e-12 for
the straight-line test, 1e-9 m for the radius spread.

## D-004 - QP solver for the Phase 1 linear MPC: cvxpy + OSQP

*Status: done (Phase 1), evidence below.*

Evidence required by the task order - installable with `pip` on this machine
and able to run its official minimal example - was collected before any Phase
1 code depended on it:

    pip install cvxpy          -> cvxpy 1.9.3, osqp 1.1.3, clarabel 0.11.1
    minimal QP (cvxpy docs):
        minimize (x0-1)^2 + (x1-2)^2  s.t. x >= 0, x0 + x1 <= 2
        status: optimal,  x = [0.5, 1.5],  objective 0.5000

The QP is written with `cp.sum_squares` instead of `cp.quad_form` so that the
parametric problem stays DPP: the solver is set up once and each later solve
takes ~2 ms instead of ~56 ms (measured: 56 ms before, 1.7 ms after).

Why OSQP: the Phase 1 problem is a sparse convex QP with box constraints, which
is exactly OSQP's target. `eps_abs = eps_rel = 1e-9` with polishing gives
constraint violations of order 1e-11, asserted at 1e-6 in the tests.

## D-005 - Nonlinear MPC for the bicycle model: direct NLP with CasADi/IPOPT

*Status: done (Phase 2), provisional - the owner may prefer linearisation.*

Two options were considered:

1. **Direct nonlinear program** (chosen). The cost and the Euler model are
   written symbolically; IPOPT solves the whole horizon at once. Prediction and
   constraints are exact for the nonlinear model, at the price of a larger
   solve time (see the measured values in `results.md`).
2. **Sequential linearisation + QP.** Linearise around the current operating
   point every step, keep the QP machinery of Phase 1. Cheaper per solve and
   reuses OSQP, but the linearisation is only valid locally, wrap-around of the
   heading needs care, and a linearised model handles the `tan(delta)`
   nonlinearity poorly when the steering saturates.

The direct NLP was chosen because the bicycle model is small (4 states, 2
inputs) and the solver is fast enough at `T = 0.1 s`; the linearisation option
remains open for a version that has to run on weaker hardware.

Evidence: `pip install casadi` -> CasADi 3.8.2 with IPOPT 3.14.19 (MUMPS);
the official Rosenbrock example returns `Solve_Succeeded` and `[1, 1]`.

## D-006 - Heading error: wrap with `atan2(sin, cos)`

*Status: done (Phase 2), provisional - the owner may prefer `1 - cos`.*

Two options:

1. **`wrap(theta - theta_ref)`** (chosen). Exactly the physical angle
   difference, zero when aligned, and keeps the meaning of `Q_theta` as a
   weight on a squared angular error. Implemented in CasADi as
   `atan2(sin(e), cos(e))`, which is smooth everywhere except at `e = pi`,
   where the wrapped error jumps between the two angle branches. The
   squared angle penalty is continuous there but has a derivative discontinuity.
2. **`1 - cos(theta - theta_ref)`**. Smooth everywhere, so the solver sees no
   kink at all, but it is 2 at `pi` and flat there: the gradient vanishes when
   the vehicle points backwards, which is the worst possible place to lose
   gradient information. It also changes the scale of the heading weight.

The wrap was chosen because the heading error stays small in these scenarios
in the tested scenarios; recovery from an exactly reversed heading is not
established by this project. The
`atan2` form is the same function used in the tests, so code and checks agree.

## D-007 - Coordinate convention: option A (+x forward, `theta_ref = 0`)

*Status: done, following `model.md`.*

`model.md` section 0 chooses option A: the vehicle points along +x, a straight
run has `theta = 0`, `theta` grows counter-clockwise and `delta > 0` steers
left. Phase 0 originally started at `theta0 = pi/2` (heading +y). That is a
legal state but it contradicted the convention, and every later phase builds
on "straight means 0", so the scenario was changed to `theta0 = 0`. The model
code was not modified: it already accepted any heading.

## D-008 - Weights

*Status: done, following `model.md`.*

Phase 1 uses the template values `Q = diag(1, 0.1)`, `R = 0.1`, `P = Q`.
Phase 2/3 use the model document values `Q = diag(4, 4, 32.8, 1)`,
`R = diag(0.25, 3.64)`, `P = 3Q`.

`S = diag(0.1, 0.5)` in the `limits + S` variant is a teaching value chosen
so that the penalty is visible in principle; it turned out to be redundant
with the steering-rate constraint (see D-009).

## D-009 - Input-change penalty `S`: small measured effect in this scenario

*Status: done (Phase 3).*

`S` is part of the cost (`NMPCConfig.s_change`, default `(0, 0)`). With the
steering-rate constraint at 0.05 rad/step, adding `S = diag(0.1, 0.5)` changes
rms lateral error from 0.2536 to 0.2538 m, while mean steering increments remain
0.0070 rad at the reported precision. The steering-rate bound is active during
part of the transient. These observations do not prove that `S` is redundant
in other scenarios or isolate the cause of the small change.

A substantial smoothness benefit from `S` is not demonstrated by this setting.
A separate experiment with a looser rate bound would be needed to study it.

## D-010 - Fair baseline

*Status: done (Phase 3).*

The pure-pursuit baseline runs on the same plant, the same path, the same
initial state **and the same actuator limits** (30 deg steering, 0.05 rad/step
slew). Without the slew limit the baseline would jump to the steering bound in
one step and would look better than it could be on a real vehicle. One
look-ahead distance (4 m) was used; it was not tuned per scenario.

## D-011 - Geometry corrections (2026-11-14)

Arc projection uses a directed angle over one full turn and chooses the nearest
endpoint outside the finite arc. Full-circle and multi-turn paths are rejected
because position alone cannot identify a lap. Signed lateral error is measured
along the left normal at the projected point. Tests cover left/right turns,
rotated paths, points beyond half a circle and finite endpoints.

Pure pursuit selects a target by path arc advance and uses its actual Euclidean
distance in the steering geometry. A coincident target requests zero steering
subject to the existing rate bound; endpoint stopping is outside this experiment.
The baseline comparison was regenerated after this correction.

## Model provenance and personal acceptance

`docs/model.md` now includes the existing learning reference, clearly labelled
as reference material, plus implementation details for the double integrator
and input-change penalty. It is not represented as the owner's completed
independent explanation. Personal understanding and acceptance remain pending
in `docs/understanding.md`.
