"""Phase 1 - linear MPC for the double integrator.

The car moves on a straight line, the controller picks the acceleration, and
the goal is to stop at `p_ref` without exceeding the speed and acceleration
limits.  There is no steering and no nonlinearity: this phase is about seeing
exactly what MPC does at each step.

Run from the repository root:

    python experiments/01_double_integrator_mpc.py

Outputs:
    outputs/01_double_integrator_single.png   one optimisation, solved once
    outputs/01_double_integrator_closed.png   the closed loop over 15 s
    outputs/01_double_integrator_closed.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from mpc_pt.models import di_step  # noqa: E402
from mpc_pt.mpc import LinearMPC, LinearMPCConfig  # noqa: E402
from mpc_pt.paths import Reference  # noqa: E402
from mpc_pt.sim import simulate_closed_loop  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs"
DT = 0.1
P_REF = 5.0
N_STEPS = 150          # 15 s
STATE0 = np.array([0.0, 0.0])   # [p, v]


def main() -> None:
    cfg = LinearMPCConfig(horizon=20, dt=DT, v_max=1.0, a_max=0.5)
    controller = LinearMPC(cfg)

    # the reference is "stand still at p_ref" for the whole run
    reference = Reference(
        states=np.tile([P_REF, 0.0], (N_STEPS + 1, 1)),
        controls=np.zeros((N_STEPS, 1)),
        dt=DT,
    )

    result = simulate_closed_loop(
        controller, STATE0, reference, N_STEPS, DT,
        plant_step=lambda x, u: di_step(x, u[0], DT),
    )
    states, controls, t = result.states, result.controls, result.t
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- single shot
    k_show = 10  # t = 1 s
    shot = controller.solve(states[k_show], reference.window(k_show, cfg.horizon).states)
    steps = np.arange(cfg.horizon + 1) * DT

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    axes[0].plot(steps, shot.states[:, 0], "-o", ms=3, lw=1.8, color="tab:blue")
    axes[0].axhline(P_REF, color="0.3", ls="--", lw=1, label="p_ref")
    axes[0].set_title("predicted position")
    axes[0].set_ylabel("p [m]")
    axes[1].plot(steps, shot.states[:, 1], "-o", ms=3, lw=1.8, color="tab:orange")
    axes[1].axhline(cfg.v_max, color="tab:red", ls=":", lw=1, label="+v_max")
    axes[1].set_title("predicted speed")
    axes[1].set_ylabel("v [m/s]")
    axes[2].step(steps[:-1], shot.controls[:, 0], where="post", lw=1.8, color="tab:green")
    axes[2].axhline(cfg.a_max, color="tab:red", ls=":", lw=1, label="+a_max")
    axes[2].axhline(-cfg.a_max, color="tab:red", ls=":", lw=1)
    axes[2].set_title("predicted input")
    axes[2].set_ylabel("a [m/s^2]")
    for ax in axes:
        ax.set_xlabel("time into the horizon [s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"one MPC solve at t = {t[k_show]:.1f} s   "
        f"(N = {cfg.horizon}, T = {DT} s, p = {states[k_show, 0]:.3f} m, "
        f"v = {states[k_show, 1]:.3f} m/s)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTPUT_DIR / "01_double_integrator_single.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- closed loop
    a = controls[:, 0]
    v = states[:, 1]
    v_series = v[:-1]  # state at the start of every control interval
    at_vmax = np.abs(np.abs(v_series) - cfg.v_max) < 1e-6
    at_amax = np.abs(np.abs(a) - cfg.a_max) < 1e-6

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True)
    axes[0].plot(t, states[:, 0], lw=2, color="tab:blue")
    axes[0].axhline(P_REF, color="0.3", ls="--", lw=1, label="p_ref")
    axes[0].set_ylabel("p [m]")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, v, lw=2, color="tab:orange")
    axes[1].axhline(cfg.v_max, color="tab:red", ls=":", lw=1, label="v_max")
    axes[1].scatter(t[:-1][at_vmax], v_series[at_vmax], s=14, color="tab:red",
                    zorder=3, label="constraint active")
    axes[1].set_ylabel("v [m/s]")

    axes[2].step(t[:-1], a, where="post", lw=2, color="tab:green", label="a(k)")
    for i, sign in enumerate((+1, -1)):
        axes[2].axhline(sign * cfg.a_max, color="tab:red", ls=":", lw=1,
                        label="a limits" if i == 0 else None)
    axes[2].scatter(t[:-1][at_amax], a[at_amax], s=14, color="tab:red", zorder=3,
                    label="constraint active")
    axes[2].set_ylabel("a [m/s^2]")
    axes[2].set_xlabel("t [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Phase 1 closed loop: double integrator MPC")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "01_double_integrator_closed.png", dpi=150)
    plt.close(fig)

    table = np.column_stack([t, states, np.full((N_STEPS + 1, 1), np.nan)])
    table[:-1, 3] = a
    np.savetxt(
        OUTPUT_DIR / "01_double_integrator_closed.csv", table,
        delimiter=",", header="t,p,v,a", comments="", fmt="%.6f",
    )

    # ---------------------------------------------------------------- summary
    print(f"horizon N       : {cfg.horizon}  (T = {DT} s -> {cfg.horizon * DT:.1f} s look-ahead)")
    print(f"limits          : |v| <= {cfg.v_max} m/s, |a| <= {cfg.a_max} m/s^2")
    print(f"final state     : p = {states[-1, 0]:.6f} m (target {P_REF}), v = {states[-1, 1]:.2e} m/s")
    print(f"max |v|         : {np.abs(v).max():.9f} m/s")
    print(f"max |a|         : {np.abs(a).max():.9f} m/s^2")
    print(f"a active steps  : {int(at_amax.sum())} of {N_STEPS}")
    print(f"v active steps  : {int(at_vmax.sum())} of {N_STEPS}")
    print(f"solve time      : mean {result.solve_times.mean()*1e3:.2f} ms, "
          f"max {result.solve_times.max()*1e3:.2f} ms, total {result.solve_times.sum():.2f} s")
    print(f"statuses        : {sorted(set(result.statuses))}")
    print(f"wrote           : {(OUTPUT_DIR / '01_double_integrator_closed.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
