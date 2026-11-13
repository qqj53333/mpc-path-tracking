"""Phase 3.3 - MPC versus the pure-pursuit baseline.

Same arc, same initial state, same steering limit.  Pure pursuit uses a fixed
look-ahead distance and no model; the MPC solves the optimisation problem of
Phase 2/3 at every step.

Run from the repository root:

    python experiments/05_mpc_vs_baseline.py
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

import _common  # noqa: E402
from mpc_pt.baselines import PurePursuitConfig, simulate_pure_pursuit  # noqa: E402
from mpc_pt.mpc import NMPCConfig, NonlinearMPC  # noqa: E402
from mpc_pt.paths import ArcPath  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs"
DT = 0.1
SPEED = 2.0
N_STEPS = 120
X0 = np.array([0.0, -0.5, 0.0, SPEED])


def lateral_series(path, states):
    return np.array([path.project(state[:2])[1] for state in states])


def main() -> None:
    path = ArcPath(radius=8.0, length=40.0, direction=+1.0)

    mpc_run = _common.run_tracking(
        path,
        NonlinearMPC(NMPCConfig(horizon=20, dt=DT, constraints=True, s_change=(0.1, 0.5))),
        X0, N_STEPS, DT, SPEED,
    )
    pp_result = simulate_pure_pursuit(
        path, X0, N_STEPS, DT, PurePursuitConfig(lookahead=4.0, speed=SPEED)
    )
    pp_lateral = lateral_series(path, pp_result.states)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    ax = axes[0]
    s_end = min(path.length, mpc_run.s[-1] + 5.0)
    s_ref = np.linspace(0.0, s_end, 400)
    rx, ry, _ = path.at(s_ref)
    ax.plot(rx, ry, "--", lw=1.4, color="0.45", label="reference path")
    ax.plot(mpc_run.result.states[:, 0], mpc_run.result.states[:, 1], lw=1.9,
            color="tab:blue", label="MPC (N = 20, constrained)")
    ax.plot(pp_result.states[:, 0], pp_result.states[:, 1], lw=1.7,
            color="tab:purple", label="pure pursuit (Ld = 4 m)")
    ax.set_title("arc: x-y trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    ax.plot(mpc_run.t, mpc_run.lateral, lw=1.9, color="tab:blue", label="MPC")
    ax.plot(pp_result.t, pp_lateral, lw=1.7, color="tab:purple", label="pure pursuit")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_title("lateral error")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e_lat [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Phase 3 - MPC vs pure pursuit on the same arc and initial offset")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "05_mpc_vs_baseline.png", dpi=150)
    plt.close(fig)

    print(f"{'controller':<16s}{'rms lat':>10s}{'max lat':>10s}{'final lat':>11s}"
          f"{'max|d|':>9s}{'mean|dd|':>10s}")
    for name, lateral, controls in (
        ("MPC", mpc_run.lateral, mpc_run.result.controls),
        ("pure pursuit", pp_lateral, pp_result.controls),
    ):
        print(f"{name:<16s}{np.sqrt(np.mean(lateral**2)):10.4f}"
              f"{np.abs(lateral).max():10.4f}{abs(lateral[-1]):11.4f}"
              f"{np.abs(controls[:, 1]).max():9.3f}"
              f"{np.abs(np.diff(controls[:, 1])).mean():10.5f}")
    print(f"\nwrote           : {(OUTPUT_DIR / '05_mpc_vs_baseline.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()