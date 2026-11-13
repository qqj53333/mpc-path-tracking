"""Phase 2 - MPC path tracking with the kinematic bicycle model.

Two reference paths, one initial lateral offset each: a straight line and a
circular arc of radius 8 m. This phase has no physical input/state constraints
and no input-change penalty; those come in Phase 3.
A steering-domain guard always keeps tan(delta) on its principal branch.  The cost is
the time-indexed tracking error with `Q`, `R` and the terminal weight `P`.

Run from the repository root:

    python experiments/02_path_tracking.py

Outputs:
    outputs/02_path_tracking.png
    outputs/02_straight.csv, outputs/02_arc.csv
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

import _common  # noqa: E402  (lives next to this script)
from mpc_pt.mpc import NMPCConfig, NonlinearMPC  # noqa: E402
from mpc_pt.paths import ArcPath, StraightPath  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs"
DT = 0.1
SPEED = 2.0


def main() -> None:
    # Phase 2: no constraints, no input-change penalty
    config = NMPCConfig(horizon=20, dt=DT, constraints=False, s_change=(0.0, 0.0))
    controller = NonlinearMPC(config)

    scenarios = [
        (
            "straight",
            StraightPath(start=(0.0, 0.0), theta=0.0, length=60.0),
            np.array([0.0, 0.5, 0.0, SPEED]),
            200,
        ),
        (
            "arc",
            ArcPath(start=(0.0, 0.0), theta=0.0, radius=8.0, length=40.0, direction=+1.0),
            np.array([0.0, -0.5, 0.0, SPEED]),
            120,
        ),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0),
                             gridspec_kw={"width_ratios": [1.0, 1.4]})

    for row, (name, path, x0, n_steps) in enumerate(scenarios):
        run = _common.run_tracking(path, controller, x0, n_steps, DT, SPEED)
        _common.save_tracking_csv(OUTPUT_DIR / f"02_{name}.csv", run)
        print(_common.describe(name, run, DT))

        ax = axes[row, 0]
        # draw the path only a little beyond where the vehicle actually goes
        s_ref = np.linspace(0.0, min(path.length, run.s[-1] + 5.0), 400)
        rx, ry, _ = path.at(s_ref)
        ax.plot(rx, ry, "--", lw=1.4, color="0.45", label="reference path")
        ax.plot(run.result.states[:, 0], run.result.states[:, 1], lw=2, color="tab:blue",
                label="vehicle")
        ax.plot(run.result.states[0, 0], run.result.states[0, 1], "o", ms=7,
                color="tab:green", label="start")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if name == "straight":
            # a 0.5 m merge over 40 m cannot be drawn to scale in this panel,
            # so the y axis is exaggerated; the arc below is to scale
            ax.set_title("straight: x-y trajectory (y axis exaggerated)")
            ax.set_xlim(-2.0, 45.0)
            ax.set_ylim(-0.8, 0.8)
        else:
            ax.set_title("arc: x-y trajectory")
            ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        ax = axes[row, 1]
        ax.plot(run.t, run.lateral, lw=1.8, color="tab:red", label="lateral error")
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_title(f"{name}: lateral error (positive = left of path)")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("e_lat [m]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Phase 2 - bicycle MPC path tracking (N = {config.horizon}, "
        f"T = {DT} s, v_ref = {SPEED} m/s, no actuator limits)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "02_path_tracking.png", dpi=150)
    plt.close(fig)

    print(f"wrote           : {(OUTPUT_DIR / '02_path_tracking.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
