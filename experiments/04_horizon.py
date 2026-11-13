"""Phase 3.2 - how the prediction horizon changes the behaviour.

Same arc, same initial offset, same constraints; only `N` changes:
5 (0.5 s), 10 (1 s), 20 (2 s) and 40 (4 s) steps.

Run from the repository root:

    python experiments/04_horizon.py
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
from mpc_pt.mpc import NMPCConfig, NonlinearMPC  # noqa: E402
from mpc_pt.paths import ArcPath  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs"
DT = 0.1
SPEED = 2.0
N_STEPS = 120
X0 = np.array([0.0, -0.5, 0.0, SPEED])
HORIZONS = (5, 10, 20, 40)


def main() -> None:
    path = ArcPath(radius=8.0, length=40.0, direction=+1.0)
    colours = ["tab:red", "tab:orange", "tab:green", "tab:blue"]

    runs = []
    for n, colour in zip(HORIZONS, colours):
        config = NMPCConfig(horizon=n, dt=DT, constraints=True, s_change=(0.1, 0.5))
        run = _common.run_tracking(path, NonlinearMPC(config), X0, N_STEPS, DT, SPEED)
        runs.append((n, colour, run))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    ax = axes[0]
    for n, colour, run in runs:
        ax.plot(run.t, run.lateral, lw=1.8, color=colour, label=f"N = {n}")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_title("lateral error")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e_lat [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    s_end = min(path.length, runs[0][2].s[-1] + 5.0)
    s_ref = np.linspace(0.0, s_end, 400)
    rx, ry, _ = path.at(s_ref)
    ax.plot(rx, ry, "--", lw=1.4, color="0.45", label="reference path")
    for n, colour, run in runs:
        ax.plot(run.result.states[:, 0], run.result.states[:, 1], lw=1.6,
                color=colour, label=f"N = {n}")
    ax.set_title("x-y trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Phase 3 - prediction horizon (arc, delta limit 30 deg, v_ref = 2 m/s)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "04_horizon.png", dpi=150)
    plt.close(fig)

    header = f"{'N':>4s}{'rms lat':>10s}{'max lat':>10s}{'final lat':>11s}{'max|d|':>9s}{'max|a|':>8s}{'solve ms':>10s}"
    print(header)
    for n, _, run in runs:
        lat = run.lateral
        print(
            f"{n:4d}{np.sqrt(np.mean(lat**2)):10.4f}{np.abs(lat).max():10.4f}"
            f"{abs(lat[-1]):11.4f}{np.abs(run.result.controls[:, 1]).max():9.3f}"
            f"{np.abs(run.result.controls[:, 0]).max():8.3f}"
            f"{run.result.solve_times.mean()*1e3:10.2f}"
        )
    print(f"\nwrote           : {(OUTPUT_DIR / '04_horizon.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()