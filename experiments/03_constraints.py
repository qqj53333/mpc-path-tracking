"""Phase 3.1 - what the constraints and the input-change penalty actually do.

Same arc, same initial offset, three controllers:

1. `no constraints`  - the Phase 2 setting: no limits, no input-change penalty.
2. `limits`         - v, a, delta and delta-rate limits on.
3. `limits + S`     - the same limits plus a penalty on Delta u.

Run from the repository root:

    python experiments/03_constraints.py
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
S_CHANGE = (0.1, 0.5)   # [1/(m/s^2)^2, 1/rad^2], a teaching value, not tuned


def metrics(run) -> dict:
    result = run.result
    d_delta = np.diff(result.controls[:, 1])
    return {
        "max |delta|": np.abs(result.controls[:, 1]).max(),
        "mean |Delta delta|": np.abs(d_delta).mean(),
        "max |a|": np.abs(result.controls[:, 0]).max(),
        "max |lat|": np.abs(run.lateral).max(),
        "rms lat": np.sqrt(np.mean(run.lateral**2)),
        "final |lat|": abs(run.lateral[-1]),
        "solve ms": result.solve_times.mean() * 1e3,
    }


def main() -> None:
    path = ArcPath(radius=8.0, length=40.0, direction=+1.0)
    variants = [
        ("no constraints", NMPCConfig(horizon=20, dt=DT, constraints=False, s_change=(0.0, 0.0)), "tab:blue"),
        ("limits", NMPCConfig(horizon=20, dt=DT, constraints=True, s_change=(0.0, 0.0)), "tab:orange"),
        ("limits + S", NMPCConfig(horizon=20, dt=DT, constraints=True, s_change=S_CHANGE), "tab:green"),
    ]

    runs = []
    for name, config, colour in variants:
        run = _common.run_tracking(path, NonlinearMPC(config), X0, N_STEPS, DT, SPEED)
        runs.append((name, config, colour, run))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

    s_end = min(path.length, runs[0][3].s[-1] + 5.0)
    s_ref = np.linspace(0.0, s_end, 400)
    rx, ry, _ = path.at(s_ref)
    ax = axes[0, 0]
    ax.plot(rx, ry, "--", lw=1.4, color="0.45", label="reference path")
    for name, _, colour, run in runs:
        ax.plot(run.result.states[:, 0], run.result.states[:, 1], lw=1.8,
                color=colour, label=name)
    ax.set_title("arc: x-y trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[0, 1]
    for name, _, colour, run in runs:
        ax.plot(run.t[:-1], run.result.controls[:, 1], lw=1.6, color=colour, label=name)
    ax.axhline(0.5236, color="tab:red", ls=":", lw=1, label="delta_max = 30 deg")
    ax.axhline(-0.5236, color="tab:red", ls=":", lw=1)
    ax.set_title("steering command")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("delta [rad]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name, _, colour, run in runs:
        ax.plot(run.t, run.lateral, lw=1.8, color=colour, label=name)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_title("lateral error")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e_lat [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for name, _, colour, run in runs:
        ax.step(run.t[:-1], run.result.controls[:, 0], where="post", lw=1.5,
                color=colour, label=name)
    ax.axhline(2.0, color="tab:red", ls=":", lw=1, label="a limits")
    ax.axhline(-2.0, color="tab:red", ls=":", lw=1)
    ax.set_title("acceleration command")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("a [m/s^2]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Phase 3 - constraints vs no constraints (arc, v_ref = 2 m/s, offset 0.5 m)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "03_constraints.png", dpi=150)
    plt.close(fig)

    print(f"{'variant':<18s}" + "".join(f"{k:>18s}" for k in metrics(runs[0][3])))
    for name, _, _, run in runs:
        m = metrics(run)
        print(f"{name:<18s}" + "".join(f"{m[k]:18.4f}" for k in m))

    print(f"\nwrote           : {(OUTPUT_DIR / '03_constraints.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()