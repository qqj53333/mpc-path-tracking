"""Phase 0 - open-loop simulation of the kinematic bicycle model.

No optimisation here: a fixed, hand-made sequence of (a, delta) inputs is fed
into the discrete model and the resulting motion is plotted, so that the
"state is updated by the input" idea is visible before any MPC is involved.

Run from the repository root:

    python experiments/00_open_loop.py

Outputs (rebuilt on every run, no randomness involved):
    outputs/00_open_loop.png
    outputs/00_open_loop.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: the script only writes files

import matplotlib.pyplot as plt  # noqa: E402

from mpc_pt.models import BicycleParams  # noqa: E402
from mpc_pt.sim import simulate_open_loop  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs"

# --- scenario -------------------------------------------------------------
# Coordinate convention (model.md, section 0): the car points along +x, so a
# straight run has theta = 0 and theta_ref = 0.  L = 2.5 m, T = 0.1 s.
PARAMS = BicycleParams(wheelbase=2.5)
DT = 0.1
STATE0 = np.array([0.0, 0.0, 0.0, 0.0])  # x, y, theta, v

# (label, number of steps, a [m/s^2], delta [rad])
PHASES = [
    ("1 accelerate, a=+0.4", 50, +0.4, 0.0),
    ("2 constant speed", 50, 0.0, 0.0),
    ("3 left turn, delta=+0.30", 50, 0.0, +0.30),
    ("4 brake, a=-0.4", 50, -0.4, -0.30),
]


def build_controls() -> tuple[np.ndarray, list[int]]:
    """Stack the piecewise-constant phases into one (N, 2) input array."""
    blocks = []
    boundaries = [0]
    for _, n_steps, a, delta in PHASES:
        blocks.append(np.tile([a, delta], (n_steps, 1)))
        boundaries.append(boundaries[-1] + n_steps)
    return np.vstack(blocks), boundaries


def main() -> None:
    controls, boundaries = build_controls()
    result = simulate_open_loop(STATE0, controls, DT, PARAMS)
    states, t = result.states, result.t

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- single CSV: states plus the input applied during each interval -----
    table = np.column_stack([t, states, np.full((len(t), 2), np.nan)])
    table[:-1, 5:] = controls
    np.savetxt(
        OUTPUT_DIR / "00_open_loop.csv",
        table,
        delimiter=",",
        header="t,x,y,theta,v,a,delta",
        comments="",
        fmt="%.6f",
    )

    # --- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(15.0, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0])
    ax_path = fig.add_subplot(gs[0, 0])
    ax_theta = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[0, 2])
    fig.subplots_adjust(left=0.05, right=0.985, top=0.84, bottom=0.19, wspace=0.26)

    # light shading marks the four input phases in the time-domain panels
    for i in range(len(PHASES)):
        if i % 2 == 0:
            for ax in (ax_theta, ax_v):
                ax.axvspan(
                    t[boundaries[i]], t[boundaries[i + 1]],
                    color="0.75", alpha=0.30, lw=0,
                )
    for b in boundaries[1:-1]:
        for ax in (ax_theta, ax_v):
            ax.axvline(t[b], color="0.35", ls=":", lw=1.0)

    ax_path.plot(states[:, 0], states[:, 1], "-", lw=2.2, color="tab:blue")
    ax_path.plot(
        states[boundaries[1:-1], 0], states[boundaries[1:-1], 1],
        "o", ms=6, mfc="white", mec="0.2", mew=1.3,
    )
    ax_path.plot(states[0, 0], states[0, 1], "o", ms=8, color="tab:green")
    ax_path.plot(states[-1, 0], states[-1, 1], "s", ms=8, color="tab:red")
    ax_path.annotate(
        "start", (states[0, 0], states[0, 1]),
        textcoords="offset points", xytext=(0, 13), ha="center",
        fontsize=8, color="tab:green",
    )
    ax_path.annotate(
        "end", (states[-1, 0], states[-1, 1]),
        textcoords="offset points", xytext=(10, -2), fontsize=8, color="tab:red",
    )
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("y [m]")
    ax_path.set_title("x-y trajectory (open loop)")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.margins(0.12)
    ax_path.grid(alpha=0.3)

    for ax, col, ylabel, title in (
        (ax_theta, 2, r"$\theta$ [rad]", r"heading $\theta(t)$"),
        (ax_v, 3, "v [m/s]", "speed v(t)"),
    ):
        ax.plot(t, states[:, col], "-", lw=2.2, color="tab:purple")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xlim(t[0], t[-1])

    fig.suptitle(
        "Phase 0 - open-loop kinematic bicycle model   "
        f"(L = {PARAMS.wheelbase:.1f} m, T = {DT:.2f} s, "
        f"v0 = {STATE0[3]:.1f} m/s, theta0 = {STATE0[2]:.2f} rad)",
        y=0.975,
    )
    fig.text(
        0.5, 0.04,
        "input phases:   " + "   |   ".join(label for label, *_ in PHASES)
        + "   |   grey bands mark phases 1 and 3   |   o = phase boundary",
        ha="center", fontsize=8.5, color="0.25",
    )
    png = OUTPUT_DIR / "00_open_loop.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)

    # --- console summary ----------------------------------------------------
    print(f"steps           : {len(controls)}  (T = {DT} s, total {t[-1]:.1f} s)")
    print(
        f"final state     : x={states[-1, 0]:8.3f} m  y={states[-1, 1]:8.3f} m"
        f"  theta={states[-1, 2]:8.3f} rad  v={states[-1, 3]:7.3f} m/s"
    )
    path_length = float(np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)))
    print(f"path length     : {path_length:.3f} m")

    # heading check for every phase: theta_dot = (v/L) tan(delta) at the
    # start-of-step state, so the discrete change is T*(1/L)*tan(delta)*sum(v_k)
    for (label, _, a_cmd, delta_cmd), k0, k1 in zip(PHASES, boundaries[:-1], boundaries[1:]):
        if delta_cmd == 0.0:
            continue
        d_theta = states[k1, 2] - states[k0, 2]
        expected = DT * (1.0 / PARAMS.wheelbase) * np.tan(delta_cmd) * states[k0:k1, 3].sum()
        print(f"   phase '{label}': dtheta = {d_theta:+.6f} rad"
              f"  (discrete sum {expected:+.6f})")
    print(f"wrote           : {png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()