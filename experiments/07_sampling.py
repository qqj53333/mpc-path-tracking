"""Sampling-time study with fixed physical horizon, steering rate and cost scale."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpc_pt.models import bicycle_step
from mpc_pt.paths import reference_from_path
from _benchmark import Scenario, run_case

OUTPUT_DIR = REPO_ROOT / "outputs"


def main():
    scenario = Scenario("sampling_left_R8", 8.0, 1, 2.0, -0.5)
    rows = []
    for dt in (0.2, 0.1, 0.05):
        n = round(2.0 / dt)
        row = run_case(scenario, "mpc", n, dt=dt, scale_cost_with_dt=True)
        ref = reference_from_path(scenario.path(), 2, dt, n)
        residuals = [np.linalg.norm((bicycle_step(ref.states[k], ref.controls[k], dt)
                                    - ref.states[k + 1])[:2]) for k in range(n)]
        row["reference_max_position_defect_m"] = float(max(residuals))
        rows.append(row)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "07_sampling.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    if not all(r["valid"] for r in rows):
        raise RuntimeError("sampling study contains invalid runs; inspect 07_sampling.json")
    with (OUTPUT_DIR / "07_sampling.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dt", "N", "reference_defect_m", "lateral_rms_m", "decision_p95_ms"])
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(dt=r["dt"], N=r["parameter"], reference_defect_m=r["reference_max_position_defect_m"],
                                 lateral_rms_m=r["metrics"]["lateral_rms_m"], decision_p95_ms=r["metrics"]["decision_p95_ms"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["dt"] for r in rows], [1000 * r["reference_max_position_defect_m"] for r in rows], "o-")
    axes[0].set_ylabel("one-step reference defect [mm]")
    axes[1].plot([r["dt"] for r in rows], [r["metrics"]["lateral_rms_m"] for r in rows], "o-")
    axes[1].set_ylabel("closed-loop lateral RMS [m]")
    for ax in axes:
        ax.set_xlabel("sampling time [s]"); ax.grid(alpha=.3)
    fig.suptitle("Fixed 2 s horizon; physical steering rate and cost scaling held consistent")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "07_sampling.png", dpi=150); plt.close(fig)
    for row in rows:
        print(f"T={row['dt']}: N={row['parameter']}, defect={row['reference_max_position_defect_m']:.8f} m, "
              f"RMS={row['metrics']['lateral_rms_m']:.6f} m, valid={row['valid']}")


if __name__ == "__main__":
    main()
