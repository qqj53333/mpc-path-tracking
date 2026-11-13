"""Equal candidate counts, held-out scenarios, and explicit robustness failures.

Run: python experiments/06_benchmark.py
"""
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _benchmark import TRAIN, TEST, MPC_CANDIDATES, PP_CANDIDATES, NOISE_STD, run_case, select_parameter

OUTPUT_DIR = REPO_ROOT / "outputs"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, candidates in (("mpc", MPC_CANDIDATES), ("pure_pursuit", PP_CANDIDATES)):
        for parameter in candidates:
            for scenario in TRAIN:
                row = run_case(scenario, kind, parameter)
                row.update(split="train", condition="nominal")
                rows.append(row)
            print(f"training {kind} parameter={parameter} complete", flush=True)
    chosen, scores = {}, {}
    for kind in ("mpc", "pure_pursuit"):
        chosen[kind], scores[kind] = select_parameter(rows, kind)
    print("Frozen parameters:", chosen, flush=True)
    conditions = [("nominal", {}), ("wheelbase_plus20pct", {"plant_wheelbase": 3.0}),
                  ("delay_100ms", {"delay_steps": 1})]
    conditions += [(f"noise_seed{seed}", {"noisy": True, "seed": seed}) for seed in (7, 11, 19)]
    for condition, kwargs in conditions:
        for scenario in TEST:
            for kind in ("mpc", "pure_pursuit"):
                row = run_case(scenario, kind, chosen[kind], **kwargs)
                row.update(split="test", condition=condition)
                rows.append(row)
        print(f"held-out {condition} complete", flush=True)
    report = dict(protocol={
        "selection": "minimum mean training lateral RMS; disqualify candidates with any invalid run",
        "budget": "4 parameter candidates x 3 training scenarios for each controller; not equal CPU time",
        "training_scenarios": [asdict(s) for s in TRAIN],
        "held_out_scenarios": [asdict(s) for s in TEST],
        "noise_std_x_y_heading_speed": NOISE_STD.tolist(),
        "noise_seeds": [7, 11, 19],
        "timing": "per control decision, including Python, excluding construction/plant/plotting",
        "failure_policy": "stop and record; never silently replace controls or exclude failed rows",
    }, selected_parameters=chosen, training_scores=scores, runs=rows)
    (OUTPUT_DIR / "06_benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    flat = [{**{k: row[k] for k in ("split", "condition", "scenario", "controller", "parameter", "valid", "reason", "accepted_decisions", "solver_failures")},
             **(row["metrics"] or {})} for row in rows]
    fields = list(dict.fromkeys(k for row in flat for k in row))
    with (OUTPUT_DIR / "06_benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(flat)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, label in zip(axes, ("lateral_rms_m", "steering_rate_rms_radps", "decision_p95_ms"),
                                ("lateral RMS [m]", "steering-rate RMS [rad/s]", "decision p95 [ms]")):
        for i, kind in enumerate(("mpc", "pure_pursuit")):
            values = [next(r for r in rows if r["split"] == "test" and r["condition"] == "nominal"
                           and r["scenario"] == s.name and r["controller"] == kind) for s in TEST]
            ax.bar(np.arange(len(TEST)) + (i - 0.5) * .36,
                   [r["metrics"][metric] if r["valid"] else np.nan for r in values], .36, label=kind)
        ax.set_xticks(range(len(TEST)), ["left R6", "left R12", "right R10", "straight"], rotation=20)
        ax.set_ylabel(label); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("Held-out nominal scenarios; parameters selected on separate training scenarios")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "06_benchmark.png", dpi=150); plt.close(fig)
    print(f"Recorded {len(rows)} runs; invalid: {sum(not r['valid'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
