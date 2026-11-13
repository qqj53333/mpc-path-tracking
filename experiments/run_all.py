"""Run tests and all experiments; record failures and return a nonzero exit code."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "00_open_loop.py", "01_double_integrator_mpc.py", "02_path_tracking.py",
    "03_constraints.py", "04_horizon.py", "05_mpc_vs_baseline.py",
    "06_benchmark.py", "07_sampling.py",
]


def main():
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    report = {"complete": False, "checks": []}
    manifest = output / "run_manifest.json"
    commands = [["-m", "pytest", "-q", "--disable-warnings"]]
    commands += [["experiments/" + name] for name in SCRIPTS]
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for args in commands:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        log = (result.stdout + result.stderr).replace(str(ROOT), "<repo>")
        log = log.replace(str(Path(sys.prefix)), "<environment>")
        report["checks"].append({"command": args, "exit_code": result.returncode, "output": log})
        manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{' '.join(args)}: exit {result.returncode}", flush=True)
        if result.returncode:
            print(log, flush=True)
            return result.returncode
    report["complete"] = True
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
