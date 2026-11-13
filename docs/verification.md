# Verification (2026-11-14)

## Environment

- Windows 11, Intel Core i9-13980HX, CPython 3.12.6 (64-bit).
- A new isolated environment was created during the review and populated from
  `requirements.txt`; the complete installed versions are recorded in
  `requirements-windows-lock.txt`.
- `python -m pip check`: no broken requirements.

## Clean-source reproduction

The final source was copied to a separate directory without `outputs/`, Python
bytecode, test caches or Git metadata. Using the isolated environment:

```text
python -m pytest -q  # 35 passed
python experiments/00_open_loop.py
python experiments/01_double_integrator_mpc.py
python experiments/02_path_tracking.py
python experiments/03_constraints.py
python experiments/04_horizon.py
python experiments/05_mpc_vs_baseline.py
```

All commands returned exit code 0. All 7 PNGs and 4 CSVs were regenerated from
scratch and matched the repository versions byte for byte (SHA-256).
[Captured verification output](verification-results.json) is included.
The timing figures in results.md use the separate recorded
[experiment run](experiment-results.json); wall-clock times vary.

## Generated artifact hashes

| File | SHA-256 |
|---|---|
| `00_open_loop.csv` | `3a2b57f8f2d2bcbfef194dd3f8b372119ca2395f587e0a48a25ae94a002b9984` |
| `00_open_loop.png` | `6f338a5e46286c8b37452424f3498d3a9eda1759b3f576b6aa46b456cc29a1c6` |
| `01_double_integrator_closed.csv` | `aa1b2941db00881dafb35265294537b9458b760da277d1dfbfc8d7545534f9ab` |
| `01_double_integrator_closed.png` | `7a9fdbb6d9ba728abac1a7527e2bb71f82ac800dd30cf1bfd867751040bd747a` |
| `01_double_integrator_single.png` | `3c0de1d380558c917842faccd9cb356a6abf4ec2a992ae0e1c3c4882a3ec26e0` |
| `02_arc.csv` | `ec48e45f6756d881abb4141b9268b04984f0579eea442a35338a1759c1667d10` |
| `02_path_tracking.png` | `e3249e515dd1cb5a9d052c8e785dc03d2786577d8e52598d8eb62cda3483d477` |
| `02_straight.csv` | `e36517ae764a16d7f2dd990d153679354dc3a8f75969b800a943d889513e8419` |
| `03_constraints.png` | `b1d82bfe110a085e68170480b3cb02883ac06f02365bf315da757a4d961bbdd3` |
| `04_horizon.png` | `31c3308f4b44c6e282b9bf96bcc109683930cef90b8fda76725dfc12af9dbdb0` |
| `05_mpc_vs_baseline.png` | `c3be65922ce9e9d0384bbf998c29037cfc071718beea2447522050dcdcc3b042` |


## Scope

The tests cover the documented scenarios and the corrected geometry boundaries.
They do not establish real-vehicle safety, global convergence, arbitrary initial
conditions or performance on untested operating systems. Personal understanding
and acceptance remain separate in [understanding.md](understanding.md).
