# Verification (2026-11-14)

Environment: Windows 11, Intel Core i9-13980HX, CPython 3.12.6 (64-bit).
Full installed package versions remain in `requirements-windows-lock.txt`.
The isolated environment created during the initial review was reused; no
new dependencies were added for this revision.

## Executed checks

`python experiments/run_all.py` ran **61 tests** and all **eight experiment
scripts**, with every exit code zero. The test runner reports 14 upstream
Matplotlib/pyparsing deprecation warnings. These are not test failures.
[Captured results](verification-results.json) and
[the run manifest](../outputs/run_manifest.json) are included.

The additional tests reproduce solver exhaustion, failure after a valid
accelerating step, invalid numeric output, N=1 shape preservation, 2 m initial
offsets without physical actuator limits, scalar path queries and finite-path
reference overflow. Comparison-script tests ensure failure stops before new
figures are written. Existing figures from older runs are not automatically
deleted; a failed manifest must not be treated as fresh verification.

The benchmark retained 72 rollouts with
0 invalid runs. Its validity checks include
numerical solution residuals, true plant states, commanded/applied inputs and
initial steering changes. Timing misses are reported separately from validity.
All three sampling-time runs were valid.

## Generated artifact hashes

These hashes identify this recorded run. Benchmark CSV/JSON timing fields vary
between runs, so byte equality of those files is not a reproducibility criterion.
Fixed-seed trajectories and non-timing metrics are expected to agree within
numerical tolerance; plots and floating-point last digits can vary by environment.

| File | SHA-256 |
|---|---|
| 00_open_loop.csv | 3a2b57f8f2d2bcbfef194dd3f8b372119ca2395f587e0a48a25ae94a002b9984 |
| 00_open_loop.png | 6f338a5e46286c8b37452424f3498d3a9eda1759b3f576b6aa46b456cc29a1c6 |
| 01_double_integrator_closed.csv | aa1b2941db00881dafb35265294537b9458b760da277d1dfbfc8d7545534f9ab |
| 01_double_integrator_closed.png | 7a9fdbb6d9ba728abac1a7527e2bb71f82ac800dd30cf1bfd867751040bd747a |
| 01_double_integrator_single.png | 3c0de1d380558c917842faccd9cb356a6abf4ec2a992ae0e1c3c4882a3ec26e0 |
| 02_arc.csv | 7c6cde9bbb891944361f904da328a8c60c4f79f7d02cfa74dbade058f173cd0d |
| 02_path_tracking.png | cb2d686f5dac0ed5b76c93404f5002d7bd10f4324f4bb81e13cd7268cd9a2752 |
| 02_straight.csv | 8003c257c381d55b70c437db79d0d930f3fa3dbecb57a1e79a2183a07d8dd913 |
| 03_constraints.png | 50df715ac6a71f60969b4b540e52f6d9288885fba14fcabced3374146b601945 |
| 04_horizon.png | 31c3308f4b44c6e282b9bf96bcc109683930cef90b8fda76725dfc12af9dbdb0 |
| 05_mpc_vs_baseline.png | 7ce29a186a6122cb70a5239a49c984e212ff1ccde928a3dbcf8e42e29b68b5f3 |
| 06_benchmark.csv | 793173c68543ab89d3f06a94e7b5a319a3878931003b01a37063aa78ac292c67 |
| 06_benchmark.png | 05a31aa61b01ec20699edc6ac2770f28383fa1f6bcaf983f430308637e5cbea0 |
| 07_sampling.csv | d320ec6880918f77c699434a2ad2f8ee3a740a4fc7f5f5b4b9fc04feeb7fde02 |
| 07_sampling.png | 307ce27a9b43023f44f753a21b1042020ccd652bd3658e28800ebf587dd3bafa |

## Scope

This verification concerns the declared numerical simulations. Personal
understanding remains pending in [understanding.md](understanding.md).
No global convergence, recursive feasibility or real-vehicle claim is made.
