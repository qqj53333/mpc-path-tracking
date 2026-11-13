"""mpc-path-tracking: MPC path tracking with a kinematic bicycle model.

Modules:

    models.py     vehicle dynamics (bicycle model, double integrator)
    paths.py      geometric paths and time-indexed reference trajectories
    mpc.py        LinearMPC (QP) and NonlinearMPC (NLP)
    sim.py        open-loop and closed-loop simulation loops
    baselines.py  pure-pursuit baseline
"""

__version__ = "1.0.0"

from .models import BicycleParams, bicycle_rhs, bicycle_step, di_step, wrap_angle
from .paths import ArcPath, Reference, StraightPath, reference_from_path

__all__ = [
    "ArcPath",
    "BicycleParams",
    "Reference",
    "StraightPath",
    "__version__",
    "bicycle_rhs",
    "bicycle_step",
    "di_step",
    "reference_from_path",
    "wrap_angle",
]
