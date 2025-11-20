"""SIR model sweep example using pypet_rebuild.

This example sweeps over infection (beta) and recovery (gamma) rates of a
simple SIR model, runs the simulation for each combination, and stores per-run
results (including a pandas DataFrame time series) in an HDF5 file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from pypet_rebuild import (
    Environment,
    HDF5StorageService,
    Parameter,
    Trajectory,
)
from pypet_rebuild.utils import inspect_h5


def simulate(traj: Trajectory) -> Mapping[str, object]:
    """Run a simple SIR model and return a dict of results.

    Parameters are read from the trajectory under the `sir.*` namespace:
    - `sir.beta`: infection rate (float)
    - `sir.gamma`: recovery rate (float)
    - `sir.i0`: initial infected fraction (0..1)
    - `sir.t_max`: simulation horizon (float)
    - `sir.dt`: time step (float)
    """

    beta: float = float(traj.parameters["sir.beta"].value)
    gamma: float = float(traj.parameters["sir.gamma"].value)
    i0: float = float(traj.parameters["sir.i0"].value)
    t_max: float = float(traj.parameters["sir.t_max"].value)
    dt: float = float(traj.parameters["sir.dt"].value)

    s0 = 1.0 - i0
    y0 = np.array([s0, i0, 0.0], dtype=float)  # S, I, R

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        s, i, r = y
        ds = -beta * s * i
        di = beta * s * i - gamma * i
        dr = gamma * i
        return np.array([ds, di, dr])

    times = np.arange(0.0, t_max + 1e-12, dt)
    sol = solve_ivp(rhs, (0.0, t_max), y0, t_eval=times, vectorized=False)

    s = sol.y[0]
    i = sol.y[1]
    r = sol.y[2]

    peak_infected = float(np.max(i))
    final_susceptible = float(s[-1])

    df = pd.DataFrame({"S": s, "I": i, "R": r}, index=times)
    df.index.name = "t"

    return {
        "sir.peak_infected": peak_infected,
        "sir.final_susceptible": final_susceptible,
        "sir.timeseries": df,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["threads", "processes"], default="threads")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("examples/output") / "sir_sweep.h5")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    traj = Trajectory(name="sir-sweep")

    # Seed parameters (defaults). Values overwritten as needed during exploration.
    traj.add_parameter(Parameter(name="sir.beta", value=0.2))
    traj.add_parameter(Parameter(name="sir.gamma", value=0.1))
    traj.add_parameter(Parameter(name="sir.i0", value=0.01))
    traj.add_parameter(Parameter(name="sir.t_max", value=60.0))
    traj.add_parameter(Parameter(name="sir.dt", value=0.5))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    storage = HDF5StorageService(file_path=out_path)
    env = Environment(trajectory=traj, storage=storage)

    space = {
        "sir.beta": [0.2, 0.3, 0.4],
        "sir.gamma": [0.05, 0.1],
        # Could also vary i0 and dt in the grid if desired
    }

    if args.mode == "threads":
        env.run_exploration_parallel(simulate, space=space, _max_workers=args.workers)
    else:
        env.run_exploration_processes(simulate, space=space, _max_workers=args.workers)

    # Summary
    print(f"Completed runs: {len(traj.list_runs())}")
    if traj.list_runs():
        rid = traj.list_runs()[0]
        print("First run params:", traj.get_run_params(rid))
        res = traj.get_run_results(rid)
        print("First run results keys:", list(res.keys()))
        print("HDF5 saved to:", storage.file_path)
    if args.inspect:
        print("HDF5 output inspection:")
        print(inspect_h5(str(storage.file_path), show_values=True))


if __name__ == "__main__":
    main()
