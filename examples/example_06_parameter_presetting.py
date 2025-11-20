from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any

import numpy as np

from pypet_rebuild.environment import Environment
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.storage import HDF5StorageService


def simulate_diff(traj: Trajectory) -> Mapping[str, Any]:
    p = traj.parameters
    diff_name = str(p["diff_name"].value)
    dt = float(p["dt"].value)
    steps = int(p["steps"].value)
    ic = np.asarray(p["initial_conditions"].value, dtype=float)

    if diff_name == "diff_lorenz":
        sigma = float(p["func_params.sigma"].value)
        beta = float(p["func_params.beta"].value)
        rho = float(p["func_params.rho"].value)

        def f(v: np.ndarray) -> np.ndarray:
            return np.array([
                sigma * (v[1] - v[0]),
                v[0] * (rho - v[2]) - v[1],
                v[0] * v[1] - beta * v[2],
            ], dtype=float)

    elif diff_name == "diff_roessler":
        a = float(p["func_params.a"].value)
        c = float(p["func_params.c"].value)
        b = a

        def f(v: np.ndarray) -> np.ndarray:
            return np.array([
                -v[1] - v[2],
                v[0] + a * v[1],
                b + v[2] * (v[0] - c),
            ], dtype=float)
    else:
        raise ValueError(f"Unknown diff_name: {diff_name}")

    path = np.zeros((steps, 3), dtype=float)
    path[0] = ic
    for i in range(1, steps):
        path[i] = path[i - 1] + dt * f(path[i - 1])

    traj.add_result(Result(name="euler.path", value=path))
    return {"euler.path": path}


essential_ics = [
    np.array([0.01, 0.01, 0.01], dtype=float),
    np.array([2.02, 0.02, 0.02], dtype=float),
    np.array([42.0, 4.2, 0.42], dtype=float),
]


def main() -> None:
    file_path = Path("examples/output") / "example_06.h5"

    traj = Trajectory(name="Example_06_Presetting")
    env = Environment(trajectory=traj, storage=HDF5StorageService(file_path=file_path))

    # Phase 1a: add parameters with control flow based on a preset selector
    traj.add_parameter(Parameter(name="steps", value=2000))
    traj.add_parameter(Parameter(name="dt", value=0.01))
    traj.add_parameter(Parameter(name="initial_conditions", value=np.array([0.0, 0.0, 0.0])))

    # Preset selector: switch between Lorenz and Roessler
    traj.add_parameter(Parameter(name="diff_name", value="diff_roessler"))

    if traj.parameters["diff_name"].value == "diff_lorenz":
        traj.add_parameter(Parameter(name="func_params.sigma", value=10.0))
        traj.add_parameter(Parameter(name="func_params.beta", value=8.0 / 3.0))
        traj.add_parameter(Parameter(name="func_params.rho", value=28.0))
    elif traj.parameters["diff_name"].value == "diff_roessler":
        traj.add_parameter(Parameter(name="func_params.a", value=0.1))
        traj.add_parameter(Parameter(name="func_params.c", value=14.0))
    else:
        raise ValueError("Unknown diff_name preset")

    # Phase 1b: exploration over initial conditions
    env.run_exploration(
        simulate_diff,
        space={
            "initial_conditions": essential_ics,
        },
    )

    # Reload and print a summary
    storage = HDF5StorageService(file_path=file_path)
    loaded = storage.load("Example_06_Presetting")
    print("Runs:", len(loaded.list_runs()))
    rid = loaded.list_runs()[0]
    res = loaded.get_run_results(rid)
    arr = np.asarray(res["euler.path"])  # type: ignore[index]
    print("Sample path shape:", arr.shape)


if __name__ == "__main__":
    main()
