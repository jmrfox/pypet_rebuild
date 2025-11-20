from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any

import numpy as np

from pypet_rebuild.environment import Environment
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.storage import HDF5StorageService


def multiply(traj: Trajectory) -> Mapping[str, Any]:
    x = int(traj.parameters["x"].value)
    y = int(traj.parameters["y"].value)
    z = x * y
    traj.add_result(Result(name="z", value=z))
    return {"z": z}


def post_process(file_path: Path, traj_name: str) -> None:
    storage = HDF5StorageService(file_path=file_path)
    traj = storage.load(traj_name)

    # Collect per-run results and params
    zs = np.array(traj.collect_runs("z"), dtype=float)

    # Summary metrics
    mean_z = float(np.mean(zs)) if zs.size else 0.0
    max_z = float(np.max(zs)) if zs.size else 0.0

    # Per-x aggregation using recorded run params
    per_x_sum: dict[str, float] = {}
    for rid in traj.list_runs():
        params = traj.get_run_params(rid)
        results = traj.get_run_results(rid)
        x = str(params.get("x"))
        z = float(results.get("z", 0.0))
        per_x_sum[x] = per_x_sum.get(x, 0.0) + z

    traj.add_result(Result(name="post.summary.mean_z", value=mean_z))
    traj.add_result(Result(name="post.summary.max_z", value=max_z))
    traj.add_result(Result(name="post.per_x.sum", value=per_x_sum))

    storage.save(traj)


def main() -> None:
    file_path = Path("examples/output") / "example_13.h5"

    # Phase 1: run exploration and persist
    env = Environment(trajectory=Trajectory(name="Example_13_PostProcessing"), storage=HDF5StorageService(file_path=file_path))
    traj = env.trajectory
    traj.add_parameter(Parameter(name="x", value=0))
    traj.add_parameter(Parameter(name="y", value=0))

    env.run_exploration(multiply, space={"x": [1, 2, 3], "y": [4, 5]})

    # Phase 2: post-processing (reload, aggregate, save derived results)
    post_process(file_path, "Example_13_PostProcessing")

    # Preview
    storage = HDF5StorageService(file_path=file_path)
    merged = storage.load("Example_13_PostProcessing")
    print("Runs:", len(merged.list_runs()))
    print("mean_z:", merged.results["post.summary.mean_z"].value)
    print("max_z:", merged.results["post.summary.max_z"].value)
    print("per_x.sum:", merged.results["post.per_x.sum"].value)


if __name__ == "__main__":
    main()
