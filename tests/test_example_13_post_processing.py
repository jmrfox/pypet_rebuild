from pathlib import Path
from typing import Mapping, Any
import numpy as np

from pypet_rebuild.environment import Environment
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.storage import HDF5StorageService
from pypet_rebuild.parameters import Result, Parameter

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

def test_example_13_post_processing_round_trip(tmp_path):
    file_path = Path(tmp_path) / "example_13.h5"
    name = "Ex13_Post"

    env = Environment(trajectory=Trajectory(name=name), storage=HDF5StorageService(file_path=file_path))
    traj = env.trajectory
    traj.add_parameter(Parameter(name="x", value=0))
    traj.add_parameter(Parameter(name="y", value=0))

    xs = [1, 2, 3]
    ys = [4, 5]
    env.run_exploration(multiply, space={"x": xs, "y": ys})

    # Post-process: computes mean, max, and per-x sums
    post_process(file_path, name)

    storage = HDF5StorageService(file_path=file_path)
    loaded = storage.load(name)

    # Check run count
    assert len(loaded.list_runs()) == len(xs) * len(ys)

    mean_z = loaded.results["post.summary.mean_z"].value
    max_z = loaded.results["post.summary.max_z"].value
    per_x = loaded.results["post.per_x.sum"].value

    # Expected values
    zs = [x * y for x in xs for y in ys]
    assert np.isclose(mean_z, np.mean(zs))
    assert np.isclose(max_z, np.max(zs))

    expected_per_x = {str(x): float(sum(x * y for y in ys)) for x in xs}
    assert per_x == expected_per_x
