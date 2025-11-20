from pathlib import Path
from typing import Any

from pypet_rebuild.environment import Environment
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.storage import HDF5StorageService


def _simulate_add_one(traj: Trajectory) -> dict[str, Any]:
    # Read x and write a result; also return mapping to exercise merge path
    x = traj.parameters["x"].value if "x" in traj.parameters else 0
    val = x + 1
    traj.add_result(Result(name="y", value=val))
    return {"y": val}


def test_resume_in_sequential(tmp_path):
    t = Trajectory(name="t1")
    t.add_parameter(Parameter(name="x", value=0))
    storage = HDF5StorageService(file_path=Path(tmp_path / "t1.h5"))
    env = Environment(trajectory=t, storage=storage)

    space = {"x": [0, 1, 2]}
    env.run_exploration(_simulate_add_one, space)
    first_runs = env.trajectory.list_runs()

    # Re-run with resume; should not add new runs
    env.run_exploration(_simulate_add_one, space, resume=True)
    second_runs = env.trajectory.list_runs()

    assert first_runs == second_runs


def test_resume_in_threads(tmp_path):
    t = Trajectory(name="t2")
    t.add_parameter(Parameter(name="x", value=0))
    storage = HDF5StorageService(file_path=Path(tmp_path / "t2.h5"))
    env = Environment(trajectory=t, storage=storage)

    space = {"x": [0, 1, 2, 3]}
    env.run_exploration_parallel(_simulate_add_one, space, _max_workers=2)
    first_runs = set(env.trajectory.list_runs())

    env.run_exploration_parallel(_simulate_add_one, space, _max_workers=2, resume=True)
    second_runs = set(env.trajectory.list_runs())

    assert first_runs == second_runs


def _simulate_process(traj: Trajectory) -> dict[str, Any]:
    # Separate top-level function for process pickling
    x = traj.parameters["x"].value
    val = x * 2
    traj.add_result(Result(name="z", value=val))
    return {"z": val}


def test_resume_in_processes(tmp_path):
    t = Trajectory(name="t3")
    t.add_parameter(Parameter(name="x", value=0))
    storage = HDF5StorageService(file_path=Path(tmp_path / "t3.h5"))
    env = Environment(trajectory=t, storage=storage)

    space = {"x": [0, 1, 2]}
    env.run_exploration_processes(_simulate_process, space, _max_workers=2)
    first_runs = set(env.trajectory.list_runs())

    env.run_exploration_processes(_simulate_process, space, _max_workers=2, resume=True)
    second_runs = set(env.trajectory.list_runs())

    assert first_runs == second_runs
