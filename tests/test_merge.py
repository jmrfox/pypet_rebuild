from pathlib import Path
from typing import Mapping, Any

from pypet_rebuild.environment import Environment
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.merge import merge_trajectories
from pypet_rebuild.storage import HDF5StorageService


def _mul(traj: Trajectory) -> Mapping[str, Any]:
    x = int(traj.parameters["x"].value)
    y = int(traj.parameters["y"].value)
    z = x * y
    traj.add_result(Result(name="z", value=z))
    return {"z": z}


def test_merge_in_memory_removes_duplicates():
    # Traj1
    t1 = Trajectory(name="Traj1")
    t1.add_parameter(Parameter(name="x", value=0))
    t1.add_parameter(Parameter(name="y", value=0))
    e1 = Environment(trajectory=t1, storage=None)
    e1.run_exploration(_mul, space={"x": [1, 2, 3, 4], "y": [6, 7, 8]})

    # Traj2 with overlap
    t2 = Trajectory(name="Traj2")
    t2.add_parameter(Parameter(name="x", value=0))
    t2.add_parameter(Parameter(name="y", value=0))
    e2 = Environment(trajectory=t2, storage=None)
    e2.run_exploration(_mul, space={"x": [3, 4, 5, 6], "y": [7, 8, 9]})

    # Merge Traj2 into Traj1
    merge_trajectories(t1, t2, remove_duplicates=True)

    # Unique combos count: 12 + 12 - 4 = 20
    assert len(t1.list_runs()) == 20

    # Ensure no duplicate param snapshots: use set of (x,y)
    seen = set()
    for rid in t1.list_runs():
        p = t1.get_run_params(rid)
        tup = (p.get("x"), p.get("y"))
        assert tup not in seen
        seen.add(tup)


def test_merge_persisted_round_trip(tmp_path):
    fp = Path(tmp_path) / "example_03.h5"

    # Prepare and save both trajectories in same file before merge
    t1 = Trajectory(name="Traj1")
    t1.add_parameter(Parameter(name="x", value=0))
    t1.add_parameter(Parameter(name="y", value=0))
    e1 = Environment(trajectory=t1, storage=HDF5StorageService(file_path=fp))
    e1.run_exploration(_mul, space={"x": [1, 2], "y": [6, 7]})

    t2 = Trajectory(name="Traj2")
    t2.add_parameter(Parameter(name="x", value=0))
    t2.add_parameter(Parameter(name="y", value=0))
    e2 = Environment(trajectory=t2, storage=HDF5StorageService(file_path=fp))
    e2.run_exploration(_mul, space={"x": [2, 3], "y": [7, 8]})

    # Merge in-memory and save merged t1
    merge_trajectories(t1, t2, remove_duplicates=True)
    e1.storage.save(t1)  # type: ignore[union-attr]

    # Load back and verify run count and a sample by_run value
    storage = HDF5StorageService(file_path=fp)
    loaded = storage.load("Traj1")
    # unique combos: 4 + 4 - 1 overlap (x=2,y=7) = 7
    assert len(loaded.list_runs()) == 7
    # Check a sample run result exists
    rid = loaded.list_runs()[0]
    res = loaded.get_run_results(rid)
    assert "z" in res
