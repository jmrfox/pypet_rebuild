from pathlib import Path

from pypet_rebuild.environment import Environment
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.storage import HDF5StorageService


def _sim(traj: Trajectory):
    # simple: z = x + y
    x = traj.parameters["x"].value
    y = traj.parameters["y"].value
    z = x + y
    traj.add_result(Result(name="z", value=z))
    return {"z": z}


def test_runs_metadata_persisted_and_loaded(tmp_path):
    file_path = tmp_path / "runs_meta.h5"

    t = Trajectory(name="runs_meta")
    t.add_parameter(Parameter(name="x", value=0))
    t.add_parameter(Parameter(name="y", value=0))

    storage = HDF5StorageService(file_path=Path(file_path))
    env = Environment(trajectory=t, storage=storage)

    space = {"x": [1, 2], "y": [6, 7]}
    env.run_exploration(_sim, space)

    # load fresh trajectory from disk
    loaded = storage.load("runs_meta")

    runs = loaded.list_runs()
    assert runs == [f"{i:05d}" for i in range(4)]

    # params snapshot should match
    first_params = loaded.get_run_params("00000")
    assert set(first_params.keys()) == {"x", "y"}

    # timestamp should be present in the internal record
    assert getattr(loaded, "_run_records")[0].get("timestamp")

    # by_run mirror should enable collecting result values
    zs = loaded.collect_runs("z")
    assert len(zs) == 4
