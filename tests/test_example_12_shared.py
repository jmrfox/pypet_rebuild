from multiprocessing import Manager
from pathlib import Path

from pypet_rebuild.environment import Environment
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.storage import HDF5StorageService


def simulate_with_shared(traj: Trajectory, shared_list):
    x = int(traj.parameters["x"].value)
    y = int(traj.parameters["y"].value)
    z = x * y
    try:
        shared_list.append((x, y, z))
    except Exception:
        pass
    traj.add_result(Result(name="z", value=z))
    return {"z": z}


def test_shared_state_across_processes(tmp_path):
    file_path = Path(tmp_path) / "example_12.h5"

    with Manager() as manager:
        shared = manager.list()

        traj = Trajectory(name="Ex12_Shared")
        env = Environment(trajectory=traj, storage=HDF5StorageService(file_path=file_path))

        traj.add_parameter(Parameter(name="x", value=0))
        traj.add_parameter(Parameter(name="y", value=0))

        env.run_exploration_processes(
            simulate_with_shared,
            space={"x": [1, 2], "y": [3, 4]},
            _max_workers=2,
            resume=False,
            func_args=(shared,),
        )

        # After processes complete, aggregate from shared state
        total = sum(z for (_, _, z) in list(shared))
        env.trajectory.add_result(Result(name="summary.total_z", value=int(total)))
        if env.storage is not None:
            env.storage.save(env.trajectory)

    # Verify persisted data
    storage = HDF5StorageService(file_path=file_path)
    loaded = storage.load("Ex12_Shared")

    # 4 runs expected (2 x 2 grid)
    assert len(loaded.list_runs()) == 4
    # Aggregated result exists and equals expected sum
    res = loaded.results["summary.total_z"].value
    assert int(res) == (1*3 + 1*4 + 2*3 + 2*4)
