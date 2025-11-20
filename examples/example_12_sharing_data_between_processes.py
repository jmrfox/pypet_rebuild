from __future__ import annotations

from multiprocessing import Manager
from pathlib import Path
from typing import Mapping, Any

from pypet_rebuild.environment import Environment
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.storage import HDF5StorageService


def simulate_with_shared(traj: Trajectory, shared_list) -> Mapping[str, Any]:
    x = int(traj.parameters["x"].value)
    y = int(traj.parameters["y"].value)
    z = x * y
    # Append minimal info to shared state (beware: resume should be False with shared state)
    try:
        shared_list.append((x, y, z))
    except Exception:
        # If not a proxy (threads mode), just ignore errors
        pass
    traj.add_result(Result(name="z", value=z))
    return {"z": z}


def main() -> None:
    file_path = Path("examples/output") / "example_12.h5"

    # Shared state across processes
    with Manager() as manager:
        shared = manager.list()

        # Environment and trajectory
        traj = Trajectory(name="Example_12_Shared")
        env = Environment(trajectory=traj, storage=HDF5StorageService(file_path=file_path))

        # Base parameters
        traj.add_parameter(Parameter(name="x", value=0))
        traj.add_parameter(Parameter(name="y", value=0))

        # Explore and run in processes; resume must be False when using shared external state
        env.run_exploration_processes(
            simulate_with_shared,
            space={"x": [1, 2, 3], "y": [4, 5]},
            _max_workers=2,
            resume=False,
            func_args=(shared,),
        )

        # Aggregate shared state after the run
        total = sum(z for (_, _, z) in list(shared))
        env.trajectory.add_result(Result(name="summary.total_z", value=int(total)))
        if env.storage is not None:
            env.storage.save(env.trajectory)

        print("Runs:", len(traj.list_runs()))
        print("Shared entries:", len(shared))
        print("Total z:", int(total))


if __name__ == "__main__":
    main()
