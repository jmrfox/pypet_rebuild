from pathlib import Path

from pypet_rebuild.environment import Environment
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.storage import HDF5StorageService
from pypet_rebuild.exploration import cartesian_product


def multiply(traj: Trajectory):
    x = int(traj.parameters["x"].value)
    y = int(traj.parameters["y"].value)
    z = x * y
    traj.add_result(Result(name="z", value=z))
    return {"z": z}


def main() -> None:
    file_path = Path("examples/output") / "example_04.h5"

    env = Environment(
        trajectory=Trajectory(name="Example_04_MP"),
        storage=HDF5StorageService(file_path=file_path),
    )

    traj = env.trajectory
    traj.add_parameter(Parameter(name="x", value=0))
    traj.add_parameter(Parameter(name="y", value=0))

    # Moderate grid for demonstration
    space = {"x": list(range(10)), "y": list(range(10))}

    # Use processes; on Windows this requires __main__ guard, which we have.
    env.run_exploration_processes(multiply, space, _max_workers=4)

    # Sanity print last run id
    print("Completed runs:", env.trajectory.list_runs()[-3:])


if __name__ == "__main__":
    main()
