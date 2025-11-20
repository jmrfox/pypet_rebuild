from pathlib import Path
from typing import Mapping, Any

from pypet_rebuild.environment import Environment
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.storage import HDF5StorageService
from pypet_rebuild.merge import merge_trajectories


def multiply(traj: Trajectory) -> Mapping[str, Any]:
    z = traj.parameters["x"].value * traj.parameters["y"].value
    traj.add_result(Result(name="z", value=z))
    return {"z": z}


def main() -> None:
    # Shared output file
    file_path = Path("examples/output") / "example_03.h5"

    # Create two environments writing to the same HDF5 file
    env1 = Environment(trajectory=Trajectory(name="Traj1"), storage=HDF5StorageService(file_path=file_path))
    env2 = Environment(trajectory=Trajectory(name="Traj2"), storage=HDF5StorageService(file_path=file_path))

    traj1 = env1.trajectory
    traj2 = env2.trajectory

    # Add parameters to both
    for t in (traj1, traj2):
        t.add_parameter(Parameter(name="x", value=1.0))
        t.add_parameter(Parameter(name="y", value=1.0))

    # Explore different (overlapping) spaces
    env1.run_exploration(multiply, space={"x": [1.0, 2.0, 3.0, 4.0], "y": [6.0, 7.0, 8.0]})
    env2.run_exploration(multiply, space={"x": [3.0, 4.0, 5.0, 6.0], "y": [7.0, 8.0, 9.0]})

    # Merge traj2 into traj1, removing duplicate parameter snapshots
    merge_trajectories(traj1, traj2, remove_duplicates=True)

    # Save merged trajectory
    env1.storage.save(traj1)  # type: ignore[union-attr]

    # Load back and print unique runs
    storage = HDF5StorageService(file_path=file_path)
    merged = storage.load("Traj1")

    print("Merged run count:", len(merged.list_runs()))
    for rid in merged.list_runs()[:5]:  # print first few
        params = merged.get_run_params(rid)
        results = merged.get_run_results(rid)
        print(f"{rid}: x={params.get('x')}, y={params.get('y')}, z={results.get('z')}")


if __name__ == "__main__":
    main()
