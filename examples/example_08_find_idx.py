from pathlib import Path

from pypet_rebuild.environment import Environment
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.storage import HDF5StorageService


def multiply(traj: Trajectory):
    x = int(traj.parameters["x"].value)
    y = int(traj.parameters["y"].value)
    z = x * y
    traj.add_result(Result(name="z", value=z))
    return {"z": z}


def main() -> None:
    file_path = Path("examples/output") / "example_08.h5"
    env = Environment(
        trajectory=Trajectory(name="Example08"),
        storage=HDF5StorageService(file_path=file_path),
    )
    traj = env.trajectory

    traj.add_parameter(Parameter(name="x", value=1))
    traj.add_parameter(Parameter(name="y", value=1))

    space = {"x": [1, 2, 3, 4], "y": [6, 7, 8]}
    env.run_exploration(multiply, space)

    # Find runs where x == 2 or y == 8
    matched = traj.find_runs(lambda x, y: (x == 2) or (y == 8), names=["x", "y"])

    print("Runs with x==2 or y==8:")
    for rid in matched:
        params = traj.get_run_params(rid)
        results = traj.get_run_results(rid)
        print(f"{rid}: x={params.get('x')}, y={params.get('y')}, z={results.get('z')}")


if __name__ == "__main__":
    main()
