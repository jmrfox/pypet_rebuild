from pathlib import Path

from pypet_rebuild.environment import Environment
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.storage import HDF5StorageService
from pypet_rebuild.exploration import cartesian_product


def multiply(traj: Trajectory):
    x = traj.parameters["x"].value
    y = traj.parameters["y"].value
    z = x * y
    # Mirror both ways: add result and return mapping
    traj.add_result(Result(name="z", value=z))
    return {"z": z}


def main() -> None:
    file_path = Path("examples/output") / "example_01.h5"

    env = Environment(
        trajectory=Trajectory(name="Multiplication"),
        storage=HDF5StorageService(file_path=file_path),
    )

    traj = env.trajectory
    traj.add_parameter(Parameter(name="x", value=1))
    traj.add_parameter(Parameter(name="y", value=1))

    space = {"x": [1, 2, 3, 4], "y": [6, 7, 8]}
    env.run_exploration(multiply, space)

    # Reload fresh and print one run result
    loaded = env.storage.load("Multiplication")  # type: ignore[union-attr]
    rid = "00001"  # 2nd run
    print("The result of run", rid, "is:", loaded.get_run_results(rid).get("z"))


if __name__ == "__main__":
    main()
