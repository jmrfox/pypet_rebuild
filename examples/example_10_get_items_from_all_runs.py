from pathlib import Path
import numpy as np

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
    file_path = Path("examples/output") / "example_10.h5"
    env = Environment(
        trajectory=Trajectory(name="Example10"),
        storage=HDF5StorageService(file_path=file_path),
    )

    traj = env.trajectory
    traj.add_parameter(Parameter(name="x", value=0))
    traj.add_parameter(Parameter(name="y", value=0))

    x_len = 6
    y_len = 5
    space = {"x": list(range(x_len)), "y": list(range(y_len))}

    env.run_exploration(multiply, space)

    # Collect results across runs in insertion order
    zs = traj.collect_runs("z")

    # Reshape to a 2D mesh (x major, y minor)
    z_mesh = np.reshape(np.array(zs), (x_len, y_len))
    print("z_mesh shape:", z_mesh.shape)
    print("z_mesh sample (0:3, 0:3):\n", z_mesh[:3, :3])


if __name__ == "__main__":
    main()
