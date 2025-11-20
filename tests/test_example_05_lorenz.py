from pathlib import Path
import numpy as np

from pypet_rebuild.environment import Environment
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.storage import HDF5StorageService


def lorenz_euler(x0, y0, z0, sigma, beta, rho, dt, steps):
    path = np.zeros((steps, 3), dtype=float)
    path[0] = [x0, y0, z0]
    for i in range(1, steps):
        x, y, z = path[i - 1]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        path[i] = path[i - 1] + dt * np.array([dx, dy, dz], dtype=float)
    return path


def simulate_lorenz(traj: Trajectory):
    p = traj.parameters
    x0 = float(p["x0"].value)
    y0 = float(p["y0"].value)
    z0 = float(p["z0"].value)
    sigma = float(p["sigma"].value)
    beta = float(p["beta"].value)
    rho = float(p["rho"].value)
    dt = float(p["dt"].value)
    steps = int(p["steps"].value)
    path = lorenz_euler(x0, y0, z0, sigma, beta, rho, dt, steps)
    traj.add_result(Result(name="lorenz.path", value=path))
    return {"lorenz.path": path}


def test_example_05_lorenz_store_and_reload(tmp_path):
    file_path = Path(tmp_path) / "example_05.h5"

    traj = Trajectory(name="Lorenz")
    env = Environment(trajectory=traj, storage=HDF5StorageService(file_path=file_path))

    # base params
    traj.add_parameter(Parameter(name="sigma", value=10.0))
    traj.add_parameter(Parameter(name="beta", value=8.0 / 3.0))
    traj.add_parameter(Parameter(name="rho", value=28.0))
    traj.add_parameter(Parameter(name="dt", value=0.01))
    traj.add_parameter(Parameter(name="steps", value=500))
    traj.add_parameter(Parameter(name="x0", value=0.1))
    traj.add_parameter(Parameter(name="y0", value=0.0))
    traj.add_parameter(Parameter(name="z0", value=0.0))

    env.run_exploration(
        simulate_lorenz,
        space={
            "rho": [28.0, 35.0],
            "x0": [0.1, 0.2],
        },
    )

    assert len(traj.list_runs()) == 4

    # Reload and validate
    storage = HDF5StorageService(file_path=file_path)
    loaded = storage.load("Lorenz")
    assert len(loaded.list_runs()) == 4

    for rid in loaded.list_runs():
        params = loaded.get_run_params(rid)
        res = loaded.get_run_results(rid)
        path = res.get("lorenz.path")
        assert isinstance(path, np.ndarray)
        assert path.shape[1] == 3
        # first row matches initial conditions
        assert np.allclose(path[0], [params.get("x0"), params.get("y0"), params.get("z0")])
        # values are finite
        assert np.isfinite(path).all()
