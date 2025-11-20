from pathlib import Path
import numpy as np

from pypet_rebuild.environment import Environment
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.storage import HDF5StorageService


def simulate_diff(traj: Trajectory):
    p = traj.parameters
    diff_name = str(p["diff_name"].value)
    dt = float(p["dt"].value)
    steps = int(p["steps"].value)
    ic = np.asarray(p["initial_conditions"].value, dtype=float)

    if diff_name == "diff_lorenz":
        sigma = float(p["func_params.sigma"].value)
        beta = float(p["func_params.beta"].value)
        rho = float(p["func_params.rho"].value)

        def f(v: np.ndarray) -> np.ndarray:
            return np.array([
                sigma * (v[1] - v[0]),
                v[0] * (rho - v[2]) - v[1],
                v[0] * v[1] - beta * v[2],
            ], dtype=float)

    elif diff_name == "diff_roessler":
        a = float(p["func_params.a"].value)
        c = float(p["func_params.c"].value)
        b = a

        def f(v: np.ndarray) -> np.ndarray:
            return np.array([
                -v[1] - v[2],
                v[0] + a * v[1],
                b + v[2] * (v[0] - c),
            ], dtype=float)
    else:
        raise ValueError(f"Unknown diff_name: {diff_name}")

    path = np.zeros((steps, 3), dtype=float)
    path[0] = ic
    for i in range(1, steps):
        path[i] = path[i - 1] + dt * f(path[i - 1])

    traj.add_result(Result(name="euler.path", value=path))
    return {"euler.path": path}


def test_presetting_roessler_branch(tmp_path):
    file_path = Path(tmp_path) / "example_06_roessler.h5"

    traj = Trajectory(name="Ex06_Roessler")
    env = Environment(trajectory=traj, storage=HDF5StorageService(file_path=file_path))

    # Shared params
    traj.add_parameter(Parameter(name="steps", value=200))
    traj.add_parameter(Parameter(name="dt", value=0.01))
    traj.add_parameter(Parameter(name="initial_conditions", value=np.array([0.0, 0.0, 0.0])))

    # Preset to Roessler
    traj.add_parameter(Parameter(name="diff_name", value="diff_roessler"))
    traj.add_parameter(Parameter(name="func_params.a", value=0.1))
    traj.add_parameter(Parameter(name="func_params.c", value=14.0))

    ics = [
        np.array([0.01, 0.01, 0.01], dtype=float),
        np.array([2.02, 0.02, 0.02], dtype=float),
    ]

    env.run_exploration(
        simulate_diff,
        space={
            "initial_conditions": ics,
        },
    )

    assert len(traj.list_runs()) == 2

    storage = HDF5StorageService(file_path=file_path)
    loaded = storage.load("Ex06_Roessler")
    assert len(loaded.list_runs()) == 2

    for rid in loaded.list_runs():
        params = loaded.get_run_params(rid)
        res = loaded.get_run_results(rid)
        path = res.get("euler.path")
        assert isinstance(path, np.ndarray)
        assert path.shape[1] == 3
        ic_saved = params.get("initial_conditions")
        assert np.allclose(path[0], ic_saved)


def test_presetting_lorenz_branch(tmp_path):
    file_path = Path(tmp_path) / "example_06_lorenz.h5"

    traj = Trajectory(name="Ex06_Lorenz")
    env = Environment(trajectory=traj, storage=HDF5StorageService(file_path=file_path))

    # Shared params
    traj.add_parameter(Parameter(name="steps", value=150))
    traj.add_parameter(Parameter(name="dt", value=0.01))
    traj.add_parameter(Parameter(name="initial_conditions", value=np.array([0.0, 0.0, 0.0])))

    # Preset to Lorenz
    traj.add_parameter(Parameter(name="diff_name", value="diff_lorenz"))
    traj.add_parameter(Parameter(name="func_params.sigma", value=10.0))
    traj.add_parameter(Parameter(name="func_params.beta", value=8.0/3.0))
    traj.add_parameter(Parameter(name="func_params.rho", value=28.0))

    ics = [
        np.array([0.1, 0.0, 0.0], dtype=float),
        np.array([0.2, 0.0, 0.0], dtype=float),
        np.array([0.3, 0.0, 0.0], dtype=float),
    ]

    env.run_exploration(
        simulate_diff,
        space={
            "initial_conditions": ics,
        },
    )

    assert len(traj.list_runs()) == 3

    storage = HDF5StorageService(file_path=file_path)
    loaded = storage.load("Ex06_Lorenz")
    assert len(loaded.list_runs()) == 3

    for rid in loaded.list_runs():
        params = loaded.get_run_params(rid)
        res = loaded.get_run_results(rid)
        path = res.get("euler.path")
        assert isinstance(path, np.ndarray)
        assert path.shape[1] == 3
        ic_saved = params.get("initial_conditions")
        assert np.allclose(path[0], ic_saved)
