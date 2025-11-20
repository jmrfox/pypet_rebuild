# pypet_rebuild

An attempt at rebuilding Robert Meyer's pypet.

The original pypet package, while fantastic, is a little outdated in terms of modern Python practices and dependencies. This is an attempt to rebuild it with modern practices in mind. This is all my own code, but credit to [Robert Meyer](https://github.com/SmokinCaterpillar) and other contributors for the original ideas and implementations.
 
---

## What is this framework?

`pypet_rebuild` is a modern Python toolkit for **parameter exploration** and **result management** in numerical simulations. It aims to preserve the core ideas of the original pypet:

- **Central trajectory abstraction** that holds parameters, results, and metadata.
- **Natural naming** for ergonomic access (e.g. `traj.parameters.traffic.ncars`).
- **Execution environment** that runs user-defined simulations over sets of parameter combinations.
- **Structured storage** of runs to disk.

The focus is on a clean, typed, and testable core that you can use for experiments, simulations, and small parameter sweeps, with room to grow into more complex workflows.

At a high level, you:

- Define a `Trajectory` and attach `Parameter` objects.
- Write a simulation function that consumes a `Trajectory` and writes back `Result` objects.
- Use an `Environment` (and helpers like `cartesian_product`) to run that simulation over many parameter combinations.
- Optionally persist everything to an HDF5 file via a `StorageService`.

---

## Current implementation status

This project is under active development and does **not** yet match the full feature set of the original pypet. So far, the following pieces are in place:

- **Core abstractions**
  - `Trajectory` with internal mappings for parameters/results and natural-naming views.
  - `Parameter[T]` and `Result[T]` as typed dataclasses.
  - `Environment` with `run` and `run_exploration` methods.

- **Exploration**
  - A small `cartesian_product` helper.
  - `Environment.run_exploration` to iterate over combinations, update the trajectory, and run a simulation function for each.

- **Storage**
  - `StorageService` protocol defining a minimal interface for backends.
  - `HDF5StorageService` implementation using `h5py`, with a simple layout:
    - `/trajectories/<name>/parameters/<param_name>`
    - `/trajectories/<name>/results/<result_name>`
  - Basic round-trip tests for JSON-serializable values.

- **Testing and tooling**
  - pytest-based test suite.
  - Project managed with `uv` and `pyproject.toml` (Python `>=3.12`).

See `TODO.md` for a more detailed roadmap and remaining work.

---

## Quick start

Below is a minimal example showing how to:

- Define a trajectory with parameters.
- Write a simulation function that computes results.
- Run the simulation over a cartesian product of parameter values.
- Optionally save the trajectory to an HDF5 file.

```python
from pathlib import Path

from pypet_rebuild import (
    Environment,
    HDF5StorageService,
    Parameter,
    Result,
    Trajectory,
    cartesian_product,
)


def simulate(traj: Trajectory) -> None:
    """Simple simulation that multiplies two parameters and records the result."""

    x = traj.parameters["x"].value
    y = traj.parameters["y"].value

    traj.add_result(Result(name=f"product.{x}_{y}", value=x * y))


def main() -> None:
    traj = Trajectory(name="multiplication")

    # Seed parameters; values will be overwritten during exploration
    traj.add_parameter(Parameter(name="x", value=0))
    traj.add_parameter(Parameter(name="y", value=0))

    storage = HDF5StorageService(file_path=Path("example.h5"))
    env = Environment(trajectory=traj, storage=storage)

    # Define the parameter space and run over all combinations
    space = {"x": [1, 2, 3], "y": [10, 20]}
    env.run_exploration(simulate, space=space)

    # After this, traj.results contains one result per combination, and
    # the data has been written to example.h5.


if __name__ == "__main__":
    main()
```

---

## Improvements over original pypet (so far)

This rebuild intentionally makes some early, concrete improvements relative to the original implementation:

- **Modern packaging and environment**
  - Uses `pyproject.toml` and `uv` instead of legacy `setup.py` workflows.
  - Targets modern Python versions (currently `>=3.12`).

- **Typed, dataclass-based domain model**
  - `Parameter[T]` and `Result[T]` are simple, generic `@dataclass` structures.
  - Clear, explicit types and signatures across the public API.

- **Clean separation of concerns**
  - `Trajectory` is responsible for in-memory parameter/result management and natural naming.
  - `Environment` coordinates execution (single run + exploration) but knows nothing about storage internals.
  - `StorageService` is a small protocol that HDF5 and future backends can implement.

- **Small, composable exploration API**
  - `cartesian_product` is a pure helper that produces combinations as dictionaries.
  - `Environment.run_exploration` consumes any mapping of parameter names to sequences; no hidden global state.

- **Simpler, explicit storage layout**
  - First-pass HDF5 backend with JSON-encoded values for basic types.
  - Clear, documented location for trajectory data in the file.

Future improvements will cover richer data type support (NumPy arrays, pandas), parallel execution, and more robust logging/resuming behavior, while keeping the design modular and testable.
