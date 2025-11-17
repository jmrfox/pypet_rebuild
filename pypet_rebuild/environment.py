"""Execution environment for simulations using pypet_rebuild.

The Environment coordinates a Trajectory and a StorageService, and provides a
simple interface for running user-defined simulation functions. The design is
intentionally minimal at this stage; features like parameter exploration,
parallelism, and resuming runs will be added iteratively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

from .exploration import cartesian_product
from .storage import StorageService
from .parameters import Result
from .trajectory import Trajectory


SimulationFunction = Callable[[Trajectory], None]


@dataclass
class Environment:
    """A minimal execution environment for simulations.

    Parameters
    ----------
    trajectory:
        The trajectory object that holds parameters, results, and metadata
        associated with this environment.
    storage:
        A storage backend responsible for persisting trajectories. This may be
        ``None`` in early usage patterns where persistence is not yet needed.
    """

    trajectory: Trajectory
    storage: StorageService | None = None

    def run(self, func: SimulationFunction) -> None:
        """Run a single simulation function against the current trajectory.

        Future versions will handle parameter exploration, iteration over
        multiple combinations of parameters, parallel execution, error
        handling, and resuming capabilities.
        """

        func(self.trajectory)

        if self.storage is not None:
            self.storage.save(self.trajectory)

    def run_exploration_parallel(
        self,
        func: SimulationFunction,
        space: Mapping[str, Sequence[Any]],
        _max_workers: int | None = None,
    ) -> None:
        """Run exploration in parallel using a thread pool.

        Notes
        -----
        - For thread safety, simulations run against temporary Trajectory
          instances and return result mappings which are merged back into the
          main trajectory. If the function does not return a mapping, any
          results written to the temporary trajectory will be collected and
          merged.
        - We use threads initially to avoid pickling constraints on Windows.
          A process-based executor can be added later with a stricter contract.
        """

        combos = list(cartesian_product(space))
        base_name = self.trajectory.name
        # Snapshot baseline parameters from the main trajectory so that workers
        # inherit defaults (e.g., values not explicitly varied in the space).
        baseline_params: dict[str, Any] = {
            name: param.value for name, param in self.trajectory.parameters.items()
        }

        def _worker(combo: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
            local = Trajectory(name=base_name)
            # Apply baseline defaults first, then override with the combo.
            local.set_parameter_values(baseline_params)
            local.set_parameter_values(combo)
            before = set(local.results.keys())
            ret = func(local)
            if isinstance(ret, Mapping):
                results_map = dict(ret)
            else:
                after = set(local.results.keys())
                new_keys = after - before
                results_map = {k: local.results[k].value for k in new_keys}
            return combo, results_map

        with ThreadPoolExecutor(max_workers=_max_workers) as ex:
            futures = [ex.submit(_worker, combo) for combo in combos]
            for idx, fut in enumerate(futures):
                params_map, results_map = fut.result()
                run_id = f"{idx:05d}"
                # Mirror results into main trajectory for convenience
                for name, value in results_map.items():
                    self.trajectory.add_result(Result(name=name, value=value))
                self.trajectory.record_run(run_id, params_map, results_map)

        if self.storage is not None:
            self.storage.save(self.trajectory)

    def run_exploration(
        self,
        func: SimulationFunction,
        space: Mapping[str, Sequence[Any]],
    ) -> None:
        """Run a simulation function over an explored parameter space.

        Parameters
        ----------
        func:
            Simulation function that consumes a :class:`Trajectory` and records
            results on it.
        space:
            Mapping from fully-qualified parameter names to sequences of values
            to be combined via a cartesian product.
        """

        for idx, combo in enumerate(cartesian_product(space)):
            # Apply parameter combination
            self.trajectory.set_parameter_values(combo)

            # Track existing results to compute delta if the function does not return a mapping
            before_keys = set(self.trajectory.results.keys())

            ret = func(self.trajectory)

            # Determine results for run record
            results_map: dict[str, Any]
            if isinstance(ret, Mapping):
                results_map = dict(ret)
                # Also mirror into trajectory results directly for convenience
                for name, value in results_map.items():
                    self.trajectory.add_result(
                        Result(name=name, value=value)
                    )
            else:
                after_keys = set(self.trajectory.results.keys())
                new_keys = after_keys - before_keys
                results_map = {k: self.trajectory.results[k].value for k in new_keys}

            # Record run snapshot and mirror results under by_run namespace
            run_id = f"{idx:05d}"
            self.trajectory.record_run(run_id, combo, results_map)

        if self.storage is not None:
            self.storage.save(self.trajectory)
