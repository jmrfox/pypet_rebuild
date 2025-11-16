"""Execution environment for simulations using pypet_rebuild.

The Environment coordinates a Trajectory and a StorageService, and provides a
simple interface for running user-defined simulation functions. The design is
intentionally minimal at this stage; features like parameter exploration,
parallelism, and resuming runs will be added iteratively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .exploration import cartesian_product
from .storage import StorageService
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
        """Placeholder API for parallel exploration.

        At this stage, this method simply delegates to :meth:`run_exploration`
        and ignores ``max_workers``. The intent is to stabilize the public
        interface so that we can later plug in a concrete parallel execution
        strategy (e.g. ``concurrent.futures`` or an external executor) without
        breaking user code.
        """

        # For now, run sequentially. Parallelism will be introduced in a
        # future iteration once the execution and error-handling model is
        # fully specified.
        self.run_exploration(func, space)

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

        for combo in cartesian_product(space):
            self.trajectory.set_parameter_values(combo)
            func(self.trajectory)

        if self.storage is not None:
            self.storage.save(self.trajectory)
