"""Trajectory abstraction for pypet_rebuild.

A trajectory represents a tree of parameters, results, and metadata associated
with an experiment or simulation campaign. The initial implementation here is a
simple, in-memory container designed to be easy to test and evolve, while also
supporting a lightweight form of natural naming for grouped parameters and
results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterable, MutableMapping, Callable, Sequence
from datetime import datetime, timezone

from .parameters import Parameter, Result


class _ParameterNamespace(Mapping[str, Parameter[Any]]):
    """A view over trajectory parameters that supports natural naming.

    At the root level, this behaves like a mapping of full parameter names to
    :class:`Parameter` instances. Attribute access is interpreted as either a
    nested group or a concrete parameter, depending on what is present in the
    underlying trajectory.
    """

    def __init__(self, trajectory: "Trajectory", prefix: str = "") -> None:
        self._trajectory = trajectory
        self._prefix = prefix

    # Mapping interface -------------------------------------------------

    def __iter__(self) -> Iterable[str]:
        prefix = self._prefix
        if not prefix:
            yield from self._trajectory._parameters.keys()
            return

        dot_prefix = prefix + "."
        prefix_len = len(dot_prefix)

        for full_name in self._trajectory._parameters.keys():
            if full_name == prefix:
                yield prefix
            elif full_name.startswith(dot_prefix):
                yield full_name[prefix_len:]

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

    def __getitem__(self, key: str) -> Parameter[Any]:
        if self._prefix:
            full_name = f"{self._prefix}.{key}"
        else:
            full_name = key
        return self._trajectory._parameters[full_name]

    # Natural naming ----------------------------------------------------

    def __getattr__(self, item: str) -> Any:
        """Resolve attribute access to a parameter or subgroup.

        If there is a parameter whose full name matches the constructed path,
        the corresponding :class:`Parameter` is returned. Otherwise, if there
        are parameters that share the constructed path as a prefix, a new
        namespace representing that subgroup is returned.
        """

        if item.startswith("_"):
            raise AttributeError(item)

        if self._prefix:
            full_name = f"{self._prefix}.{item}"
        else:
            full_name = item

        if full_name in self._trajectory._parameters:
            return self._trajectory._parameters[full_name]

        dot_prefix = full_name + "."
        for key in self._trajectory._parameters.keys():
            if key.startswith(dot_prefix):
                return _ParameterNamespace(self._trajectory, prefix=full_name)

        raise AttributeError(item)


class _ResultNamespace(Mapping[str, Result[Any]]):
    """A view over trajectory results that supports natural naming."""

    def __init__(self, trajectory: "Trajectory", prefix: str = "") -> None:
        self._trajectory = trajectory
        self._prefix = prefix

    def __iter__(self) -> Iterable[str]:
        prefix = self._prefix
        if not prefix:
            yield from self._trajectory._results.keys()
            return

        dot_prefix = prefix + "."
        prefix_len = len(dot_prefix)

        for full_name in self._trajectory._results.keys():
            if full_name == prefix:
                yield prefix
            elif full_name.startswith(dot_prefix):
                yield full_name[prefix_len:]

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

    def __getitem__(self, key: str) -> Result[Any]:
        if self._prefix:
            full_name = f"{self._prefix}.{key}"
        else:
            full_name = key
        return self._trajectory._results[full_name]

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)

        if self._prefix:
            full_name = f"{self._prefix}.{item}"
        else:
            full_name = item

        if full_name in self._trajectory._results:
            return self._trajectory._results[full_name]

        dot_prefix = full_name + "."
        for key in self._trajectory._results.keys():
            if key.startswith(dot_prefix):
                return _ResultNamespace(self._trajectory, prefix=full_name)

        raise AttributeError(item)


@dataclass
class Trajectory:
    """A minimal trajectory implementation with natural naming support.

    Parameters and results are stored internally in flat mappings keyed by
    their fully-qualified names (for example, ``"traffic.ncars"``). The
    :class:`_ParameterNamespace` and :class:`_ResultNamespace` views expose a
    grouped, natural-naming interface where dotted paths map to nested
    namespaces or leaf objects.
    """

    name: str
    _parameters: MutableMapping[str, Parameter[Any]] = field(default_factory=dict)
    _results: MutableMapping[str, Result[Any]] = field(default_factory=dict)
    _run_records: list[dict[str, Any]] = field(default_factory=list)

    # --- Parameters ---

    def add_parameter(self, parameter: Parameter[Any]) -> None:
        """Register a parameter with the trajectory.

        If a parameter with the same name already exists, it is currently
        overwritten. Future versions may enforce stronger constraints or
        provide explicit update semantics.
        """

        self._parameters[parameter.name] = parameter

    def set_parameter_values(self, values: Mapping[str, Any]) -> None:
        """Set or update parameter values from a simple mapping.

        Keys are interpreted as fully-qualified parameter names, for example
        ``"traffic.ncars"``. If a parameter with a given name does not yet
        exist in the trajectory, a new :class:`Parameter` is created.
        """

        for name, value in values.items():
            if name in self._parameters:
                self._parameters[name].value = value
            else:
                self._parameters[name] = Parameter(name=name, value=value)

    @property
    def parameters(self) -> Mapping[str, Parameter[Any]]:
        """View over parameters attached to this trajectory.

        The returned object behaves as a mapping and also supports attribute
        access for natural naming, for example ``traj.parameters.traffic.ncars``.
        """

        return _ParameterNamespace(self)

    # --- Results ---

    def add_result(self, result: Result[Any]) -> None:
        """Attach a result produced by a simulation run."""

        self._results[result.name] = result

    @property
    def results(self) -> Mapping[str, Result[Any]]:
        """View over results attached to this trajectory.

        The returned object behaves as a mapping and also supports attribute
        access for natural naming.
        """

        return _ResultNamespace(self)

    # --- Run grouping --------------------------------------------------

    def record_run(self, run_id: str, params: Mapping[str, Any], results: Mapping[str, Any]) -> None:
        """Record a run snapshot and mirror results under a by_run namespace.

        This method appends a compact record to an internal list for quick
        access and also adds namespaced results like
        ``by_run.<run_id>.<result_name>`` to the results mapping to support
        natural naming and HDF5 persistence.
        """

        self._run_records.append({
            "id": run_id,
            "params": dict(params),
            "results": dict(results),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

        for name, value in results.items():
            namespaced = f"by_run.{run_id}.{name}"
            self._results[namespaced] = Result(name=namespaced, value=value)

    def list_runs(self) -> list[str]:
        """Return the list of recorded run IDs in insertion order."""

        return [r["id"] for r in self._run_records]

    def get_run_params(self, run_id: str) -> Mapping[str, Any]:
        """Return the parameter snapshot for a given run ID."""

        for rec in self._run_records:
            if rec.get("id") == run_id:
                return dict(rec.get("params", {}))
        return {}

    def get_run_results(self, run_id: str) -> Mapping[str, Any]:
        """Return the results mapping for a given run ID."""

        for rec in self._run_records:
            if rec.get("id") == run_id:
                return dict(rec.get("results", {}))
        return {}

    # --- Run utilities -------------------------------------------------

    def find_runs(self, predicate: Callable[..., bool], names: Sequence[str]) -> list[str]:
        """Return run IDs where predicate over selected parameter names is True.

        Parameters
        ----------
        predicate:
            Callable receiving parameter values in the same order as `names`.
        names:
            Parameter names to extract from the recorded run parameter snapshots.
        """

        matched: list[str] = []
        for rec in self._run_records:
            params = rec.get("params", {})
            values = [params.get(n) for n in names]
            try:
                if predicate(*values):
                    matched.append(rec.get("id", ""))
            except (TypeError, ValueError):
                # If predicate raises due to arg mismatch or bad values, skip this run
                continue
        return matched

    def collect_runs(self, result_name: str) -> list[Any]:
        """Collect a result value across runs using the by_run mirror.

        Returns a list of values ordered by `list_runs()`; missing entries are skipped.
        """

        values: list[Any] = []
        for run_id in self.list_runs():
            key = f"by_run.{run_id}.{result_name}"
            if key in self._results:
                values.append(self._results[key].value)
        return values
