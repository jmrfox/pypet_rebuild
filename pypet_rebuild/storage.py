"""Storage abstractions for pypet_rebuild.

The goal is to cleanly separate the core experiment/trajectory logic from
persistence concerns. Storage backends should implement the StorageService
protocol so that they can be swapped or extended without touching the
high-level APIs.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Protocol
import json
from io import StringIO

import h5py
import numpy as np
import pandas as pd

from .parameters import Parameter, Result
from .trajectory import Trajectory
from .constants import (
    HDF5_ROOT_GROUP,
    HDF5_PARAMETERS_GROUP,
    HDF5_RESULTS_GROUP,
)


class StorageService(Protocol):
    """Protocol for storage backends.

    Concrete implementations are responsible for persisting trajectories and
    restoring them from disk or other storage systems.
    """

    def save(self, trajectory: Trajectory) -> None:  # pragma: no cover - protocol
        """Persist the given trajectory.

        Implementations decide how to map the trajectory structure and data
        into their underlying representation.
        """

    def load(self, name: str) -> Trajectory:  # pragma: no cover - protocol
        """Load a trajectory by name.

        Implementations may use the name as a key, filename, or path within a
        larger storage hierarchy.
        """


class HDF5StorageService(ABC):
    """Base class for an HDF5-backed storage service.

    This is a placeholder for a future implementation that uses ``h5py`` or
    ``tables`` to persist trajectories to disk. For now, the methods raise
    ``NotImplementedError`` so we can design the interface and write tests
    around the expected behavior.
    """

    def __init__(self, file_path: Path) -> None:
        self._file_path = Path(file_path)

    @property
    def file_path(self) -> Path:
        """Location of the underlying HDF5 file."""

        return self._file_path

    # Minimal, concrete implementation ---------------------------------

    def save(self, trajectory: Trajectory) -> None:
        """Persist the given trajectory into an HDF5 file.

        The current layout is intentionally simple:

        - ``/trajectories/<name>/parameters/<param_name>``
        - ``/trajectories/<name>/results/<result_name>``

        Each leaf is a group with attributes describing the stored value.

        For now we support several storage modes:

        - ``kind = "json"``: ``value`` attribute contains a JSON-encoded
          representation (for basic scalars and small containers).
        - ``kind = "ndarray"``: a dataset named ``"data"`` stores the NumPy
          array and the ``dtype``/``shape`` are taken from the array itself.
        - ``kind = "pandas_series"``: ``value`` attribute holds ``Series.to_json``.
        - ``kind = "pandas_frame"``: ``value`` attribute holds ``DataFrame.to_json``.
        """

        file_path = self._file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(file_path, "a") as h5:
            root = h5.require_group(HDF5_ROOT_GROUP)

            if trajectory.name in root:
                del root[trajectory.name]

            traj_group = root.create_group(trajectory.name)
            params_group = traj_group.create_group(HDF5_PARAMETERS_GROUP)
            results_group = traj_group.create_group(HDF5_RESULTS_GROUP)
            runs_group = traj_group.create_group("runs")

            for name, param in trajectory.parameters.items():
                g = params_group.create_group(name)
                value = param.value
                if isinstance(value, np.ndarray):
                    g.attrs["kind"] = "ndarray"
                    g.create_dataset("data", data=value)
                elif isinstance(value, pd.Series):
                    g.attrs["kind"] = "pandas_series"
                    g.attrs["value"] = value.to_json(orient="split")
                    g.attrs["pandas_dtype"] = str(value.dtype)
                elif isinstance(value, pd.DataFrame):
                    g.attrs["kind"] = "pandas_frame"
                    g.attrs["value"] = value.to_json(orient="split")
                    g.attrs["pandas_dtypes"] = json.dumps(
                        {col: str(dt) for col, dt in value.dtypes.items()}
                    )
                else:
                    g.attrs["kind"] = "json"
                    g.attrs["value"] = json.dumps(value)

                if param.comment is not None:
                    g.attrs["comment"] = param.comment

            for name, result in trajectory.results.items():
                g = results_group.create_group(name)
                value = result.value
                if isinstance(value, np.ndarray):
                    g.attrs["kind"] = "ndarray"
                    g.create_dataset("data", data=value)
                elif isinstance(value, pd.Series):
                    g.attrs["kind"] = "pandas_series"
                    g.attrs["value"] = value.to_json(orient="split")
                    g.attrs["pandas_dtype"] = str(value.dtype)
                elif isinstance(value, pd.DataFrame):
                    g.attrs["kind"] = "pandas_frame"
                    g.attrs["value"] = value.to_json(orient="split")
                    g.attrs["pandas_dtypes"] = json.dumps(
                        {col: str(dt) for col, dt in value.dtypes.items()}
                    )
                else:
                    g.attrs["kind"] = "json"
                    g.attrs["value"] = json.dumps(value)

                if result.comment is not None:
                    g.attrs["comment"] = result.comment

            # Persist run records (parameters snapshot + timestamp). We do not duplicate
            # per-run result values since these are mirrored under results/by_run.*
            for rec in getattr(trajectory, "_run_records", []):
                run_id = str(rec.get("id", ""))
                rg = runs_group.create_group(run_id)
                rg.attrs["params"] = json.dumps(rec.get("params", {}))
                ts = rec.get("timestamp")
                if ts is not None:
                    rg.attrs["timestamp"] = ts

    def load(self, name: str) -> Trajectory:
        """Load a trajectory by name from the HDF5 file.

        Parameters and results are reconstructed from the JSON-encoded
        attributes stored by :meth:`save`.
        """

        with h5py.File(self._file_path, "r") as h5:
            root = h5[HDF5_ROOT_GROUP]
            traj_group = root[name]

            traj = Trajectory(name=name)

            params_group = traj_group.get(HDF5_PARAMETERS_GROUP)
            if params_group is not None:
                for param_name, g in params_group.items():
                    kind = g.attrs.get("kind", "json")
                    if kind == "ndarray":
                        value = np.array(g["data"][...])
                    elif kind == "pandas_series":
                        raw_json = g.attrs["value"]
                        if isinstance(raw_json, bytes):
                            raw_json = raw_json.decode("utf-8")
                        value = pd.read_json(StringIO(raw_json), typ="series", orient="split")
                        dtype_attr = g.attrs.get("pandas_dtype")
                        if isinstance(dtype_attr, bytes):
                            dtype_attr = dtype_attr.decode("utf-8")
                        if dtype_attr:
                            value = value.astype(dtype_attr)  # type: ignore[arg-type]
                    elif kind == "pandas_frame":
                        raw_json = g.attrs["value"]
                        if isinstance(raw_json, bytes):
                            raw_json = raw_json.decode("utf-8")
                        value = pd.read_json(StringIO(raw_json), orient="split")
                        dtypes_json = g.attrs.get("pandas_dtypes")
                        if isinstance(dtypes_json, bytes):
                            dtypes_json = dtypes_json.decode("utf-8")
                        if dtypes_json:
                            try:
                                dtypes_map = json.loads(dtypes_json)
                                value = value.astype(dtypes_map)
                            except (TypeError, ValueError, json.JSONDecodeError):
                                pass
                    else:
                        raw = g.attrs["value"]
                        try:
                            value = json.loads(raw)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            value = raw

                    comment = g.attrs.get("comment")
                    if isinstance(comment, bytes):
                        comment = comment.decode("utf-8")

                    traj.add_parameter(
                        Parameter(
                            name=param_name,
                            value=value,
                            comment=comment,
                        )
                    )

            results_group = traj_group.get(HDF5_RESULTS_GROUP)
            if results_group is not None:
                for result_name, g in results_group.items():
                    kind = g.attrs.get("kind", "json")
                    if kind == "ndarray":
                        value = np.array(g["data"][...])
                    elif kind == "pandas_series":
                        raw_json = g.attrs["value"]
                        if isinstance(raw_json, bytes):
                            raw_json = raw_json.decode("utf-8")
                        value = pd.read_json(StringIO(raw_json), typ="series", orient="split")
                        dtype_attr = g.attrs.get("pandas_dtype")
                        if isinstance(dtype_attr, bytes):
                            dtype_attr = dtype_attr.decode("utf-8")
                        if dtype_attr:
                            value = value.astype(dtype_attr)  # type: ignore[arg-type]
                    elif kind == "pandas_frame":
                        raw_json = g.attrs["value"]
                        if isinstance(raw_json, bytes):
                            raw_json = raw_json.decode("utf-8")
                        value = pd.read_json(StringIO(raw_json), orient="split")
                        dtypes_json = g.attrs.get("pandas_dtypes")
                        if isinstance(dtypes_json, bytes):
                            dtypes_json = dtypes_json.decode("utf-8")
                        if dtypes_json:
                            try:
                                dtypes_map = json.loads(dtypes_json)
                                value = value.astype(dtypes_map)
                            except (TypeError, ValueError, json.JSONDecodeError):
                                pass
                    else:
                        raw = g.attrs["value"]
                        try:
                            value = json.loads(raw)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            value = raw

                    comment = g.attrs.get("comment")
                    if isinstance(comment, bytes):
                        comment = comment.decode("utf-8")

                    traj.add_result(
                        Result(
                            name=result_name,
                            value=value,
                            comment=comment,
                        )
                    )

            # Reconstruct run records from 'runs' group and by_run mirrors
            runs_group = traj_group.get("runs")
            if runs_group is not None:
                # Build an index of by_run values once for efficiency
                by_run_index: dict[str, dict[str, object]] = {}
                for res_name, res in traj._results.items():  # type: ignore[attr-defined]
                    if res_name.startswith("by_run."):
                        parts = res_name.split(".", 2)
                        if len(parts) == 3:
                            _, rid, leaf = parts
                            by_run_index.setdefault(rid, {})[leaf] = res.value

                for run_id, rg in runs_group.items():
                    params_json = rg.attrs.get("params", "{}")
                    if isinstance(params_json, bytes):
                        params_json = params_json.decode("utf-8")
                    try:
                        params_map = json.loads(params_json) if params_json else {}
                    except (TypeError, ValueError, json.JSONDecodeError):
                        params_map = {}
                    timestamp = rg.attrs.get("timestamp")
                    if isinstance(timestamp, bytes):
                        timestamp = timestamp.decode("utf-8")

                    results_map = by_run_index.get(run_id, {})
                    # Append without re-mirroring results
                    traj._run_records.append(  # type: ignore[attr-defined]
                        {
                            "id": run_id,
                            "params": dict(params_map),
                            "results": dict(results_map),
                            "timestamp": timestamp,
                        }
                    )

        return traj

    # Dynamic loading (ndarray slices) ---------------------------------

    def load_param_array_slice(self, traj_name: str, param_name: str, index):
        with h5py.File(self._file_path, "r") as h5:
            root = h5[HDF5_ROOT_GROUP]
            traj_group = root[traj_name]
            g = traj_group[HDF5_PARAMETERS_GROUP][param_name]
            kind = g.attrs.get("kind", "json")
            if kind != "ndarray":
                raise TypeError(f"Parameter '{param_name}' is not stored as ndarray")
            return np.array(g["data"][index])

    def load_result_array_slice(self, traj_name: str, result_name: str, index):
        with h5py.File(self._file_path, "r") as h5:
            root = h5[HDF5_ROOT_GROUP]
            traj_group = root[traj_name]
            g = traj_group[HDF5_RESULTS_GROUP][result_name]
            kind = g.attrs.get("kind", "json")
            if kind != "ndarray":
                raise TypeError(f"Result '{result_name}' is not stored as ndarray")
            return np.array(g["data"][index])

    # Partial loading APIs ---------------------------------------------

    def load_partial(
        self,
        name: str,
        *,
        load_parameters: int = 2,
        load_results: int = 2,
        load_only: list[str] | None = None,
    ) -> Trajectory:
        """Partially load a trajectory with skeleton/data flags.

        load_parameters / load_results semantics:
        - 0: skip
        - 1: skeleton (create items with value=None)
        - 2: load data (default)
        If load_only is provided, it filters which results are loaded/skeletonized.
        """

        with h5py.File(self._file_path, "r") as h5:
            root = h5[HDF5_ROOT_GROUP]
            traj_group = root[name]
            traj = Trajectory(name=name)

            # Parameters
            params_group = traj_group.get(HDF5_PARAMETERS_GROUP)
            if params_group is not None and load_parameters > 0:
                for param_name, g in params_group.items():
                    if load_parameters == 1:
                        traj.add_parameter(Parameter(name=param_name, value=None))  # type: ignore[arg-type]
                        continue
                    # load_parameters == 2
                    kind = g.attrs.get("kind", "json")
                    if kind == "ndarray":
                        value = np.array(g["data"][...])
                    elif kind == "pandas_series":
                        raw_json = g.attrs["value"]
                        if isinstance(raw_json, bytes):
                            raw_json = raw_json.decode("utf-8")
                        value = pd.read_json(StringIO(raw_json), typ="series", orient="split")
                        dtype_attr = g.attrs.get("pandas_dtype")
                        if isinstance(dtype_attr, bytes):
                            dtype_attr = dtype_attr.decode("utf-8")
                        if dtype_attr:
                            value = value.astype(dtype_attr)  # type: ignore[arg-type]
                    elif kind == "pandas_frame":
                        raw_json = g.attrs["value"]
                        if isinstance(raw_json, bytes):
                            raw_json = raw_json.decode("utf-8")
                        value = pd.read_json(StringIO(raw_json), orient="split")
                        dtypes_json = g.attrs.get("pandas_dtypes")
                        if isinstance(dtypes_json, bytes):
                            dtypes_json = dtypes_json.decode("utf-8")
                        if dtypes_json:
                            try:
                                dtypes_map = json.loads(dtypes_json)
                                value = value.astype(dtypes_map)
                            except (TypeError, ValueError, json.JSONDecodeError):
                                pass
                    else:
                        raw = g.attrs["value"]
                        try:
                            value = json.loads(raw)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            value = raw
                    traj.add_parameter(Parameter(name=param_name, value=value))

            # Results
            results_group = traj_group.get(HDF5_RESULTS_GROUP)
            if results_group is not None and load_results > 0:
                for result_name, g in results_group.items():
                    if load_only is not None and result_name not in load_only:
                        # skeleton or skip
                        if load_results == 1:
                            traj.add_result(Result(name=result_name, value=None))
                        continue
                    if load_results == 1:
                        traj.add_result(Result(name=result_name, value=None))
                        continue
                    # load_results == 2
                    kind = g.attrs.get("kind", "json")
                    if kind == "ndarray":
                        value = np.array(g["data"][...])
                    elif kind == "pandas_series":
                        raw_json = g.attrs["value"]
                        if isinstance(raw_json, bytes):
                            raw_json = raw_json.decode("utf-8")
                        value = pd.read_json(StringIO(raw_json), typ="series", orient="split")
                        dtype_attr = g.attrs.get("pandas_dtype")
                        if isinstance(dtype_attr, bytes):
                            dtype_attr = dtype_attr.decode("utf-8")
                        if dtype_attr:
                            value = value.astype(dtype_attr)  # type: ignore[arg-type]
                    elif kind == "pandas_frame":
                        raw_json = g.attrs["value"]
                        if isinstance(raw_json, bytes):
                            raw_json = raw_json.decode("utf-8")
                        value = pd.read_json(StringIO(raw_json), orient="split")
                        dtypes_json = g.attrs.get("pandas_dtypes")
                        if isinstance(dtypes_json, bytes):
                            dtypes_json = dtypes_json.decode("utf-8")
                        if dtypes_json:
                            try:
                                dtypes_map = json.loads(dtypes_json)
                                value = value.astype(dtypes_map)
                            except (TypeError, ValueError, json.JSONDecodeError):
                                pass
                    else:
                        raw = g.attrs["value"]
                        try:
                            value = json.loads(raw)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            value = raw
                    traj.add_result(Result(name=result_name, value=value))

            # Rebuild run records too (same as load)
            runs_group = traj_group.get("runs")
            if runs_group is not None:
                by_run_index: dict[str, dict[str, object]] = {}
                for res_name, res in traj._results.items():  # type: ignore[attr-defined]
                    if res_name.startswith("by_run."):
                        parts = res_name.split(".", 2)
                        if len(parts) == 3:
                            _, rid, leaf = parts
                            by_run_index.setdefault(rid, {})[leaf] = res.value
                for run_id, rg in runs_group.items():
                    params_json = rg.attrs.get("params", "{}")
                    if isinstance(params_json, bytes):
                        params_json = params_json.decode("utf-8")
                    try:
                        params_map = json.loads(params_json) if params_json else {}
                    except (TypeError, ValueError, json.JSONDecodeError):
                        params_map = {}
                    timestamp = rg.attrs.get("timestamp")
                    if isinstance(timestamp, bytes):
                        timestamp = timestamp.decode("utf-8")
                    results_map = by_run_index.get(run_id, {})
                    traj._run_records.append(  # type: ignore[attr-defined]
                        {
                            "id": run_id,
                            "params": dict(params_map),
                            "results": dict(results_map),
                            "timestamp": timestamp,
                        }
                    )

        return traj

    def load_result_frame_slice(
        self,
        traj_name: str,
        result_name: str,
        rows: slice | list[int] | None = None,
        cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load a pandas DataFrame result and optionally subset rows/columns.

        Note: This currently reads the full JSON and slices in-memory.
        """

        with h5py.File(self._file_path, "r") as h5:
            root = h5[HDF5_ROOT_GROUP]
            traj_group = root[traj_name]
            g = traj_group[HDF5_RESULTS_GROUP][result_name]
            kind = g.attrs.get("kind", "json")
            if kind != "pandas_frame":
                raise TypeError(f"Result '{result_name}' is not stored as pandas_frame")
            raw_json = g.attrs["value"]
            if isinstance(raw_json, bytes):
                raw_json = raw_json.decode("utf-8")
            df = pd.read_json(StringIO(raw_json), orient="split")
            dtypes_json = g.attrs.get("pandas_dtypes")
            if isinstance(dtypes_json, bytes):
                dtypes_json = dtypes_json.decode("utf-8")
            if dtypes_json:
                try:
                    dtypes_map = json.loads(dtypes_json)
                    df = df.astype(dtypes_map)
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
            if cols is not None:
                df = df[cols]
            if rows is not None:
                df = df.iloc[rows]
            return df

    # Per-item store APIs ----------------------------------------------

    def store_parameter(self, trajectory: Trajectory, name: str) -> None:
        """Persist a single parameter from a trajectory.

        Creates the HDF5 groups as needed. Overwrites existing datasets/attrs for the item.
        """

        value = trajectory.parameters[name].value
        with h5py.File(self._file_path, "a") as h5:
            root = h5.require_group(HDF5_ROOT_GROUP)
            traj_group = root.require_group(trajectory.name)
            params_group = traj_group.require_group(HDF5_PARAMETERS_GROUP)
            g = params_group.require_group(name)

            # Clean previous content
            if "data" in g:
                del g["data"]
            for k in ["value", "pandas_dtype", "pandas_dtypes", "kind", "comment"]:
                if k in g.attrs:
                    del g.attrs[k]

            if isinstance(value, np.ndarray):
                g.attrs["kind"] = "ndarray"
                g.create_dataset("data", data=value)
            elif isinstance(value, pd.Series):
                g.attrs["kind"] = "pandas_series"
                g.attrs["value"] = value.to_json(orient="split")
                g.attrs["pandas_dtype"] = str(value.dtype)
            elif isinstance(value, pd.DataFrame):
                g.attrs["kind"] = "pandas_frame"
                g.attrs["value"] = value.to_json(orient="split")
                g.attrs["pandas_dtypes"] = json.dumps({col: str(dt) for col, dt in value.dtypes.items()})
            else:
                g.attrs["kind"] = "json"
                g.attrs["value"] = json.dumps(value)

            param = trajectory.parameters[name]
            if param.comment is not None:
                g.attrs["comment"] = param.comment

    def store_result(self, trajectory: Trajectory, name: str) -> None:
        """Persist a single result from a trajectory.

        Creates the HDF5 groups as needed. Overwrites existing datasets/attrs for the item.
        """

        value = trajectory.results[name].value
        with h5py.File(self._file_path, "a") as h5:
            root = h5.require_group(HDF5_ROOT_GROUP)
            traj_group = root.require_group(trajectory.name)
            results_group = traj_group.require_group(HDF5_RESULTS_GROUP)
            g = results_group.require_group(name)

            # Clean previous content
            if "data" in g:
                del g["data"]
            for k in ["value", "pandas_dtype", "pandas_dtypes", "kind", "comment"]:
                if k in g.attrs:
                    del g.attrs[k]

            if isinstance(value, np.ndarray):
                g.attrs["kind"] = "ndarray"
                g.create_dataset("data", data=value)
            elif isinstance(value, pd.Series):
                g.attrs["kind"] = "pandas_series"
                g.attrs["value"] = value.to_json(orient="split")
                g.attrs["pandas_dtype"] = str(value.dtype)
            elif isinstance(value, pd.DataFrame):
                g.attrs["kind"] = "pandas_frame"
                g.attrs["value"] = value.to_json(orient="split")
                g.attrs["pandas_dtypes"] = json.dumps({col: str(dt) for col, dt in value.dtypes.items()})
            else:
                g.attrs["kind"] = "json"
                g.attrs["value"] = json.dumps(value)

            res = trajectory.results[name]
            if res.comment is not None:
                g.attrs["comment"] = res.comment
