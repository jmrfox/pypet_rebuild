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

        return traj
