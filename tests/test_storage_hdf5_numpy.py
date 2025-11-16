"""Tests for NumPy array round-trips via HDF5StorageService."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pypet_rebuild import Parameter, Result, Trajectory
from pypet_rebuild.storage import HDF5StorageService


def test_hdf5_storage_round_trip_numpy(tmp_path) -> None:  # type: ignore[no-untyped-def]
    file_path = Path(tmp_path) / "traj_numpy.h5"
    storage = HDF5StorageService(file_path=file_path)

    original = Trajectory(name="numpy")
    array_param = np.arange(6, dtype=float).reshape(2, 3)
    original.add_parameter(Parameter(name="array_param", value=array_param))
    original.add_result(Result(name="array_result", value=array_param * 2))

    storage.save(original)

    loaded = storage.load("numpy")

    assert "array_param" in loaded.parameters
    assert "array_result" in loaded.results

    loaded_param = loaded.parameters["array_param"].value
    loaded_result = loaded.results["array_result"].value

    assert isinstance(loaded_param, np.ndarray)
    assert isinstance(loaded_result, np.ndarray)

    np.testing.assert_array_equal(loaded_param, array_param)
    np.testing.assert_array_equal(loaded_result, array_param * 2)
