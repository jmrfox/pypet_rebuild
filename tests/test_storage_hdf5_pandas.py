"""Tests for pandas Series/DataFrame round-trips via HDF5StorageService."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pypet_rebuild import Parameter, Result, Trajectory
from pypet_rebuild.storage import HDF5StorageService


def test_hdf5_storage_round_trip_pandas(tmp_path) -> None:  # type: ignore[no-untyped-def]
    file_path = Path(tmp_path) / "traj_pandas.h5"
    storage = HDF5StorageService(file_path=file_path)

    original = Trajectory(name="pandas")

    series = pd.Series([1, 2, 3], index=["a", "b", "c"], dtype=float)
    frame = pd.DataFrame({"x": np.arange(3), "y": np.array([10.0, 20.0, 30.0])})

    original.add_parameter(Parameter(name="series_param", value=series))
    original.add_result(Result(name="frame_result", value=frame))

    storage.save(original)

    loaded = storage.load("pandas")

    assert "series_param" in loaded.parameters
    assert "frame_result" in loaded.results

    loaded_series = loaded.parameters["series_param"].value
    loaded_frame = loaded.results["frame_result"].value

    assert isinstance(loaded_series, pd.Series)
    assert isinstance(loaded_frame, pd.DataFrame)

    pd.testing.assert_series_equal(loaded_series.sort_index(), series.sort_index())
    pd.testing.assert_frame_equal(loaded_frame.sort_index(axis=0), frame.sort_index(axis=0))
