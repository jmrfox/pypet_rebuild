"""Tests for the minimal HDF5StorageService implementation."""

from __future__ import annotations

from pathlib import Path

from pypet_rebuild import Parameter, Result, Trajectory
from pypet_rebuild.storage import HDF5StorageService


def test_hdf5_storage_round_trip_simple_trajectory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    file_path = Path(tmp_path) / "traj.h5"
    storage = HDF5StorageService(file_path=file_path)

    original = Trajectory(name="roundtrip")
    original.add_parameter(Parameter(name="x", value=1, comment="int param"))
    original.add_parameter(Parameter(name="config.mode", value="test"))
    original.add_result(Result(name="metrics.loss", value=0.123, comment="float"))

    storage.save(original)

    loaded = storage.load("roundtrip")

    # Basic structural checks
    assert loaded.name == original.name

    # Parameters
    assert set(loaded.parameters.keys()) == {"x", "config.mode"}
    assert loaded.parameters["x"].value == 1
    assert loaded.parameters["x"].comment == "int param"
    assert loaded.parameters["config.mode"].value == "test"

    # Results
    assert set(loaded.results.keys()) == {"metrics.loss"}
    assert loaded.results["metrics.loss"].value == 0.123
    assert loaded.results["metrics.loss"].comment == "float"
