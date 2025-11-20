from pathlib import Path

import numpy as np
import pandas as pd

from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.storage import HDF5StorageService


def _make_sample_traj(name: str) -> tuple[Trajectory, np.ndarray, np.ndarray, pd.DataFrame, pd.Series]:
    t = Trajectory(name=name)
    # parameters
    t.add_parameter(Parameter(name="alpha", value=42))
    parr = np.arange(20).reshape(4, 5)
    t.add_parameter(Parameter(name="arr", value=parr))

    # results
    rarr = np.arange(12).reshape(3, 4)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    ser = pd.Series([100, 200, 300], index=["u", "v", "w"], name="s")
    t.add_result(Result(name="arr_r", value=rarr))
    t.add_result(Result(name="df", value=df))
    t.add_result(Result(name="ser", value=ser))
    return t, parr, rarr, df, ser


def test_load_partial_skeleton_and_load_only(tmp_path):
    file_path = tmp_path / "partial.h5"
    name = "partial"

    traj, parr, rarr, df, ser = _make_sample_traj(name)
    storage = HDF5StorageService(file_path=Path(file_path))
    storage.save(traj)

    # Skeleton only
    t_skel = storage.load_partial(name, load_parameters=1, load_results=1)
    assert "alpha" in t_skel.parameters
    assert t_skel.parameters["alpha"].value is None
    assert "df" in t_skel.results
    assert t_skel.results["df"].value is None

    # Load only a specific result (df), skip others
    t_df_only = storage.load_partial(name, load_parameters=0, load_results=2, load_only=["df"])
    assert "df" in t_df_only.results
    assert t_df_only.results["df"].value.equals(df)
    assert "arr_r" not in t_df_only.results
    assert "ser" not in t_df_only.results


def test_load_result_frame_slice_and_array_slice(tmp_path):
    file_path = tmp_path / "partial2.h5"
    name = "partial2"

    traj, parr, rarr, df, ser = _make_sample_traj(name)
    storage = HDF5StorageService(file_path=Path(file_path))
    storage.save(traj)

    # DataFrame slice
    sliced = storage.load_result_frame_slice(name, "df", rows=slice(0, 2), cols=["b"])
    pd.testing.assert_frame_equal(sliced.reset_index(drop=True), df.loc[0:1, ["b"]].reset_index(drop=True))

    # ndarray slice
    got_arr = storage.load_result_array_slice(name, "arr_r", np.s_[1:3, 1:3])
    np.testing.assert_array_equal(got_arr, rarr[1:3, 1:3])
