import numpy as np
import pandas as pd

from pathlib import Path

from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.storage import HDF5StorageService


def test_hdf5_dynamic_array_slice(tmp_path):
    file_path = tmp_path / "dyn.h5"
    name = "traj1"

    traj = Trajectory(name=name)
    arr = np.arange(100).reshape(10, 10)
    rarr = np.arange(60).reshape(6, 10)
    traj.add_parameter(Parameter(name="arr", value=arr))
    traj.add_result(Result(name="rarr", value=rarr))

    storage = HDF5StorageService(file_path=Path(file_path))
    storage.save(traj)

    # Slice parameters ndarray
    sl = np.s_[2:5, 3:7]
    got = storage.load_param_array_slice(name, "arr", sl)
    np.testing.assert_array_equal(got, arr[sl])

    # Slice results ndarray
    sl2 = np.s_[1:4, 2:9]
    got2 = storage.load_result_array_slice(name, "rarr", sl2)
    np.testing.assert_array_equal(got2, rarr[sl2])
