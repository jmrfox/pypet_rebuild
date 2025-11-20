from pathlib import Path
import numpy as np
import pandas as pd

from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Result
from pypet_rebuild.storage import HDF5StorageService


def main() -> None:
    # Create a trajectory with large-ish arrays
    file_path = Path("examples/output") / "example_09.h5"
    name = "example_09_huge_data"

    traj = Trajectory(name=name)
    traj.add_result(Result(name="huge_matrices.mat1", value=np.random.rand(100, 100, 20)))
    traj.add_result(Result(name="huge_matrices.mat2", value=np.random.rand(500, 500)))
    traj.add_result(Result(name="huge_matrices.note", value="Always look on the bright side of life!"))

    storage = HDF5StorageService(file_path=file_path)
    storage.save(traj)

    # Show element access from the saved file via partial loading
    # Skeleton load results (no data), then selectively load only 'note' and 'mat1'
    t_skel = storage.load_partial(name, load_parameters=0, load_results=1)
    print("Skeleton only? mat1:", t_skel.results["huge_matrices.mat1"].value)

    t_sel = storage.load_partial(
        name,
        load_parameters=0,
        load_results=2,
        load_only=["huge_matrices.note", "huge_matrices.mat1"],
    )
    mat1 = t_sel.results["huge_matrices.mat1"].value  # type: ignore[assignment]
    note = t_sel.results["huge_matrices.note"].value
    print("mat1 shape:", np.asarray(mat1).shape)
    print("note:", note)

    # Demonstrate ndarray slicing direct from disk (no full materialization)
    sl = np.s_[10:15, 20:25, 1:5]
    sl_data = storage.load_result_array_slice(name, "huge_matrices.mat1", sl)
    print("slice shape:", sl_data.shape)


if __name__ == "__main__":
    main()
