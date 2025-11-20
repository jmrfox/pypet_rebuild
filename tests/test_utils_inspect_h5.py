from pathlib import Path

import h5py

from pypet_rebuild.utils import inspect_h5


def test_inspect_h5_handles_unparseable_json(tmp_path):
    fp = Path(tmp_path) / "badjson.h5"
    with h5py.File(fp, "w") as h5:
        root = h5.require_group("trajectories")
        g = root.create_group("t1")
        inner = g.create_group("parameters")
        p = inner.create_group("p1")
        p.attrs["kind"] = "json"
        p.attrs["value"] = "{not json}"
    out = inspect_h5(fp, show_values=True)
    assert "value=<unparseable>" in out


def test_inspect_h5_dataset_preview(tmp_path):
    fp = Path(tmp_path) / "dataset.h5"
    with h5py.File(fp, "w") as h5:
        d = h5.create_dataset("arr", data=[1, 2, 3, 4])
    out = inspect_h5(fp)
    assert "[Dataset] /arr" in out
