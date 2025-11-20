from __future__ import annotations

from typing import Mapping

from pypet_rebuild import Environment, Parameter, Result, Trajectory


def simulate_proc(t: Trajectory) -> Mapping[str, object]:
    x = int(t.parameters["x"].value)
    y = int(t.parameters["y"].value)
    # Write a direct result as well as return a mapping
    t.add_result(Result(name="prod", value=x * y))
    return {"sum": x + y}


def test_process_parallel_records_runs_and_results() -> None:
    traj = Trajectory(name="proc-runs")
    traj.add_parameter(Parameter(name="x", value=0))
    traj.add_parameter(Parameter(name="y", value=0))

    env = Environment(trajectory=traj, storage=None)

    env.run_exploration_processes(
        simulate_proc,
        space={"x": [1, 2], "y": [10, 20]},
        _max_workers=2,
    )

    runs = traj.list_runs()
    assert len(runs) == 4

    rid = runs[0]
    assert f"by_run.{rid}.sum" in traj.results
    assert f"by_run.{rid}.prod" in traj.results
