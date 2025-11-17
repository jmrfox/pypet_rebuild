from __future__ import annotations

from typing import Mapping

from pypet_rebuild import Environment, Parameter, Result, Trajectory


def test_run_grouping_sequential_records_runs_and_results() -> None:
    traj = Trajectory(name="seq-runs")
    traj.add_parameter(Parameter(name="x", value=0))
    traj.add_parameter(Parameter(name="y", value=0))

    def simulate(t: Trajectory) -> Mapping[str, object]:
        x = int(t.parameters["x"].value)
        y = int(t.parameters["y"].value)
        return {"sum": x + y, "prod": x * y}

    env = Environment(trajectory=traj, storage=None)

    env.run_exploration(simulate, space={"x": [1, 2], "y": [10, 20]})

    runs = traj.list_runs()
    assert len(runs) == 4

    # Check one run's params/results
    rid = runs[0]
    params = traj.get_run_params(rid)
    results = traj.get_run_results(rid)
    assert set(results.keys()) == {"sum", "prod"}
    assert set(params.keys()) == {"x", "y"}

    # Mirrored results under by_run namespace should exist
    assert f"by_run.{rid}.sum" in traj.results
    assert f"by_run.{rid}.prod" in traj.results


def test_run_grouping_parallel_records_runs_and_results() -> None:
    traj = Trajectory(name="par-runs")
    traj.add_parameter(Parameter(name="x", value=0))
    traj.add_parameter(Parameter(name="y", value=0))

    def simulate(t: Trajectory) -> Mapping[str, object]:
        x = int(t.parameters["x"].value)
        y = int(t.parameters["y"].value)
        # Also add a direct result to exercise non-return path
        t.add_result(Result(name="diff", value=x - y))
        return {"sum": x + y}

    env = Environment(trajectory=traj, storage=None)

    env.run_exploration_parallel(simulate, space={"x": [1, 2], "y": [10, 20]}, _max_workers=2)

    runs = traj.list_runs()
    assert len(runs) == 4

    rid = runs[0]
    # Returned mapping keys plus direct result should be present
    assert f"by_run.{rid}.sum" in traj.results
    assert f"by_run.{rid}.diff" in traj.results
