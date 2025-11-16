"""Basic tests for the initial trajectory and environment scaffolding.

These tests intentionally exercise only a thin slice of behavior to keep the
scaffold small but meaningful. They help ensure that the public API surface can
stabilize before we add more complex features like exploration and storage.
"""

from __future__ import annotations

from pypet_rebuild import Environment, Parameter, Result, Trajectory


def test_add_parameter_and_result_round_trip() -> None:
    traj = Trajectory(name="example")

    x = Parameter(name="x", value=1.0, comment="first parameter")
    traj.add_parameter(x)

    assert "x" in traj.parameters
    assert traj.parameters["x"].value == 1.0

    z = Result(name="z", value=2.0, comment="a result")
    traj.add_result(z)

    assert "z" in traj.results
    assert traj.results["z"].value == 2.0


def test_environment_runs_simulation_function() -> None:
    traj = Trajectory(name="env-example")

    def simulate(t: Trajectory) -> None:
        t.add_result(Result(name="answer", value=42))

    env = Environment(trajectory=traj, storage=None)
    env.run(simulate)

    assert "answer" in traj.results
    assert traj.results["answer"].value == 42


def test_natural_naming_for_parameters_and_results() -> None:
    traj = Trajectory(name="natural-naming")

    traffic_param = Parameter(name="traffic.ncars", value=10)
    traj.add_parameter(traffic_param)

    # Mapping-style access should continue to work
    assert "traffic.ncars" in traj.parameters
    assert traj.parameters["traffic.ncars"].value == 10

    # Natural naming via attributes
    group = traj.parameters.traffic
    assert getattr(group, "ncars").value == 10

    # Natural naming for results mirrors parameters
    traj.add_result(Result(name="metrics.accuracy", value=0.9))
    assert "metrics.accuracy" in traj.results
    metrics_group = traj.results.metrics
    assert getattr(metrics_group, "accuracy").value == 0.9
