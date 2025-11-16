"""Tests for the basic exploration utilities and environment integration."""

from __future__ import annotations

from pypet_rebuild import Environment, Parameter, Result, Trajectory, cartesian_product


def test_cartesian_product_helper() -> None:
    space = {"x": [1, 2], "y": [10, 20]}

    combos = list(cartesian_product(space))

    assert {tuple(sorted(d.items())) for d in combos} == {
        (("x", 1), ("y", 10)),
        (("x", 1), ("y", 20)),
        (("x", 2), ("y", 10)),
        (("x", 2), ("y", 20)),
    }


def test_environment_run_exploration_updates_parameters_and_results() -> None:
    traj = Trajectory(name="explore")

    # Seed trajectory with parameters (optional; they would be created on demand otherwise)
    traj.add_parameter(Parameter(name="x", value=0))
    traj.add_parameter(Parameter(name="y", value=0))

    def simulate(t: Trajectory) -> None:
        x = t.parameters["x"].value
        y = t.parameters["y"].value
        t.add_result(Result(name=f"product.{x}_{y}", value=x * y))

    env = Environment(trajectory=traj, storage=None)

    env.run_exploration(
        simulate,
        space={
            "x": [1, 2],
            "y": [10, 20],
        },
    )

    # We expect one result per combination
    expected_names = {
        "product.1_10",
        "product.1_20",
        "product.2_10",
        "product.2_20",
    }

    assert expected_names.issubset(set(traj.results.keys()))
