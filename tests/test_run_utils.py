from pypet_rebuild.environment import Environment
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.exploration import cartesian_product


def _simulate_mul(traj: Trajectory):
    x = traj.parameters["x"].value
    y = traj.parameters["y"].value
    z = x * y
    traj.add_result(Result(name="z", value=z))
    return {"z": z}


def test_find_runs_and_collect_runs():
    t = Trajectory(name="t_find_collect")
    t.add_parameter(Parameter(name="x", value=0))
    t.add_parameter(Parameter(name="y", value=0))

    env = Environment(trajectory=t, storage=None)
    space = {"x": [1, 2, 3, 4], "y": [6, 7, 8]}

    env.run_exploration(_simulate_mul, space)

    # find runs where x==2 or y==8
    matched = t.find_runs(lambda x, y: (x == 2) or (y == 8), names=["x", "y"])
    # Validate that each matched run satisfies predicate
    for run_id in matched:
        p = t.get_run_params(run_id)
        assert (p.get("x") == 2) or (p.get("y") == 8)

    # collect z across runs in execution order and verify values
    zs = t.collect_runs("z")
    expected = [c["x"] * c["y"] for c in cartesian_product(space)]
    assert zs == expected
