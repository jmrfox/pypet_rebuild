from pathlib import Path

from pypet_rebuild.environment import Environment
from pypet_rebuild.trajectory import Trajectory
from pypet_rebuild.parameters import Parameter, Result
from pypet_rebuild.storage import HDF5StorageService


def main() -> None:
    # Create environment and trajectory
    file_path = Path("examples/output") / "example_02.h5"
    env = Environment(
        trajectory=Trajectory(name="Example02"),
        storage=HDF5StorageService(file_path=file_path),
    )
    traj = env.trajectory

    # Add nested parameters using dotted names
    traj.add_parameter(Parameter(name="starwars.characters.han_solo", value="Harrison Ford"))
    traj.add_parameter(Parameter(name="starwars.characters.luke_skywalker", value="Mark Hamill"))

    # Natural naming access (attribute-style) and mapping-style
    characters = traj.parameters.starwars.characters
    print("Luke played by:", characters["luke_skywalker"].value)

    # Store the trajectory
    env.storage.save(traj)  # type: ignore[union-attr]

    # Add a large-ish JSON result and persist it individually
    traj.add_result(Result(name="starwars.gross_income_of_film", value={"amount": 10.1 ** 11, "currency": "$$$"}))
    env.storage.store_result(traj, "starwars.gross_income_of_film")  # type: ignore[union-attr]

    # Demonstrate skeleton vs full load and selective loading
    storage = env.storage  # type: ignore[assignment]

    # Skeleton load: parameters with data; results as skeletons (value=None)
    t_skel = storage.load_partial("Example02", load_parameters=2, load_results=1)
    print("Skeleton result present?", "starwars.gross_income_of_film" in t_skel.results)
    print("Skeleton value:", t_skel.results["starwars.gross_income_of_film"].value)

    # Selective load of just the film income result
    t_sel = storage.load_partial("Example02", load_parameters=0, load_results=2, load_only=["starwars.gross_income_of_film"])
    print("Loaded income:", t_sel.results["starwars.gross_income_of_film"].value)


if __name__ == "__main__":
    main()
