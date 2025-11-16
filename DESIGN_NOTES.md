# pypet_rebuild Design Notes

This document records intentional design choices and deviations from the original **pypet** framework. It should be read together with `PYPET_FRAMEWORK.md`, which summarizes the original design.

---

## 1. Scope of the rebuild

- **Goal**: capture the core ideas of pypet (trajectory, parameter exploration, structured storage) in a modern, typed, and modular codebase.
- **Non-goal**: 1:1 re-implementation of all legacy behaviors, integrations, and APIs.
- When in doubt, we favor **clarity, modularity, and type-safety** over strict backward compatibility.

---

## 2. Trajectory and data model

### 2.1 Flat mappings + namespaces vs. explicit tree nodes

- Original pypet uses an explicit tree structure with groups and leaves.
- In `pypet_rebuild`, the first implementation of `Trajectory` uses:
  - Internal **flat mappings** (`_parameters`, `_results`) keyed by fully-qualified names like `"traffic.ncars"`.
  - Lightweight **namespace views** for natural naming (`traj.parameters.traffic.ncars`).
- Rationale:
  - Keeps the initial implementation small and easy to reason about.
  - Works well with typing and IDEs (mappings + attribute access).
  - We can later introduce richer tree semantics without breaking the public pattern of `traj.parameters.<group>.<leaf>`.

### 2.2 Typed dataclasses for Parameter and Result

- Original pypet provides a deeper and more specialized hierarchy of parameter and result types.
- `pypet_rebuild` starts with simple, generic dataclasses:
  - `Parameter[T]` and `Result[T]`.
- Rationale:
  - Emphasizes type-safety and readability.
  - Makes it straightforward to extend with additional types, while keeping the core minimal.

---

## 3. Environment and execution model

- Original pypet’s `Environment` handles many concerns (exploration, multiprocessing, storage, logging, resuming).
- In `pypet_rebuild`, `Environment` is deliberately **narrow**:
  - `run(func)` to execute a single simulation against the current `Trajectory`.
  - `run_exploration(func, space)` to execute over a cartesian product of parameters.
  - Optional `storage` hook called after runs complete.
- Rationale:
  - Clear separation between **execution control** and **storage**.
  - Easy to extend to parallelism and advanced orchestration later without entangling concerns.

---

## 4. Exploration

- Original pypet provides exploration helpers integrated into the trajectory and environment.
- `pypet_rebuild` exposes a **small, standalone helper**:
  - `cartesian_product(space: Mapping[str, Sequence[Any]]) -> Iterable[dict[str, Any]]`.
  - Consumed by `Environment.run_exploration`.
- Rationale:
  - Exploration is a pure function and does not need access to storage or logging.
  - Easier to swap in more advanced exploration strategies (e.g. optimizers, random search) while keeping the interface simple.

---

## 5. Storage design

### 5.1 StorageService protocol

- Original pypet tightly couples storage logic to its internal tree model and HDF5 backend.
- `pypet_rebuild` introduces a `StorageService` protocol:
  - `save(trajectory: Trajectory) -> None`
  - `load(name: str) -> Trajectory`
- Rationale:
  - Allows HDF5 to be just one backend among others.
  - Enables testing and experimentation with alternate formats (parquet, zarr, etc.) later.

### 5.2 Minimal HDF5 layout and JSON-encoded values

- The first `HDF5StorageService` uses a simple layout:
  - `/trajectories/<name>/parameters/<param_name>`
  - `/trajectories/<name>/results/<result_name>`
  - Values and comments are stored as JSON-encoded attributes.
- Rationale:
  - Keeps the initial implementation small and explicit.
  - Adequate for basic scalar and simple container types.
  - Leaves room to add specialized storage for arrays/DataFrames later (e.g. datasets for NumPy arrays).

---

## 6. Versioning and compatibility

- **Python versions**
  - Original pypet supported Python 2.x in older releases and later Python 3.
  - `pypet_rebuild` targets modern Python only (currently `>=3.12`).

- **Backward compatibility**
  - No guarantee of drop-in compatibility with existing pypet scripts or HDF5 files.
  - Where feasible, we may provide migration helpers or compatibility shims, but this is not a primary goal.

---

## 7. Future design questions

- How far should the new tree model mirror original pypet’s grouping semantics?
- Which integrations (Brian2, Sumatra, DEAP, SCOOP, etc.) belong in-core vs. as separate companion packages?
- What is the minimal, composable API for parallel execution that fits well with `Environment` and `StorageService`?

These questions will be refined as the project evolves, and this document should be updated alongside major design changes.

---

## 8. Parallel execution design (early sketch)

- `Environment.run_exploration_parallel` currently delegates to `run_exploration` and ignores `max_workers`.
- This stabilizes a public entry point for parallel execution without committing to a specific backend.

Planned direction (subject to change):

- Use a pluggable executor interface (e.g. based on `concurrent.futures.Executor`) that can be provided to `Environment`.
- Keep the core API agnostic to the exact parallelism mechanism (threads, processes, external executors).
- Make error handling and partial-failure semantics explicit in this document before implementing true parallelism.
