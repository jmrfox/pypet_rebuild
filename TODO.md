# pypet_rebuild TODO

High-level roadmap and progress tracking for rebuilding the original **pypet** framework with modern Python practices.

---

## 1. Project Scope and Goals

- **Goal**
  - Rebuild pypet as a modern, well-typed, well-tested Python package focused on:
    - Parameter exploration for numerical simulations.
    - Robust, extensible data storage and metadata handling.
    - Clean APIs and maintainable internals.

- **Non-goals (tentative)**
  - Full 1:1 re-implementation of every legacy integration from day one.
  - Support for obsolete Python versions (Python 2.x).

(*These can be refined as we iterate.*)

---

## 2. Immediate Tasks

- [ ] Document original framework
  - [x] Create `PYPET_FRAMEWORK.md` summarizing original pypet design.
  - [ ] Identify which legacy features we want to keep, simplify, or drop.

- [x] Establish project scaffolding
  - [x] Choose minimum supported Python version (currently `>=3.12` in `pyproject.toml`).
  - [x] Create package structure (`pypet_rebuild/` and `tests/`).
  - [x] Add `pyproject.toml` (modern packaging) managed via `uv`.
  - [ ] Configure basic CI (lint + tests) and code style (e.g. `ruff`, `black`, `isort`).

- [x] Define first slice of the core domain model
  - [x] Specify initial **Trajectory**, **Parameter**, and **Result** APIs (typed dataclasses, mapping views).
  - [x] Implement a lightweight natural naming approach (attribute access + mapping protocol) for parameters/results.
  - [ ] Design error handling strategy (custom exceptions, invariants) and document it.

---

## 3. Core Framework Design

- [x] **Environment and Execution Engine**
  - [x] Design an `Environment` abstraction for running simulations (`run`, `run_exploration`).
  - [x] Define the interface of a simulation function (callable taking a `Trajectory`).
  - [ ] Decide on how to configure runs (config objects, builder patterns, or simple kwargs).

- [x] **Trajectory and Data Tree** (first pass)
  - [x] Implement a modern `Trajectory` with flat internal mappings and namespace-based grouping.
  - [x] Provide natural naming that works well with IDEs (attribute access + mapping semantics).
  - [ ] Explore stronger tree semantics and separation of in-memory vs persisted representation.

- [ ] **Parameters and Results API**
  - [x] Introduce initial, composable `Parameter[T]` and `Result[T]` dataclasses.
  - [ ] Decide how to support generic containers (lists, dicts) vs. specialized types.
  - [ ] Document constraints and patterns for adding new parameter types.

---

## 4. Storage and I/O

- [x] **Storage abstraction layer**
  - [x] Create a storage interface (`StorageService` protocol).
  - [x] Implement a minimal HDF5-based backend (`HDF5StorageService` using `h5py`).
  - [ ] Keep the design flexible enough to plug in alternative backends later (e.g. parquet, zarr).

- [ ] **Data type support**
  - [x] Start with core types (Python primitives and JSON-serializable structures).
  - [ ] Add support for NumPy arrays.
  - [ ] Add support for pandas (DataFrame, Series) where it adds clear value.
  - [ ] Plan how/if to support SciPy sparse matrices and domain-specific types.

- [ ] **Dynamic loading and partial reads**
  - [ ] Design a mechanism for lazy/dynamic loading of data.
  - [ ] Document guarantees about when data is in-memory vs. on disk.

---

## 5. Parallelism and Distributed Execution

- [ ] Choose the primary parallelism model(s):
  - [ ] `multiprocessing` / `concurrent.futures` for local parallelism.
  - [ ] Optional integration points for external tools (Dask, Ray, SCOOP, etc.).

- [ ] Define clear boundaries so that:
  - [ ] The core library stays usable in single-process mode.
  - [ ] Parallelism is opt-in and doesnt complicate basic usage.

---

## 6. Integrations and Optional Features

- [ ] **Git / Version control integration**
  - [ ] Decide whether to provide built-in git integration (e.g. optional helper) or just document best practices.

- [ ] **Experiment tracking / lab notebooks**
  - [ ] Evaluate whether to reintroduce Sumatra-style integration or instead provide hooks for modern tools (e.g. MLflow, Weights & Biases).

- [ ] **Domain-specific support (e.g. Brian2)**
  - [ ] Decide whether to keep such integrations in-core or move them to separate companion packages.

---

## 7. UX, API Design, and Documentation

- [ ] **API ergonomics**
  - [ ] Provide clear, minimal hello-world examples (e.g. multiplication example updated).
  - [ ] Optimize naming and module structure for discoverability.

- [ ] **Documentation**
  - [ ] Set up Sphinx or MkDocs.
  - [ ] Port/modernize key conceptual docs from original pypet.
  - [ ] Provide a tutorial and cookbook-like examples.

- [ ] **Type hints and IDE support**
  - [ ] Fully type-hint the public API.
  - [ ] Use mypy or pyright for type checking (at least for CI).

---

## 8. Testing and Quality

- [x] **Testing strategy**
  - [x] Set up pytest-based test suite.
  - [x] Add basic unit tests for core abstractions (trajectory, environment, exploration, storage).
  - [ ] Include property-based tests where beneficial (e.g. `hypothesis`) for data round-trips.

- [ ] **Backward-compatibility strategy**
  - [ ] Decide how closely to follow original semantics and behavior.
  - [ ] Add tests around any compatibility shims or intentional deviations.

---

## 9. Documentation and Design Notes

- [ ] **README and high-level docs**
  - [x] Add an overview of the framework and goals to `README.md`.
  - [x] Document current improvements over the original pypet (modern typing, storage abstraction, exploration API, etc.).
  - [ ] Keep README in sync as major features (parallelism, richer storage) are added.

- [ ] **Deeper design documentation**
  - [ ] Create a `DESIGN_NOTES.md` (or similar) capturing intentional deviations from original pypet.
  - [ ] Cross-reference `PYPET_FRAMEWORK.md` where appropriate.

---

## 10. Milestones

*(Initial sketch; can be refined as we go.)*

- **Milestone 1: Core skeleton & minimal trajectory**
  - [x] Package scaffold.
  - [x] Basic `Trajectory` with parameters and natural naming.
  - [x] Minimal HDF5 storage backend (write/read simple runs).

- **Milestone 2: Usable basic framework**
  - [x] `Environment` for running simple simulations.
  - [x] Basic exploration (cartesian product) and result storage.
  - [ ] Basic documentation and examples (hello-world, quick start).

- **Milestone 3: Extensions and quality**
  - [ ] Parallel execution support.
  - [ ] Extended data type support.
  - [ ] More robust logging, resuming, and metadata features.

As we make decisions and implement features, we can update this `TODO.md` to track what’s done and what’s changed from the original pypet design.
