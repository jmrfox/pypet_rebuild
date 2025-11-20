# Original pypet Examples: Reference Guide

This document summarizes the examples from the original pypet repository (branch `develop`) under `examples/`. It explains what each example does, the key pypet features demonstrated, and ideas for how/if to recreate it in `pypet_rebuild`.

---

## Core Getting Started

- **example_01_first_steps.py**
  - Multiplies two parameters `x` and `y` across a cartesian grid and stores result `z`.
  - Demonstrates: Environment + Trajectory basics, `f_explore(cartesian_product)`, `env.run`, HDF5 file creation, reloading a trajectory and printing specific run results.
  - Rebuild mapping: Already covered by our SIR sweep and simple multiplication patterns.
  - Porting notes: Implement a minimal `examples/multiply.py` using `Environment.run_exploration(_parallel/_processes)` and `by_run` mirrors; add a tiny notebook variant.

- **example_02_trajectory_access_and_storage.py**
  - Shows natural naming, adding groups and parameters, unique vs non-unique node access, fast access (`v_fast_access`), storing to HDF5, storing individual items, and reloading skeleton vs data.
  - Demonstrates: rich tree operations, `f_store`, `f_store_item`, partial load (`load_parameters` / `load_results`), and value vs object access distinctions.
  - Rebuild mapping: We have basic storage and natural naming; future work: richer tree semantics, skeleton/data separation, per-item store/load.
  - Porting notes: Add section to `Trajectory` on unique vs non-unique semantics (defer strict errors); expose item-level load/store API in storage; add demo to show skeleton vs data (lazy slice helpers exist for ndarray).

- **example_03_trajectory_merging.py**
  - Runs two trajectories with overlapping parameter spaces and merges them, removing duplicates; loads data and iterates runs.
  - Demonstrates: merging trajectories (`f_merge`), duplicate removal, backups, moving/deleting nodes, and `f_set_crun` run scoping.
  - Rebuild mapping: Not implemented; consider later: merge utilities and run-scoped views.
  - Porting notes: Provide an offline merge utility operating on HDF5 files/trajectories with duplicate detection by param snapshot; add read-only run-scoped view helpers.

## Parallel and Many Runs

- **example_04_multiprocessing.py**
  - Multiprocessing via Environment config (`multiproc`, `ncores`, pool, freeze input), graceful exit, queue wrapping. Runs 20x20 grid.
  - Demonstrates: process-based parallelism with various knobs (pool, freeze_input), CTRL+C friendly shutdown.
  - Rebuild mapping: We have threads/processes; advanced knobs (freeze input, wrap modes) are out-of-scope initially.
  - Porting notes: Map to `run_exploration_processes(..., _max_workers, resume=True)`; document Windows-safe pickling contract; skip pool/wrap knobs.

- **example_12_sharing_data_between_processes.py**
  - Uses a `multiprocessing.Manager().list()` to share interim data across processes and adds final result after run.
  - Demonstrates: shared state pattern, disabling continuation when shared state exists.
  - Rebuild mapping: Documented pattern; we won’t ship shared state helpers initially.
  - Porting notes: Provide a recipe in docs showing Manager/list usage and warning about `resume=False` due to external state; prefer returning mappings.

- **example_16_multiproc_context.py**
  - `MultiprocContext` to wrap storage with locking; multiple processes mutate a trajectory-safe result; reload and show last process name.
  - Demonstrates: storage wrapping and locks for safe concurrent writes.
  - Rebuild mapping: Useful concept; likely a future add-on; for now use post-merge writes only.
  - Porting notes: Consider context-manager wrapper for `HDF5StorageService` with file locks; out-of-scope for MVP.

- **example_18_many_runs.py**
  - Explores 2,500 runs; uses `$set`/$ bucket wildcards to group runs to avoid HDF5 node limits; demonstrates auto-loading and run-to-set/index helpers.
  - Demonstrates: run bucketing, wildcards, auto-load, run index shortcuts.
  - Rebuild mapping: We have run grouping; bucketing/wildcards are advanced and can be deferred.
  - Porting notes: If needed, add optional bucketing config to storage (e.g., group runs by thousands) and helpers; otherwise rely on HDF5 node counts being manageable.

- **example_21_scoop_multiprocessing.py**
  - Runs with SCOOP executor; requires `python -m scoop` launch; LOCAL/NETLOCK wrapping.
  - Demonstrates: distributed multiprocessing backend.
  - Rebuild mapping: Out-of-scope for MVP; consider adapters later.
  - Porting notes: Add docs note about potential adapters (Dask/Ray) rather than SCOOP; keep interface via `Environment`.

## Data Management and Access Patterns

- **example_08_f_find_idx.py**
  - Executes grid and loads results; uses `f_find_idx` with a predicate over parameters to find matching runs and print results.
  - Demonstrates: run filtering via parameter predicates.
  - Rebuild mapping: Provide a simple run-filter utility over recorded run snapshots.
  - Porting notes: Implement `Trajectory.find_runs(predicate, names=[...]) -> list[run_id]` using `_run_records`; add example usage and tests.

- **example_09_large_results.py**
  - Adds huge arrays, stores trajectory, empties memory, later loads skeleton and selectively loads specific fields (`load_only`), shows element access.
  - Demonstrates: partial (lazy) loading, per-item store, large array handling.
  - Rebuild mapping: We started with ndarray slicing helpers; add structured per-item load API and skeleton/data split.
  - Porting notes: Extend storage with per-item load (`load_only`) and skeleton mode flags; keep ndarray slicing as first milestone; provide notebook showcasing memory savings.

- **example_10_get_items_from_all_runs.py**
  - After computing `z`, loads all results, fetches parameter ranges and gathers all `z` across runs for plotting; reshapes into 2D mesh for 3D surface plot.
  - Demonstrates: retrieving values from all runs, parameter range access.
  - Rebuild mapping: Provide convenience to collect a named result across runs from `by_run.*` (we have per-run mirrors already).
  - Porting notes: Add helper `collect_runs(result_name) -> list[Any] | DataFrame` leveraging `by_run` mirror; show plotting recipe.

- **example_14_links.py**
  - Shows creating links (aliases) to parameters (`f_add_link`) and using them within simulations.
  - Demonstrates: link nodes for aliasing; colon notation to create groups on the fly.
  - Rebuild mapping: Aliases/links are advanced; defer until tree semantics mature.
  - Porting notes: Explore alias support post tree-semantic design; meanwhile encourage explicit names and avoid aliasing in core.

- **example_15_more_ways_to_add_data.py**
  - Multiple ways to add parameters/results (direct instance assignment, auto-renaming empty names, dot-path group creation), and adding a link via assignment.
  - Demonstrates: flexible adds and links via natural naming syntax sugar.
  - Rebuild mapping: Keep core adds explicit for clarity; revisit sugar once tree is richer.
  - Porting notes: Consider minimal sugar for nested creation via dot-paths; defer auto-renaming and links.

## DEAP and Domain Examples

- **example_19_using_deap.py**, **example_19b_using_deap_less_overhead.py**, **example_19c_using_deap_with_post_processing.py**, **example_20_using_deap_manual_runs.py**
  - Integrate evolutionary search with DEAP; variants reduce overhead or add post-processing; manual run mode variant.
  - Demonstrates: adaptive exploration, post-processing integration points, manual run orchestration.
  - Rebuild mapping: Provide hooks for external search tools; DEAP adapter can be a separate recipe later.
  - Porting notes: Start with a generic optimization loop recipe using our `Environment` + run grouping; later add a DEAP-specific how-to.

- **example_23_brian2_network.py**
  - Builds/executes a Brian2 neural network; uses `Brian2Parameter` and `Brian2MonitorResult` to store monitors.
  - Demonstrates: domain-specific parameter/result types; dynamic class import for storage.
  - Rebuild mapping: Out-of-core integration; move to companion package or recipes.
  - Porting notes: Plan a companion `pypet_rebuild-brian2` package; document how to serialize monitors (HDF5 groups + attrs) cleanly.

- **example_13_post_processing/** (dir)
  - Post-processing pipeline example (folder contains more than one file; not expanded here).
  - Demonstrates: structuring post-processing workflows after runs.
  - Rebuild mapping: Our hooks system can support post-run/campaign post-processing.
  - Porting notes: Add `Environment` hooks (pre_run, post_run, post_campaign) and a simple post-processing script example.

- **example_17_wrapping_an_existing_project/** (dir)
  - Shows how to wrap an existing project within pypet’s trajectory/storage model.
  - Demonstrates: integration wrapper patterns.
  - Rebuild mapping: Consider a short guide once core API stabilizes.
  - Porting notes: Provide a template wrapper that maps an external project’s config/results to `Trajectory` parameters/results.

- **example_22_saga_python/**, **example_24_large_scale_brian2_simulation/** (dirs)
  - Advanced external tooling and large-scale simulation scaffolding.
  - Rebuild mapping: Out-of-scope for MVP, capture as future work.
  - Porting notes: Keep as future work items; note likely requirements (scheduler integration, dataset partitioning, chunked writes).

---

## Recommendations for pypet_rebuild Examples

- **Short term to recreate**
  - First steps (01) with multiplication (we already have SIR; add a minimal multiply example).
  - Trajectory access & storage (02): natural naming, store/load skeleton vs data (partial), per-item store/load (as available).
  - Multiprocessing (04): threads/processes example (done) with resume semantics.
  - Large results (09): demonstrate lazy/slicing and selective load.
  - Get items from all runs (10): helper to collect `by_run.*.<name>`.
  - Find idx (08): simple filtering API over run snapshots.

- **Later**
  - Merging (03), links (14), sugar adds (15), many runs bucketing (18), SCOOP (21), DEAP (19*), Brian2 (23), post-processing dirs (13, 17, 22, 24).
