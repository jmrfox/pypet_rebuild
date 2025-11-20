# Original pypet Framework

This document summarizes the design and capabilities of the original **pypet** (Python Parameter Exploration Toolkit) project. It is intended as a concise, comprehensive reference to the original framework’s concepts, modules, and behavior.

---

## 1. High-Level Purpose and Philosophy

- **Problem domain**
  - Numerical simulations and scientific experiments often need to:
    - **Systematically explore parameter spaces** (sweeps, grids, optimization loops).
    - **Store large amounts of simulation data** in a structured, durable, queryable format.
  - Ad-hoc I/O and parameter management leads to brittle scripts, hard-to-reuse results, and difficulty reproducing or extending past work.

- **pypet’s core idea**
  - Provide a **unified framework** to:
    - Declare and organize parameters.
    - Explore combinations of parameter values as a **trajectory** through parameter space.
    - Run simulations (possibly in parallel) under framework control.
    - Store parameters, results, and metadata **automatically** into HDF5 via PyTables.
  - Introduce a central **tree-like container** (`Trajectory`) for parameters/results with **natural naming access** (e.g. `traj.parameters.traffic.ncars`).

- **Primary benefits**
  - Separation of concerns:
    - User defines the **simulation function** (what to compute for a given parameter set).
    - `pypet` handles **exploration, orchestration, storage, and metadata**.
  - Emphasis on **reproducibility, reusability, and scalability** (local machine → clusters, single process → multiprocessing / SCOOP).

---

## 2. Core Concepts and Data Model

### 2.1 Trajectory

- **Trajectory** is the central container representing:
  - A **parameter space** and the exploration over that space.
  - The **results** and associated metadata for all runs.
- Conceptually:
  - A **tree structure** of **groups** and **leaves**.
  - Groups can hold subgroups, parameters, results, and annotations.
- Usage patterns:
  - Define parameters (with default values and metadata).
  - Call exploration methods to generate combinations of those parameters.
  - For each combination, the trajectory is updated to reflect the “current run”.
  - The simulation function reads parameters and writes back results into the trajectory.

### 2.2 Parameters and Results

- Defined and managed via `pypet.parameters` and `pypet.trajectory`.
- **Parameters**
  - Represent **inputs** to simulations.
  - Have:
    - A **name** (hierarchical path in the tree).
    - A **value**.
    - A **comment / annotation**.
    - Type / container semantics depending on parameter subclass.
- **Results**
  - Represent **outputs** produced by simulations.
  - Stored back into the trajectory under `results`-related groups.
  - Can be created at runtime via something like `traj.f_add_result(...)`.

### 2.3 Natural Naming

- **Natural naming** is a key ergonomic feature:
  - Access parameters and results as attributes instead of manual lookups.
  - Example: `traj.parameters.traffic.ncars` instead of low-level key-based API.
- Implemented by the `pypet.naturalnaming` module, which maps attribute access to the underlying tree nodes.

### 2.4 Groups and Annotations

- **Groups**
  - Logical containers in the trajectory tree: e.g. `parameters`, `results`, domain-specific subgroups.
  - Allow meaningful organization of complex experiments.
- **Annotations**
  - Used to tag parameters, results, and groups with additional metadata.
  - Facilitate documentation, analysis, and filtering.

---

## 3. Main Framework Components (Package Organization)

From the original project overview:

- **`pypet.environment`**
  - Handles **running simulations**.
  - Creates and configures the `Trajectory` and storage backend.
  - Orchestrates:
    - Exploration over parameter combinations.
    - Running the user’s simulation function.
    - Multiprocessing and/or distributed execution (e.g. SCOOP).

- **`pypet.trajectory`**
  - Core implementation of the **Trajectory** data structure.
  - Manages:
    - Parameters and results.
    - Group hierarchy.
    - Exploration state and current run context.
  - Integrates with natural naming and storage services.

- **`pypet.naturalnaming`**
  - Provides natural-name access to trajectory nodes.
  - Maps dotted attribute-like access (`traj.parameters.foo.bar`) to actual nodes.

- **`pypet.parameters`**
  - Defines parameter and result container classes.
  - Supports different data types and containers (scalars, arrays, pandas objects, etc.).

- **`pypet.storageservice`**
  - Implements **saving data to disk**.
  - Primary backend: **HDF5** via PyTables.
  - Supports dynamic loading (only load portions of data on demand).
  - Integrates with the trajectory tree to persist parameters, results, and metadata.

These modules together form the basis for `Environment` + `Trajectory` + I/O abstraction.

---

## 4. Storage Model and Supported Data Types

### 4.1 Storage Backend

- Uses **HDF5** as the storage format via **PyTables**.
- Key characteristics:
  - Hierarchical structure matches the trajectory tree.
  - Efficient for large numerical datasets.
  - Supports partial reads (dynamic loading).

### 4.2 Supported Data

The original `pypet` supports storing many data types, including:

- **Python native types**
  - `bool`, `int`, `long` (legacy), `float`, `str`, `complex`.
- **Containers**
  - `list`, `tuple`, `dict`.
- **Scientific array/matrix types**
  - NumPy arrays and matrices.
  - SciPy sparse matrices.
- **Tabular and time-series**
  - pandas `DataFrame`.
- **Domain-specific integrations**
  - Brian2 quantities and monitors.

The design emphasizes **extensibility**, allowing users to add support for additional data formats through custom parameter/result types or storage logic.

---

## 5. Parameter Exploration and Execution Model

### 5.1 Exploration

- `pypet` offers tools to build **exploration sets** over parameters, such as:
  - Cartesian products of parameter value lists.
  - Potential integration with optimization frameworks (e.g. DEAP) for adaptive exploration.
- Typical workflow:
  1. Add parameters with default values to the trajectory.
  2. Use an exploration builder (e.g. `cartesian_product`) to define combinations.
  3. Call `traj.f_explore(...)` to register the exploration plan.

### 5.2 Environment and Simulation Function

- Users define a **simulation function** such as:

  ```python
  def simulate(traj):
      # read parameters from traj
      # perform computation
      # write results to traj via traj.f_add_result(...)
  ```

- Then create an **Environment**:
  - Configure trajectory name, HDF5 filename, title, comments, etc.
  - Possibly set multiprocessing options (number of processors, etc.).

- Execution pattern:
  1. Environment creates a `Trajectory` and attaches storage service.
  2. User adds parameters and exploration to the trajectory.
  3. Environment **runs** the simulation function over all parameter combinations:
     - For each combination, set current parameter values in the trajectory.
     - Call the simulation function with that trajectory.
     - Store results and metadata.

### 5.3 Quick Working Example (Conceptual)

From the original docs, a minimal example:

- Aim: multiply two values `x` and `y` across a cartesian product of possible values.
- Workflow:
  - Define `multiply(traj)` that uses `traj.x`, `traj.y` and stores `z` via `traj.f_add_result('z', z, ...)`.
  - Create an `Environment` with a trajectory name and HDF5 output file.
  - Retrieve the `Trajectory` from `env.trajectory`.
  - Add parameters `x` and `y` with default values.
  - Explore via cartesian product over lists of `x` and `y` values (`traj.f_explore(...)`).
  - Call `env.run(multiply)` to execute all runs and store data.

This example illustrates the **core pattern** for all simulations implemented with `pypet`.

---

## 6. Parallelism, Clusters, and Integration with Other Tools

### 6.1 Multiprocessing and Clusters

- `pypet` supports **multiprocessing** to run simulations in parallel on a single machine.
- In combination with **SCOOP**, `pypet` can:
  - Run on **multiple servers or cluster nodes**.
  - Distribute runs over a network while still using the central trajectory and storage model.

### 6.2 Integration with External Tools

- **DEAP (optimization)**
  - pypet can be integrated with the DEAP evolutionary algorithms framework for **adaptive parameter exploration**.
  - Typical use: search parameter space intelligently instead of exhaustive cartesian sweeps.

- **Sumatra (electronic lab notebook)**
  - Optional **Sumatra integration** allows automatic recording of simulations in an electronic lab notebook.
  - Supports tracking experiments, parameters, versions, etc.

- **Git integration**
  - pypet can make **automatic Git commits** of the codebase associated with runs.
  - Helps with **reproducibility and provenance**.

- **Brian2 (neural simulation)**
  - Specialized support for **Brian2** parameters, results, monitors, and a Brian2 network framework.
  - pypet can act as a parameter exploration and storage layer for Brian2-based simulations.

---

## 7. Logging, Metadata, and Robustness Features

### 7.1 Logging and Diagnostics

- Dedicated logging utilities and configuration via `pypet` modules (e.g. slots/logging docs).
- Logging is integrated with the Environment and trajectory to record:
  - Run progress.
  - Errors and exceptions.
  - Status of storage.

### 7.2 Dynamic Loading

- **Dynamic loading** allows loading only the parts of the data that are needed.
  - Critical for large HDF5 files.
  - Fits well with interactive analysis workflows and memory constraints.

### 7.3 Resume Crashed or Halted Simulations

- pypet supports **resuming** simulations:
  - Continue trajectories that were partially executed or interrupted.
  - Feature relies heavily on HDF5 storage and trajectory-based bookkeeping.

### 7.4 Continuing Crashed Trajectories (Optional Dependencies)

- The docs mention optional packages like `dill` for **continuing crashed trajectories**.
- Additional optional dependencies like `psutil` can be used for process/multiprocessing cap features.

---

## 8. Dependencies and Optional Features (Original Project)

### 8.1 Core Requirements

- **Python**: 3.7+ (legacy releases supported Python 2.7 up to pypet 0.3.0).
- Core scientific stack:
  - `numpy`
  - `scipy`
  - `tables` (PyTables)
  - `pandas`
  - HDF5 library (system-level dependency).

### 8.2 Optional Dependencies

- **SCOOP** (>= 0.7.1) for distributed execution.
- **GitPython** (>= 3.1.3) for Git integration.
- **psutil** (>= 5.7.0) for multiprocessing cap features.
- **dill** (>= 0.3.1) for continuing crashed trajectories.
- **Sumatra** (>= 0.7.1) for electronic lab notebook integration.
- **Brian2** (not strictly required but widely supported) for neural simulations.

---

## 9. Documentation Layout (Original Project)

The original docs (ReadTheDocs) organize content into:

- **User Manual**
  - What is pypet all about?
  - Getting Started
  - Tutorial
  - Cookbook (detailed manual)
  - Examples
  - Optimization Tips
  - FAQs and Known Issues

- **Library Reference**
  - The Environment
  - The Trajectory and Group Nodes
  - Parameters and Results
  - Annotations
  - Utils
  - Exceptions
  - Global Constants
  - Slots and Logging
  - Storage Services
  - Brian2 Parameters, Results, Monitors, and Network Framework

- **Miscellaneous**
  - Publication information
  - Acknowledgments
  - Tests
  - Changelog

This structure reflects the framework’s emphasis on **conceptual introduction**, practical **tutorials**, and detailed **API-level documentation**.

---

 
## 11. Feature Checklist (Original pypet)

- **Trajectory (Tree Container)**
  - Hierarchical groups and leaves for parameters and results
  - Natural naming and dot-path access
  - Annotations/metadata on nodes; comments and tags

- **Parameters**
  - Typed parameter nodes (int/float/str/bool, arrays, pandas, etc.)
  - Exploration ranges and sampling; cartesian expansion
  - Derived parameters and linking between nodes

- **Results**
  - Result nodes for simulation outputs
  - Storage of scalars, arrays, tabular data, and custom objects (with pickling)

- **Exploration & Execution**
  - Cartesian product exploration over parameter space
  - Environment coordinates runs and manages lifecycle hooks
  - Post-processing and analysis phases

- **Storage (HDF5 via PyTables)**
  - Structured, automatic persistence of trajectory tree
  - Resume/continue runs; dynamic/lazy loading of data
  - Compression/chunking options; version and provenance info

- **Parallel & Distributed**
  - Local multiprocessing
  - Optional SCOOP-based distributed execution

- **Integrations**
  - Brian2 (spiking neural networks)
  - DEAP (evolutionary algorithms)
  - Sumatra (experiment tracking)
  - GitPython, psutil, dill and other ecosystem tools

- **Robustness & Reproducibility**
  - Run grouping, snapshots, and provenance metadata
  - Logging and error handling; partial run recovery

- **Documentation & Examples**
  - Comprehensive user manual and library reference
  - Tutorials and domain-specific guides
