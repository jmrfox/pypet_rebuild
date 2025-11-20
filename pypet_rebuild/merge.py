from __future__ import annotations

from typing import Any, Mapping
import json

from .trajectory import Trajectory
from .parameters import Result, Parameter


def _params_signature(params: Mapping[str, Any]) -> str:
    """Compute a stable signature for a params mapping for duplicate detection.

    Tries JSON encoding with sorted keys, falls back to repr strings.
    """
    try:
        return json.dumps(params, sort_keys=True, default=str)
    except Exception:
        items = sorted((k, repr(v)) for k, v in params.items())
        return json.dumps(items)


def merge_trajectories(
    target: Trajectory,
    source: Trajectory,
    *,
    remove_duplicates: bool = True,
) -> None:
    """Merge `source` into `target` in-memory.

    - Parameters with the same name: keep target's value on conflict.
    - Results with the same name: keep target's value on conflict.
    - Runs: append runs from source; if `remove_duplicates`, skip runs whose
      parameter snapshot matches an existing run in target.
    - New runs are assigned new sequential run IDs in `target`.
    """

    # Merge parameters
    for name, param in source._parameters.items():  # noqa: SLF001
        if name not in target._parameters:  # noqa: SLF001
            target.add_parameter(Parameter(name=name, value=param.value, comment=param.comment))

    # Merge non-by_run results first
    for name, res in source._results.items():  # noqa: SLF001
        if name.startswith("by_run."):
            continue
        if name not in target._results:  # noqa: SLF001
            target.add_result(Result(name=name, value=res.value, comment=res.comment))

    # Prepare existing run signatures in target
    existing_sigs: set[str] = set()
    for rec in target._run_records:  # noqa: SLF001
        existing_sigs.add(_params_signature(rec.get("params", {})))

    # Append runs from source
    for rec in source._run_records:  # noqa: SLF001
        params = rec.get("params", {})
        results = rec.get("results", {})
        sig = _params_signature(params)
        if remove_duplicates and sig in existing_sigs:
            continue
        run_id = f"{len(target._run_records):05d}"  # noqa: SLF001
        target.record_run(run_id, params, results)
        existing_sigs.add(sig)
