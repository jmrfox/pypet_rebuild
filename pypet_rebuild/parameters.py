"""Parameter and result abstractions for pypet_rebuild.

These are intentionally small, typed containers that will likely evolve into a
hierarchy of more specialized parameter and result types. For now, they provide
strongly-typed, documented building blocks for trajectories and storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class Parameter(Generic[T]):
    """Represents an input to a simulation.

    Parameters are identified by a logical name (which may encode a hierarchy),
    hold a value of generic type ``T``, and may carry an optional comment or
    human-readable description.
    """

    name: str
    value: T
    comment: Optional[str] = None


@dataclass(slots=True)
class Result(Generic[T]):
    """Represents an output produced by a simulation run.

    Results mirror parameters structurally but are typically created at runtime
    and attached to a trajectory after computation.
    """

    name: str
    value: T
    comment: Optional[str] = None
