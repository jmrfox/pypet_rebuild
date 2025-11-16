"""Exploration utilities for pypet_rebuild.

This module provides small, typed helpers for defining how parameters should be
varied across multiple simulation runs. The initial focus is on cartesian
products of discrete parameter value lists.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from typing import Any, Dict, Iterator


def cartesian_product(space: Mapping[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    """Yield dictionaries representing the cartesian product of a parameter space.

    Parameters
    ----------
    space:
        A mapping from parameter names to sequences of candidate values.

    Yields
    ------
    dict[str, Any]
        One dictionary per combination, mapping parameter names to chosen
        values.
    """

    if not space:
        return []  # type: ignore[return-value]

    keys = list(space.keys())
    value_lists = [space[key] for key in keys]

    def _iter() -> Iterator[Dict[str, Any]]:
        for combo in product(*value_lists):
            yield dict(zip(keys, combo))

    return _iter()
