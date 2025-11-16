"""Utility helpers for pypet_rebuild.

This module is intentionally small; it hosts generic helpers that do not belong
in a more specific module.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar


T = TypeVar("T")


def flatten(nested: Iterable[Iterable[T]]) -> list[T]:
    """Flatten a two-dimensional iterable into a list.

    This is a convenience function and may be replaced or removed as the design
    evolves.
    """

    return [item for sub in nested for item in sub]
