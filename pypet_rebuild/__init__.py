"""Core public API surface for the pypet_rebuild package.

This module exposes the fundamental abstractions that users will interact with
most frequently: Environment, Trajectory, Parameter, Result, and the
StorageService protocol.

At this stage, these are minimal, typed stubs intended to stabilize naming and
basic structure. Their behavior will be expanded as the design matures.
"""

from __future__ import annotations

from .environment import Environment
from .exploration import cartesian_product
from .exceptions import ConfigurationError, PypetRebuildError, StorageError
from .logging_utils import get_logger
from .parameters import Parameter, Result
from .storage import HDF5StorageService, StorageService
from .trajectory import Trajectory

__all__ = [
    "Environment",
    "Trajectory",
    "Parameter",
    "Result",
    "StorageService",
    "HDF5StorageService",
    "cartesian_product",
    "PypetRebuildError",
    "StorageError",
    "ConfigurationError",
    "get_logger",
]
