"""Custom exceptions for pypet_rebuild.

These are thin wrappers around standard exceptions for now, intended to make it
clear when an error originates from the framework itself.
"""

from __future__ import annotations


class PypetRebuildError(Exception):
    """Base exception for all pypet_rebuild-specific errors."""


class StorageError(PypetRebuildError):
    """Raised when a storage backend encounters a recoverable error."""


class ConfigurationError(PypetRebuildError):
    """Raised for invalid or inconsistent configuration of the framework."""
