"""Logging configuration for pypet_rebuild.

Rather than introducing a custom logging system, pypet_rebuild uses Python's
standard :mod:`logging` library. This module provides convenience functions for
creating loggers with a consistent naming scheme and default configuration.
"""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger for pypet_rebuild.

    If *name* is ``None``, the top-level ``"pypet_rebuild"`` logger is
    returned. Callers are free to further configure handlers or integrate with
    application-level logging configuration.
    """

    logger_name = "pypet_rebuild" if name is None else f"pypet_rebuild.{name}"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
