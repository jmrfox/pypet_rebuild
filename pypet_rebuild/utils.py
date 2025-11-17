"""Utility helpers for pypet_rebuild.

This module is intentionally small; it hosts generic helpers that do not belong
in a more specific module.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar
from pathlib import Path
import json

import h5py


T = TypeVar("T")


def flatten(nested: Iterable[Iterable[T]]) -> list[T]:
    """Flatten a two-dimensional iterable into a list.

    This is a convenience function and may be replaced or removed as the design
    evolves.
    """

    return [item for sub in nested for item in sub]


def inspect_h5(file_path: str | Path, *, max_preview: int = 3) -> str:
    lines: list[str] = []

    def _decode_attr(v: object) -> object:
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8")
            except Exception:
                return repr(v)
        return v

    def _visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Group):
            attrs = {k: _decode_attr(v) for k, v in obj.attrs.items()}
            desc = f"[Group] /{name}"
            kind = attrs.get("kind") if isinstance(attrs.get("kind"), str) else None
            if kind in {"json", "pandas_series", "pandas_frame"}:
                raw = attrs.get("value")
                preview = None
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        if kind == "pandas_series":
                            idx = parsed.get("index", [])
                            preview = f"pandas.Series len={len(idx)}"
                        elif kind == "pandas_frame":
                            idx = parsed.get("index", [])
                            cols = parsed.get("columns", [])
                            preview = f"pandas.DataFrame shape=({len(idx)},{len(cols)})"
                        else:
                            if isinstance(parsed, dict):
                                keys = list(parsed.keys())[:max_preview]
                                preview = f"json keys={keys}"
                            else:
                                preview = f"json type={type(parsed).__name__}"
                    except Exception:
                        preview = "value=<unparseable>"
                if preview:
                    desc += f" kind={kind} {preview}"
            lines.append(desc)
        else:
            shape = obj.shape
            dtype = obj.dtype
            desc = f"[Dataset] /{name} shape={shape} dtype={dtype}"
            try:
                if obj.size > 0:
                    slices = tuple(slice(0, min(3, n)) for n in shape)
                    data = obj[slices].tolist()
                    desc += f" preview={str(data)[:120]}"
            except Exception:
                pass
            lines.append(desc)

    with h5py.File(Path(file_path), "r") as h5:
        h5.visititems(_visit)

    return "\n".join(lines)
