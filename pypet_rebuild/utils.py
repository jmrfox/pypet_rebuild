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


def inspect_h5(
    file_path: str | Path,
    *,
    max_preview: int = 3,
    show_values: bool = False,
    value_max_chars: int = 200,
    show_attrs: bool = False,
) -> str:
    lines: list[str] = []

    def _decode_attr(v: object) -> object:
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8")
            except UnicodeDecodeError:
                return repr(v)
        return v

    def _truncate(v: object) -> str:
        s = repr(v)
        return s if len(s) <= value_max_chars else s[: value_max_chars - 3] + "..."

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
                if show_values and isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        if kind == "json":
                            if isinstance(parsed, dict):
                                for k, v in list(parsed.items())[:max_preview]:
                                    lines.append(f"  value.{k} = {_truncate(v)}")
                            else:
                                lines.append(f"  value = {_truncate(parsed)}")
                        elif kind == "pandas_series":
                            idx = parsed.get("index", [])
                            data = parsed.get("data", [])
                            for i in range(min(max_preview, len(idx))):
                                lines.append(f"  series[{_truncate(idx[i])}] = {_truncate(data[i])}")
                        elif kind == "pandas_frame":
                            idx = parsed.get("index", [])
                            cols = parsed.get("columns", [])
                            data = parsed.get("data", [])
                            for r in range(min(max_preview, len(idx))):
                                row = data[r] if r < len(data) else []
                                for c in range(min(max_preview, len(cols))):
                                    val = row[c] if c < len(row) else None
                                    lines.append(
                                        f"  df[{_truncate(idx[r])},{_truncate(cols[c])}] = {_truncate(val)}"
                                    )
                    except (json.JSONDecodeError, TypeError, ValueError):
                        lines.append("  value=<unparseable>")
            if show_attrs and attrs:
                safe_attrs = {k: _truncate(v) for k, v in attrs.items() if k != "value"}
                if safe_attrs:
                    lines.append(f"  attrs={safe_attrs}")
            lines.append(desc)
        else:
            shape = obj.shape
            dtype = obj.dtype
            desc = f"[Dataset] /{name} shape={shape} dtype={dtype}"
            try:
                if obj.size > 0:
                    slices = tuple(slice(0, min(max_preview, n)) for n in shape)
                    data = obj[slices].tolist()
                    if show_values and (obj.size <= max_preview ** max(1, len(shape))):
                        desc += f" values={_truncate(data)}"
                    else:
                        desc += f" preview={_truncate(data)}"
            except (RuntimeError, TypeError, ValueError):
                pass
            lines.append(desc)

    with h5py.File(Path(file_path), "r") as h5:
        h5.visititems(_visit)

    return "\n".join(lines)
