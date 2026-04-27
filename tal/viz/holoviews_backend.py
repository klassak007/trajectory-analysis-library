from __future__ import annotations

import importlib
from typing import Any

from .prepare import VizPreparedPayload


def _require_holoviews_hvplot(*, owner: str) -> None:
    try:
        importlib.import_module("holoviews")
        importlib.import_module("hvplot.xarray")
    except ImportError as exc:
        raise ImportError(
            f"{owner}: holoviews and hvplot are required. Install optional dependency set 'tal[viz]'."
        ) from exc


def _as_backend_channel(value: tuple[str, ...], *, explorer_mode: bool) -> str | list[str] | None:
    if not value:
        return None
    if explorer_mode:
        return list(value)
    if len(value) == 1:
        return value[0]
    return list(value)


def _render_kind_name(kind: str) -> str:
    return kind


def render_hvplot(payload: VizPreparedPayload, *, owner: str) -> Any:
    _require_holoviews_hvplot(owner=owner)
    kwargs = dict(payload.kwargs)
    if payload.x is not None:
        kwargs.setdefault("x", payload.x)
    explorer_mode = payload.kind == "explorer"
    by = _as_backend_channel(payload.by, explorer_mode=explorer_mode)
    if by is not None:
        kwargs.setdefault("by", by)
    groupby = _as_backend_channel(payload.groupby, explorer_mode=explorer_mode)
    if groupby is not None:
        kwargs.setdefault("groupby", groupby)
    method_name = _render_kind_name(payload.kind)
    accessor = payload.data.hvplot
    if payload.kind == "explorer":
        if not hasattr(accessor, "explorer"):
            raise ValueError(
                f"{owner}: hvplot accessor does not provide explorer() in this hvplot version."
            )
        return accessor.explorer(**kwargs)
    if not hasattr(accessor, method_name):
        raise ValueError(f"{owner}: hvplot accessor does not provide method {method_name!r}.")
    method = getattr(accessor, method_name)
    return method(**kwargs)


__all__ = ["render_hvplot"]
