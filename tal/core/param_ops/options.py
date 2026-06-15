from __future__ import annotations

import datetime as _datetime
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .guards import coerce_float_scalar, validate_query_dim_name
from .types import ParamKind, ParamEvalOptions, ParamSelectOptions, ParamSyncOptions

_SYNC_JOINS = {"left", "right", "outer", "inner", "domain", "exact", "override"}
_SYNC_HOW = {"interp", "nearest", "fill"}
_SYNC_BATCH_JOIN = {"inner", "outer", "exact"}
_EVAL_METHODS = {"nearest", "linear"}
_DUPLICATE_POLICIES = {"invalid", "left", "right", "raise"}


def coerce_select_options(opts: object | None, *, owner: str) -> ParamSelectOptions:
    if opts is None:
        out = ParamSelectOptions()
    elif isinstance(opts, ParamSelectOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be ParamSelectOptions or None.")
    validate_query_dim_name(out.query_dim, owner=owner)
    validate_select_options(out, owner=owner)
    return out


def coerce_eval_options(opts: object | None, *, owner: str) -> ParamEvalOptions:
    if opts is None:
        out = ParamEvalOptions()
    elif isinstance(opts, ParamEvalOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be ParamEvalOptions or None.")
    validate_query_dim_name(out.query_dim, owner=owner)
    validate_eval_options(out, owner=owner)
    return out


def coerce_sync_options(opts: object | None, *, owner: str) -> ParamSyncOptions:
    if opts is None:
        out = ParamSyncOptions()
    elif isinstance(opts, ParamSyncOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be ParamSyncOptions or None.")
    validate_query_dim_name(out.query_dim, owner=owner)
    validate_sync_options(out, owner=owner)
    return out


def validate_select_options(opts: ParamSelectOptions, *, owner: str) -> None:
    if opts.method != "nearest":
        raise ValueError(f"{owner}: opts.method must be 'nearest'.")
    if opts.layout not in ("packed", "padded"):
        raise ValueError(f"{owner}: opts.layout must be one of ['packed', 'padded'].")


def validate_eval_options(opts: ParamEvalOptions, *, owner: str) -> None:
    if opts.method not in _EVAL_METHODS:
        raise ValueError(
            f"{owner}: opts.method must be one of {sorted(_EVAL_METHODS)!r}, got {opts.method!r}."
        )
    if opts.duplicate_policy not in _DUPLICATE_POLICIES:
        raise ValueError(
            f"{owner}: opts.duplicate_policy must be one of "
            f"{sorted(_DUPLICATE_POLICIES)!r}, got {opts.duplicate_policy!r}."
        )


def validate_sync_options(opts: ParamSyncOptions, *, owner: str) -> None:
    if opts.join not in _SYNC_JOINS:
        raise ValueError(
            f"{owner}: opts.join must be one of {sorted(_SYNC_JOINS)!r}, got {opts.join!r}."
        )
    if opts.how not in _SYNC_HOW:
        raise ValueError(
            f"{owner}: opts.how must be one of {sorted(_SYNC_HOW)!r}, got {opts.how!r}."
        )
    if opts.batch_join not in _SYNC_BATCH_JOIN:
        raise ValueError(
            f"{owner}: opts.batch_join must be one of {sorted(_SYNC_BATCH_JOIN)!r}, got {opts.batch_join!r}."
        )


def _is_timedelta_like(value: object) -> bool:
    return isinstance(value, (np.timedelta64, _datetime.timedelta, pd.Timedelta))


def _timedelta_ns(value: object, *, owner: str) -> int:
    try:
        delta = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: opts.tol must be a nonnegative timedelta for datetime64 params.") from exc
    if pd.isna(delta) or delta < pd.Timedelta(0):
        raise ValueError(f"{owner}: opts.tol must be a nonnegative timedelta for datetime64 params.")
    return int(np.asarray(delta.to_timedelta64()).astype("timedelta64[ns]").astype("int64"))


def _normalize_datetime_tol(value: object, *, owner: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{owner}: opts.tol must be a timedelta for datetime64 params; numeric bool is invalid.")
    if _is_timedelta_like(value):
        return _timedelta_ns(value, owner=owner)
    numeric = coerce_float_scalar(value, owner=owner, field="opts.tol")
    if numeric == 0.0:
        return 0
    raise ValueError(f"{owner}: nonzero numeric opts.tol is not valid for datetime64 params; use a timedelta.")


def normalize_sync_tol(opts: ParamSyncOptions, *, owner: str, param_kind: ParamKind) -> float | int:
    if param_kind == "datetime64":
        return _normalize_datetime_tol(opts.tol, owner=owner)
    if _is_timedelta_like(opts.tol):
        raise ValueError(f"{owner}: timedelta opts.tol is only valid for datetime64 params.")
    tol = coerce_float_scalar(opts.tol, owner=owner, field="opts.tol")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f"{owner}: opts.tol must be finite and >= 0.0, got {opts.tol!r}.")
    return tol


def normalize_sync_fill_value(value: object, *, owner: str) -> float | int:
    if value is None:
        return float("nan")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{owner}: opts.fill_value must be a numeric scalar or None/np.nan.")
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{owner}: opts.fill_value must be a numeric scalar or None/np.nan.")
    if not np.isscalar(value):
        raise ValueError(f"{owner}: opts.fill_value must be a numeric scalar or None/np.nan.")
    arr = np.asarray(value)
    if arr.ndim != 0 or arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{owner}: opts.fill_value must be a numeric scalar or None/np.nan.")
    return arr.item()


def resolve_sync_runtime(opts: ParamSyncOptions, *, owner: str, param_kind: ParamKind) -> tuple[float | int, float | int]:
    tol = normalize_sync_tol(opts, owner=owner, param_kind=param_kind)
    fill = normalize_sync_fill_value(opts.fill_value, owner=owner) if opts.how == "fill" else float("nan")
    return tol, fill


__all__ = [
    "coerce_eval_options",
    "coerce_select_options",
    "coerce_sync_options",
    "normalize_sync_fill_value",
    "normalize_sync_tol",
    "resolve_sync_runtime",
    "validate_select_options",
]
