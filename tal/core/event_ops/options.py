from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from ..orchestration.lazy import is_chunked_dataarray
from ..param_ops.guards import coerce_float_scalar
from .types import (
    AroundOptions,
    AtBoundariesOptions,
    ConditionEvalOptions,
    EventExtractOptions,
    IntervalExtractOptions,
    WhenOptions,
)

_VALID_AO_INTERP = {"linear", "nearest"}
_VALID_VALIDITY_MODE = {"auto", "prefix_only", "finite_gather"}
_VALID_TRUTH_EVAL = {"exact", "before", "after", "nearest"}
_VALID_AT_BOUNDARIES_EDGES = {"all", "enter", "exit"}
_VALID_AT_BOUNDARIES_MODES = {"all", "first", "first_n", "last"}
_VALID_AT_BOUNDARIES_ON_EMPTY = {"empty", "error"}
_VALID_WHEN_LAYOUTS = {"mask", "segments", "stream"}
_VALID_WHEN_ON_EMPTY = {"empty", "error"}
_VALID_AROUND_EDGES = {"all", "enter", "exit"}
_VALID_AROUND_LAYOUTS = {"segments", "stacked"}


class _AroundUnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


_AROUND_UNSET = _AroundUnsetType()


def _validate_coord_name(name: str, *, owner: str) -> None:
    if isinstance(name, str) and name:
        return
    raise ValueError(f"{owner}: opts.coord_name must be a non-empty string.")


def _validate_sample_dim(sample_dim: str | None, *, owner: str) -> None:
    if sample_dim is None:
        return
    if isinstance(sample_dim, str) and sample_dim:
        return
    raise ValueError(f"{owner}: opts.sample_dim must be a non-empty string or None.")


def _validate_interp(method: str, *, owner: str) -> None:
    if method in _VALID_AO_INTERP:
        return
    raise ValueError(f"{owner}: opts.ao_interp must be one of {sorted(_VALID_AO_INTERP)!r}.")


def _validate_validity_mode(mode: str, *, owner: str) -> None:
    if mode in _VALID_VALIDITY_MODE:
        return
    raise ValueError(f"{owner}: opts.validity_mode must be one of {sorted(_VALID_VALIDITY_MODE)!r}.")


def _validate_tolerance(value: float, *, owner: str, field: str) -> None:
    if np.isfinite(value) and value >= 0.0:
        return
    raise ValueError(f"{owner}: opts.{field} must be finite and >= 0.0.")


def _coerce_tolerance(value: object, *, owner: str, field: str) -> float:
    out = coerce_float_scalar(value, owner=owner, field=f"opts.{field}")
    _validate_tolerance(out, owner=owner, field=field)
    return out


def validate_condition_eval_options(
    opts: ConditionEvalOptions,
    *,
    owner: str,
) -> None:
    _validate_coord_name(opts.coord_name, owner=owner)
    _validate_sample_dim(opts.sample_dim, owner=owner)
    _validate_interp(opts.ao_interp, owner=owner)
    _validate_validity_mode(opts.validity_mode, owner=owner)
    _coerce_tolerance(opts.eq_atol, owner=owner, field="eq_atol")
    _coerce_tolerance(opts.eq_rtol, owner=owner, field="eq_rtol")


def _validate_include_initial(include_initial: bool, *, owner: str) -> None:
    if isinstance(include_initial, bool):
        return
    raise ValueError(f"{owner}: opts.include_initial must be bool.")


def _validate_truth_eval(truth_eval: str, *, owner: str) -> None:
    if truth_eval in _VALID_TRUTH_EVAL:
        return
    raise ValueError(f"{owner}: opts.truth_eval must be one of {sorted(_VALID_TRUTH_EVAL)!r}.")


def _validate_positive_int_or_none(value: int | None, *, owner: str, field: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{owner}: opts.{field} must be a positive int or None.")


def _coerce_finite_scalar(value: object, *, owner: str, field: str) -> float:
    out = coerce_float_scalar(value, owner=owner, field=f"opts.{field}")
    if not np.isfinite(out):
        raise ValueError(f"{owner}: opts.{field} must be finite.")
    return out


def _all_finite_unchunked(da: xr.DataArray) -> bool:
    finite = xr.apply_ufunc(np.isfinite, da, dask="allowed").astype(bool)
    return bool(finite.all())


def coerce_around_grid(
    grid: xr.DataArray | np.ndarray,
    *,
    owner: str,
) -> xr.DataArray:
    da = grid if isinstance(grid, xr.DataArray) else xr.DataArray(np.asarray(grid))
    if da.ndim != 1:
        raise ValueError(f"{owner}: opts.grid must be a 1-D array of finite numeric values.")
    try:
        out = da.astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: opts.grid must be numeric (coercible to float64).") from exc
    if is_chunked_dataarray(out):
        raise ValueError(
            f"{owner}: opts.grid must be unchunked; chunked around grids are not supported."
        )
    if not _all_finite_unchunked(out):
        raise ValueError(f"{owner}: opts.grid must contain only finite values.")
    return out


def validate_event_extract_options(
    opts: EventExtractOptions,
    *,
    owner: str,
) -> None:
    if not isinstance(opts.eval, ConditionEvalOptions):
        raise TypeError(f"{owner}: opts.eval must be ConditionEvalOptions.")
    validate_condition_eval_options(opts.eval, owner=owner)
    _validate_include_initial(opts.include_initial, owner=owner)
    _validate_truth_eval(opts.truth_eval, owner=owner)
    _validate_positive_int_or_none(opts.max_events, owner=owner, field="max_events")
    _coerce_tolerance(opts.dedupe_atol, owner=owner, field="dedupe_atol")


def validate_interval_extract_options(
    opts: IntervalExtractOptions,
    *,
    owner: str,
) -> None:
    if not isinstance(opts.eval, ConditionEvalOptions):
        raise TypeError(f"{owner}: opts.eval must be ConditionEvalOptions.")
    validate_condition_eval_options(opts.eval, owner=owner)
    _validate_include_initial(opts.include_initial, owner=owner)
    _validate_truth_eval(opts.truth_eval, owner=owner)
    _validate_positive_int_or_none(opts.max_segments, owner=owner, field="max_segments")


def validate_at_boundaries_options(
    opts: AtBoundariesOptions,
    *,
    owner: str,
) -> None:
    if not isinstance(opts.eval, ConditionEvalOptions):
        raise TypeError(f"{owner}: opts.eval must be ConditionEvalOptions.")
    validate_condition_eval_options(opts.eval, owner=owner)
    if opts.edges not in _VALID_AT_BOUNDARIES_EDGES:
        raise ValueError(
            f"{owner}: opts.edges must be one of {sorted(_VALID_AT_BOUNDARIES_EDGES)!r}."
        )
    if opts.mode not in _VALID_AT_BOUNDARIES_MODES:
        raise ValueError(
            f"{owner}: opts.mode must be one of {sorted(_VALID_AT_BOUNDARIES_MODES)!r}."
        )
    if opts.on_empty not in _VALID_AT_BOUNDARIES_ON_EMPTY:
        raise ValueError(
            f"{owner}: opts.on_empty must be one of {sorted(_VALID_AT_BOUNDARIES_ON_EMPTY)!r}."
        )
    _validate_positive_int_or_none(opts.max_events, owner=owner, field="max_events")
    if opts.mode == "first_n" and opts.max_events is None:
        raise ValueError(f"{owner}: opts.mode='first_n' requires opts.max_events.")


def validate_when_options(
    opts: WhenOptions,
    *,
    owner: str,
) -> None:
    if not isinstance(opts.eval, ConditionEvalOptions):
        raise TypeError(f"{owner}: opts.eval must be ConditionEvalOptions.")
    validate_condition_eval_options(opts.eval, owner=owner)
    if opts.layout not in _VALID_WHEN_LAYOUTS:
        raise ValueError(f"{owner}: opts.layout must be one of {sorted(_VALID_WHEN_LAYOUTS)!r}.")
    if not isinstance(opts.inside, bool):
        raise ValueError(f"{owner}: opts.inside must be bool.")
    if opts.on_empty not in _VALID_WHEN_ON_EMPTY:
        raise ValueError(f"{owner}: opts.on_empty must be one of {sorted(_VALID_WHEN_ON_EMPTY)!r}.")
    _validate_positive_int_or_none(opts.max_segments, owner=owner, field="max_segments")


def validate_around_options(
    opts: AroundOptions,
    *,
    owner: str,
) -> None:
    if not isinstance(opts.eval, ConditionEvalOptions):
        raise TypeError(f"{owner}: opts.eval must be ConditionEvalOptions.")
    validate_condition_eval_options(opts.eval, owner=owner)
    if opts.edge not in _VALID_AROUND_EDGES:
        raise ValueError(f"{owner}: opts.edge must be one of {sorted(_VALID_AROUND_EDGES)!r}.")
    if opts.layout not in _VALID_AROUND_LAYOUTS:
        raise ValueError(f"{owner}: opts.layout must be one of {sorted(_VALID_AROUND_LAYOUTS)!r}.")
    pre = _coerce_finite_scalar(opts.pre, owner=owner, field="pre")
    post = _coerce_finite_scalar(opts.post, owner=owner, field="post")
    if pre < 0.0 or post < 0.0:
        raise ValueError(f"{owner}: opts.pre and opts.post must be >= 0.0.")
    if opts.dt is None and opts.grid is None:
        raise ValueError(f"{owner}: opts.dt is required when opts.grid is None.")
    if opts.dt is not None and opts.grid is not None:
        raise ValueError(f"{owner}: opts.dt and opts.grid are mutually exclusive.")
    if opts.dt is not None:
        dt = _coerce_finite_scalar(opts.dt, owner=owner, field="dt")
        if dt <= 0.0:
            raise ValueError(f"{owner}: opts.dt must be > 0.0 when provided.")
    if opts.grid is not None:
        if not isinstance(opts.grid, (xr.DataArray, np.ndarray)):
            raise ValueError(f"{owner}: opts.grid must be xr.DataArray, np.ndarray, or None.")
        coerce_around_grid(opts.grid, owner=owner)


def coerce_condition_eval_options(
    opts: object | None,
    *,
    owner: str,
) -> ConditionEvalOptions:
    if opts is None:
        out = ConditionEvalOptions()
    elif isinstance(opts, ConditionEvalOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be ConditionEvalOptions or None.")
    validate_condition_eval_options(out, owner=owner)
    return out


def coerce_event_extract_options(
    opts: object | None,
    *,
    owner: str,
) -> EventExtractOptions:
    if opts is None:
        out = EventExtractOptions()
    elif isinstance(opts, EventExtractOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be EventExtractOptions or None.")
    validate_event_extract_options(out, owner=owner)
    return out


def coerce_interval_extract_options(
    opts: object | None,
    *,
    owner: str,
) -> IntervalExtractOptions:
    if opts is None:
        out = IntervalExtractOptions()
    elif isinstance(opts, IntervalExtractOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be IntervalExtractOptions or None.")
    validate_interval_extract_options(out, owner=owner)
    return out


def coerce_at_boundaries_options(
    opts: object | None,
    *,
    owner: str,
) -> AtBoundariesOptions:
    if opts is None:
        out = AtBoundariesOptions()
    elif isinstance(opts, AtBoundariesOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be AtBoundariesOptions or None.")
    validate_at_boundaries_options(out, owner=owner)
    return out


def coerce_when_options(
    opts: object | None,
    *,
    owner: str,
) -> WhenOptions:
    if opts is None:
        out = WhenOptions()
    elif isinstance(opts, WhenOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be WhenOptions or None.")
    validate_when_options(out, owner=owner)
    return out


def coerce_around_options(
    opts: object | None,
    *,
    owner: str,
    edge: Literal["enter", "exit", "all"] | _AroundUnsetType = _AROUND_UNSET,
    pre: float | _AroundUnsetType = _AROUND_UNSET,
    post: float | _AroundUnsetType = _AROUND_UNSET,
    dt: float | None | _AroundUnsetType = _AROUND_UNSET,
    layout: Literal["segments", "stacked"] | _AroundUnsetType = _AROUND_UNSET,
) -> AroundOptions:
    overrides = {
        name: value
        for name, value in (
            ("edge", edge),
            ("pre", pre),
            ("post", post),
            ("dt", dt),
            ("layout", layout),
        )
        if value is not _AROUND_UNSET
    }
    if opts is not None and overrides:
        fields = ", ".join(overrides)
        raise TypeError(f"{owner}: opts cannot be combined with keyword overrides: {fields}.")
    if opts is None:
        out = AroundOptions(**overrides)
    elif isinstance(opts, AroundOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be AroundOptions or None.")
    validate_around_options(out, owner=owner)
    return out


__all__ = [
    "coerce_condition_eval_options",
    "coerce_around_grid",
    "coerce_around_options",
    "coerce_when_options",
    "coerce_event_extract_options",
    "coerce_interval_extract_options",
    "coerce_at_boundaries_options",
    "validate_condition_eval_options",
    "validate_around_options",
    "validate_when_options",
    "validate_event_extract_options",
    "validate_interval_extract_options",
    "validate_at_boundaries_options",
]
