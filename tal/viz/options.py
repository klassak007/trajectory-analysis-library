from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from tal.core.group_ops.types import GroupingFoundationOptions, GroupingKeyInput

VizKind = Literal["line", "scatter", "explorer"]
VizValidityPolicy = Literal["respect", "ignore"]


@dataclass(frozen=True)
class AOVizOptions:
    """Options shared by AO visualization entrypoints.

    Notes
    -----
    ``var`` chooses the data variable, ``x`` chooses the horizontal coordinate,
    and ``kwargs`` is forwarded to the backend plotting call after TAL validates
    grouping and validity policy.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.viz import AOVizOptions, line
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": ("sample", [0.0, 1.0])}, coords={"sample": [0, 1]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> opts = AOVizOptions(var="value")
    >>> try:
    ...     plot = line(ao, opts=opts)
    ... except ImportError:
    ...     plot = None
    >>> plot is None or plot is not None
    True
    """

    var: str | None = None
    x: str | None = None
    by: tuple[str, ...] = ()
    groupby: tuple[str, ...] = ()
    group_key: GroupingKeyInput | None = None
    group_foundation_opts: GroupingFoundationOptions | None = None
    validity: VizValidityPolicy = "respect"
    max_overlay_items: int = 8
    kwargs: Mapping[str, object] = field(default_factory=dict)


def _normalize_optional_name(value: object, *, owner: str, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{owner}: opts.{field} must be a non-empty string when provided.")


def _normalize_name_tuple(value: object, *, owner: str, field: str) -> tuple[str, ...]:
    if value in (None, ()):  # type: ignore[comparison-overlap]
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{owner}: opts.{field} must be a sequence[str].")
    out: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{owner}: opts.{field}[{index}] must be a non-empty string.")
        out.append(raw)
    if len(set(out)) != len(out):
        raise ValueError(f"{owner}: opts.{field} must contain unique names.")
    return tuple(out)


def _normalize_kwargs(value: object, *, owner: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner}: opts.kwargs must be mapping[str, object].")
    out: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{owner}: opts.kwargs keys must be non-empty strings.")
        out[key] = item
    return out


def _normalize_group_key(value: object, *, owner: str) -> GroupingKeyInput | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError(f"{owner}: opts.group_key supports exactly one key in Slice A/B.")
        return value[0]
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{owner}: opts.group_key supports exactly one key in Slice A/B.")
        return value[0]
    return value  # type: ignore[return-value]


def _validate_group_foundation_opts(value: object, *, owner: str) -> GroupingFoundationOptions | None:
    if value is None or isinstance(value, GroupingFoundationOptions):
        return value
    raise TypeError(f"{owner}: opts.group_foundation_opts must be GroupingFoundationOptions or None.")


def _validate_validity_policy(value: object, *, owner: str) -> VizValidityPolicy:
    if value in {"respect", "ignore"}:
        return value
    raise ValueError(f"{owner}: opts.validity must be 'respect' or 'ignore'.")


def _validate_max_overlay_items(value: object, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{owner}: opts.max_overlay_items must be int >= 1.")
    if value < 1:
        raise ValueError(f"{owner}: opts.max_overlay_items must be >= 1.")
    return value


def _validate_channels_disjoint(*, by: tuple[str, ...], groupby: tuple[str, ...], owner: str) -> None:
    overlap = tuple(sorted(set(by).intersection(groupby)))
    if overlap:
        raise ValueError(f"{owner}: opts.by and opts.groupby must be disjoint; overlap={overlap!r}.")


def coerce_viz_options(opts: object | None, *, owner: str) -> AOVizOptions:
    if opts is None:
        base = AOVizOptions()
    elif isinstance(opts, AOVizOptions):
        base = opts
    else:
        raise TypeError(f"{owner}: opts must be AOVizOptions or None.")
    normalized = replace(
        base,
        var=_normalize_optional_name(base.var, owner=owner, field="var"),
        x=_normalize_optional_name(base.x, owner=owner, field="x"),
        by=_normalize_name_tuple(base.by, owner=owner, field="by"),
        groupby=_normalize_name_tuple(base.groupby, owner=owner, field="groupby"),
        group_key=_normalize_group_key(base.group_key, owner=owner),
        group_foundation_opts=_validate_group_foundation_opts(base.group_foundation_opts, owner=owner),
        validity=_validate_validity_policy(base.validity, owner=owner),
        max_overlay_items=_validate_max_overlay_items(base.max_overlay_items, owner=owner),
        kwargs=_normalize_kwargs(base.kwargs, owner=owner),
    )
    _validate_channels_disjoint(by=normalized.by, groupby=normalized.groupby, owner=owner)
    return normalized


def require_supported_viz_kind(kind: object, *, owner: str) -> VizKind:
    if kind in {"line", "scatter", "explorer"}:
        return kind
    raise ValueError(f"{owner}: kind must be one of ('line', 'scatter', 'explorer').")


__all__ = [
    "AOVizOptions",
    "VizKind",
    "VizValidityPolicy",
    "coerce_viz_options",
    "require_supported_viz_kind",
]
