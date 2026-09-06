from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from .grouped_types import (
    BatchGroupReduceOptions,
    GroupByOptions,
    GroupedLayout,
    GroupMaterializeOptions,
)

_GROUP_LAYOUTS = frozenset({"padded", "stacked"})


class _GroupByUnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


_GROUPBY_UNSET = _GroupByUnsetType()


@dataclass(frozen=True)
class ResolvedGroupedReducerOptions:
    """Topology-neutral grouped reducer options."""

    group_dim: str
    include_empty_groups: bool
    sequence_options: GroupMaterializeOptions | None


def _require_nonempty_name(value: object, *, owner: str, field: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{owner}: {field} must be a non-empty string.")


def _validate_groupby_option_fields(opts: GroupByOptions, *, owner: str) -> None:
    if not isinstance(opts.preserve_batch, bool):
        raise ValueError(f"{owner}: opts.preserve_batch must be bool.")  # noqa: TRY004
    _require_nonempty_name(opts.group_dim, owner=owner, field="opts.group_dim")
    _require_nonempty_name(opts.member_dim, owner=owner, field="opts.member_dim")
    _require_nonempty_name(
        opts.sequence_index_coord,
        owner=owner,
        field="opts.sequence_index_coord",
    )


def validate_sequence_groupby_options(opts: GroupByOptions, *, owner: str) -> None:
    """Validate naming relationships used only by sequence materialization."""
    if len({opts.group_dim, opts.member_dim, opts.sequence_index_coord}) != 3:
        raise ValueError(
            f"{owner}: opts.group_dim, opts.member_dim, and opts.sequence_index_coord "
            "must be distinct."
        )


def coerce_groupby_options(opts: object | None, *, owner: str) -> GroupByOptions:
    if opts is None:
        out = GroupByOptions()
    elif isinstance(opts, GroupByOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be GroupByOptions or None.")
    _validate_groupby_option_fields(out, owner=owner)
    return out


def coerce_groupby_call_options(
    opts: object | None,
    *,
    preserve_batch: bool | _GroupByUnsetType,
    owner: str,
) -> GroupByOptions:
    if opts is not None and preserve_batch is not _GROUPBY_UNSET:
        raise TypeError(f"{owner}: opts cannot be combined with preserve_batch.")
    if preserve_batch is _GROUPBY_UNSET:
        return coerce_groupby_options(opts, owner=owner)
    if not isinstance(preserve_batch, bool):
        raise TypeError(f"{owner}: preserve_batch must be bool when supplied.")
    return coerce_groupby_options(
        GroupByOptions(preserve_batch=preserve_batch),
        owner=owner,
    )


def validate_batch_groupby_options(opts: GroupByOptions, *, owner: str) -> None:
    if opts.preserve_batch:
        raise ValueError(f"{owner}: preserve_batch=True is not supported for batch-only grouping.")
    defaults = GroupByOptions()
    if opts.member_dim != defaults.member_dim:
        raise ValueError(f"{owner}: opts.member_dim applies only to sequence grouping.")
    if opts.sequence_index_coord != defaults.sequence_index_coord:
        raise ValueError(
            f"{owner}: opts.sequence_index_coord applies only to sequence grouping."
        )


def coerce_group_materialize_options(
    opts: object | None,
    *,
    owner: str,
) -> GroupMaterializeOptions:
    if opts is None:
        out = GroupMaterializeOptions()
    elif isinstance(opts, GroupMaterializeOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be GroupMaterializeOptions or None.")
    layout: GroupedLayout = out.layout
    if layout not in _GROUP_LAYOUTS:
        raise ValueError(f"{owner}: opts.layout must be one of {tuple(sorted(_GROUP_LAYOUTS))!r}.")
    if not isinstance(out.include_empty_groups, bool):
        raise ValueError(f"{owner}: opts.include_empty_groups must be bool.")  # noqa: TRY004
    for field in ("group_dim", "member_dim", "sequence_index_coord"):
        value = getattr(out, field)
        if value is not None:
            _require_nonempty_name(value, owner=owner, field=f"opts.{field}")
    return out


def _coerce_batch_group_reduce_options(
    opts: object | None,
    *,
    owner: str,
) -> BatchGroupReduceOptions:
    if opts is None:
        out = BatchGroupReduceOptions()
    elif isinstance(opts, BatchGroupReduceOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be BatchGroupReduceOptions or None for batch grouping.")
    if not isinstance(out.include_empty_groups, bool):
        raise TypeError(f"{owner}: opts.include_empty_groups must be bool.")
    if out.group_dim is not None:
        _require_nonempty_name(out.group_dim, owner=owner, field="opts.group_dim")
    return out


def resolve_layout_names(
    *,
    group_dim: str,
    member_dim: str,
    sequence_index_coord: str,
    opts: GroupMaterializeOptions,
    owner: str,
) -> tuple[str, str, str]:
    resolved_group = group_dim if opts.group_dim is None else opts.group_dim
    resolved_member = member_dim if opts.member_dim is None else opts.member_dim
    resolved_sequence_index = (
        sequence_index_coord
        if opts.sequence_index_coord is None
        else opts.sequence_index_coord
    )
    names = {resolved_group, resolved_member, resolved_sequence_index}
    if len(names) != 3:
        raise ValueError(
            f"{owner}: group/member/sequence-index names must be distinct; "
            f"got {resolved_group!r}, {resolved_member!r}, {resolved_sequence_index!r}."
        )
    return resolved_group, resolved_member, resolved_sequence_index


def _source_namespace_hits(name: str, *, ds: xr.Dataset) -> tuple[str, ...]:
    hits: list[str] = []
    if name in ds.dims:
        hits.append("dim")
    if name in ds.coords:
        hits.append("coord")
    if name in ds.data_vars:
        hits.append("data_var")
    return tuple(hits)


def validate_group_name_collision(
    *,
    ds: xr.Dataset,
    group_dim: str,
    owner: str,
) -> None:
    hits = _source_namespace_hits(group_dim, ds=ds)
    if not hits:
        return
    raise ValueError(
        f"{owner}: opts.group_dim={group_dim!r} collides with source namespace "
        f"({', '.join(hits)})."
    )


def _validate_sequence_layout_name_collisions(
    *,
    ds: xr.Dataset,
    member_dim: str,
    sequence_index_coord: str,
    owner: str,
) -> None:
    checks = (
        ("member_dim", member_dim),
        ("sequence_index_coord", sequence_index_coord),
    )
    for field, value in checks:
        hits = _source_namespace_hits(value, ds=ds)
        if not hits:
            continue
        kinds = ", ".join(hits)
        raise ValueError(
            f"{owner}: opts.{field}={value!r} collides with source namespace ({kinds})."
        )


def _resolved_grouped_reducer_options(
    *,
    ds: xr.Dataset,
    group_dim: str,
    include_empty_groups: bool,
    sequence_options: GroupMaterializeOptions | None,
    owner: str,
) -> ResolvedGroupedReducerOptions:
    validate_group_name_collision(ds=ds, group_dim=group_dim, owner=owner)
    return ResolvedGroupedReducerOptions(
        group_dim=group_dim,
        include_empty_groups=include_empty_groups,
        sequence_options=sequence_options,
    )


def resolve_batch_group_reduce_options(
    opts: object | None,
    *,
    defaults: GroupByOptions,
    ds: xr.Dataset,
    owner: str,
) -> ResolvedGroupedReducerOptions:
    out = _coerce_batch_group_reduce_options(opts, owner=owner)
    group_dim = defaults.group_dim if out.group_dim is None else out.group_dim
    return _resolved_grouped_reducer_options(
        ds=ds,
        group_dim=group_dim,
        include_empty_groups=out.include_empty_groups,
        sequence_options=None,
        owner=owner,
    )


def resolve_sequence_group_reduce_options(
    opts: object | None,
    *,
    defaults: GroupByOptions,
    ds: xr.Dataset,
    owner: str,
) -> ResolvedGroupedReducerOptions:
    materialize = (
        GroupMaterializeOptions(layout="padded", include_empty_groups=False)
        if opts is None
        else coerce_group_materialize_options(opts, owner=owner)
    )
    group_dim, member_dim, sequence_index = resolve_layout_names(
        group_dim=defaults.group_dim,
        member_dim=defaults.member_dim,
        sequence_index_coord=defaults.sequence_index_coord,
        opts=materialize,
        owner=owner,
    )
    _validate_sequence_layout_name_collisions(
        ds=ds,
        member_dim=member_dim,
        sequence_index_coord=sequence_index,
        owner=owner,
    )
    resolved = GroupMaterializeOptions(
        layout=materialize.layout,
        group_dim=group_dim,
        member_dim=member_dim,
        sequence_index_coord=sequence_index,
        include_empty_groups=materialize.include_empty_groups,
    )
    return _resolved_grouped_reducer_options(
        ds=ds,
        group_dim=group_dim,
        include_empty_groups=materialize.include_empty_groups,
        sequence_options=resolved,
        owner=owner,
    )


def validate_layout_name_collisions(
    *,
    ds: xr.Dataset,
    group_dim: str,
    member_dim: str,
    sequence_index_coord: str,
    owner: str,
) -> None:
    validate_group_name_collision(ds=ds, group_dim=group_dim, owner=owner)
    _validate_sequence_layout_name_collisions(
        ds=ds,
        member_dim=member_dim,
        sequence_index_coord=sequence_index_coord,
        owner=owner,
    )


__all__ = [
    "ResolvedGroupedReducerOptions",
    "coerce_group_materialize_options",
    "coerce_groupby_call_options",
    "coerce_groupby_options",
    "resolve_batch_group_reduce_options",
    "resolve_layout_names",
    "resolve_sequence_group_reduce_options",
    "validate_batch_groupby_options",
    "validate_group_name_collision",
    "validate_layout_name_collisions",
    "validate_sequence_groupby_options",
]
