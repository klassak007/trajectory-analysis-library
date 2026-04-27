from __future__ import annotations

import xarray as xr

from .grouped_types import GroupByOptions, GroupMaterializeOptions, GroupedLayout

_GROUP_LAYOUTS = frozenset({"padded", "stacked"})


def _require_nonempty_name(value: object, *, owner: str, field: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{owner}: {field} must be a non-empty string.")


def _validate_groupby_options(opts: GroupByOptions, *, owner: str) -> None:
    if not isinstance(opts.preserve_batch, bool):
        raise ValueError(f"{owner}: opts.preserve_batch must be bool.")
    _require_nonempty_name(opts.group_dim, owner=owner, field="opts.group_dim")
    _require_nonempty_name(opts.member_dim, owner=owner, field="opts.member_dim")
    _require_nonempty_name(
        opts.sequence_index_coord,
        owner=owner,
        field="opts.sequence_index_coord",
    )
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
    _validate_groupby_options(out, owner=owner)
    return out


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
        raise ValueError(f"{owner}: opts.include_empty_groups must be bool.")
    for field in ("group_dim", "member_dim", "sequence_index_coord"):
        value = getattr(out, field)
        if value is not None:
            _require_nonempty_name(value, owner=owner, field=f"opts.{field}")
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


def validate_layout_name_collisions(
    *,
    ds: xr.Dataset,
    group_dim: str,
    member_dim: str,
    sequence_index_coord: str,
    owner: str,
) -> None:
    checks = (
        ("group_dim", group_dim),
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


__all__ = [
    "coerce_group_materialize_options",
    "coerce_groupby_options",
    "resolve_layout_names",
    "validate_layout_name_collisions",
]
