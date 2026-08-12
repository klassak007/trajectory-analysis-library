from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from .backends import datatree_child_payload_dataset


@dataclass(frozen=True)
class BatchCoordSpec:
    name: str
    template: xr.DataArray


def plan_datatree_child_payloads(
    children: Mapping[str, xr.DataTree],
    *,
    labels: tuple[str, ...],
    batch_dim: str,
    owner: str,
) -> tuple[tuple[xr.Dataset, ...], tuple[BatchCoordSpec, ...]]:
    payloads = tuple(
        datatree_child_payload_dataset(
            children[label],
            batch_dim=batch_dim,
            batch_position=position,
        )
        for position, label in enumerate(labels)
    )
    specs = _batch_coord_specs(children, batch_dim=batch_dim, owner=owner)
    return payloads, specs


def expand_datatree_rows(
    payloads: tuple[xr.Dataset, ...],
    *,
    children: Mapping[str, xr.DataTree],
    labels: tuple[str, ...],
    batch_dim: str,
    batch_coord_specs: tuple[BatchCoordSpec, ...],
    owner: str,
) -> list[xr.Dataset]:
    return [
        _expand_child_row(
            ds,
            child=children[label],
            batch_dim=batch_dim,
            batch_position=position,
            batch_coord_specs=batch_coord_specs,
            label=label,
            owner=owner,
        )
        for position, (ds, label) in enumerate(zip(payloads, labels, strict=True))
    ]


def canonicalize_selected_payload_vars(
    payloads: tuple[xr.Dataset, ...],
    *,
    selected: tuple[str, ...],
    labels: tuple[str, ...],
    owner: str,
) -> tuple[xr.Dataset, ...]:
    expected = {name: payloads[0][name].dims for name in selected}
    return tuple(
        _canonicalize_payload_vars(
            payload,
            selected=selected,
            expected=expected,
            label=label,
            owner=owner,
        )
        for payload, label in zip(payloads, labels, strict=True)
    )


def _canonicalize_payload_vars(
    ds: xr.Dataset,
    *,
    selected: tuple[str, ...],
    expected: Mapping[str, tuple[str, ...]],
    label: str,
    owner: str,
) -> xr.Dataset:
    updates: dict[str, xr.Variable] = {}
    for name in selected:
        dims = ds[name].dims
        target = expected[name]
        if set(dims) != set(target):
            raise ValueError(
                f"{owner}: variable {name!r} has incompatible dimensions across DataTree groups; "
                f"expected {target!r}, got {dims!r} for group {label!r}."
            )
        if dims != target:
            updates[name] = ds[name].transpose(*target).variable
    return ds.assign(updates) if updates else ds


def concat_datatree_rows(rows: Sequence[xr.Dataset], *, batch_dim: str) -> xr.Dataset:
    return xr.concat(
        rows,
        dim=batch_dim,
        join="exact",
        compat="identical",
        coords="minimal",
        combine_attrs="drop_conflicts",
    )


def structural_scalar_metadata_coord_names(
    specs: tuple[BatchCoordSpec, ...],
) -> frozenset[str]:
    return frozenset(spec.name for spec in specs if spec.template.dims == ())


def expand_empty_datatree_template(
    ds: xr.Dataset,
    *,
    batch_dim: str,
) -> tuple[xr.Dataset, frozenset[str]]:
    names = frozenset(
        name
        for name, coord in ds.coords.items()
        if coord.dims == ()
    )
    payload = ds.drop_vars(sorted(names))
    labels = xr.DataArray(np.asarray([], dtype=object), dims=(batch_dim,))
    expanded = payload.expand_dims({batch_dim: labels})
    coords = {
        name: ds.coords[name].expand_dims({batch_dim: labels}).variable
        for name in names
    }
    return (expanded.assign_coords(coords) if coords else expanded), names


def _batch_coord_specs(
    children: Mapping[str, xr.DataTree],
    *,
    batch_dim: str,
    owner: str,
) -> tuple[BatchCoordSpec, ...]:
    specs: list[BatchCoordSpec] = []
    by_name: dict[str, BatchCoordSpec] = {}
    for label, child in children.items():
        local = child.to_dataset(inherit=False)
        for name, coord in local.coords.items():
            if _is_dimension_coord(local, name=name):
                continue
            if name == batch_dim:
                raise ValueError(
                    f"{owner}: child-local coordinate {name!r} in DataTree group {label!r} "
                    f"collides with resolved batch_dim {batch_dim!r}."
                )
            existing = by_name.get(name)
            if existing is not None:
                _require_compatible_coord_dims(
                    coord,
                    spec=existing,
                    label=label,
                    owner=owner,
                )
                continue
            spec = BatchCoordSpec(name=name, template=coord)
            specs.append(spec)
            by_name[name] = spec
    _append_root_batch_coord_specs(
        specs,
        by_name=by_name,
        children=children,
        batch_dim=batch_dim,
    )
    return tuple(specs)


def _is_dimension_coord(ds: xr.Dataset, *, name: str) -> bool:
    return name in ds.dims and ds.coords[name].dims == (name,)


def _append_root_batch_coord_specs(
    specs: list[BatchCoordSpec],
    *,
    by_name: Mapping[str, BatchCoordSpec],
    children: Mapping[str, xr.DataTree],
    batch_dim: str,
) -> None:
    if not children:
        return
    child_values = tuple(children.values())
    parent = child_values[0].parent
    if parent is None:
        return
    local_dims = tuple(set(child.to_dataset(inherit=False).dims) for child in child_values)
    for name, coord in parent.to_dataset(inherit=False).coords.items():
        row_dims = tuple(dim for dim in coord.dims if dim != batch_dim)
        if name in by_name or batch_dim not in coord.dims or not row_dims:
            continue
        if not any(set(row_dims).issubset(dims) for dims in local_dims):
            continue
        specs.append(
            BatchCoordSpec(
                name=name,
                template=coord.isel({batch_dim: 0}, drop=True),
            )
        )


def _require_compatible_coord_dims(
    coord: xr.DataArray,
    *,
    spec: BatchCoordSpec,
    label: str,
    owner: str,
) -> None:
    if set(coord.dims) == set(spec.template.dims):
        return
    raise ValueError(
        f"{owner}: coordinate {spec.name!r} has incompatible dimensions across DataTree groups; "
        f"expected {spec.template.dims!r}, got {coord.dims!r} for group {label!r}."
    )


def _expand_child_row(
    ds: xr.Dataset,
    *,
    child: xr.DataTree,
    batch_dim: str,
    batch_position: int,
    batch_coord_specs: tuple[BatchCoordSpec, ...],
    label: str,
    owner: str,
) -> xr.Dataset:
    payload = _without_batch_dependent_coords(ds, specs=batch_coord_specs)
    try:
        expanded = payload.expand_dims({batch_dim: [label]})
    except ValueError as exc:
        raise ValueError(
            f"{owner}: explicit batch_dim {batch_dim!r} collides with child payload dims for group "
            f"{label!r}; choose a non-colliding constructor batch_dim."
        ) from exc
    batch_coords = _child_batch_coords(
        ds,
        child=child,
        specs=batch_coord_specs,
        batch_dim=batch_dim,
        batch_position=batch_position,
        label=label,
        owner=owner,
    )
    return expanded.assign_coords(batch_coords)


def _without_batch_dependent_coords(
    ds: xr.Dataset,
    *,
    specs: tuple[BatchCoordSpec, ...],
) -> xr.Dataset:
    names = [
        spec.name
        for spec in specs
        if spec.name in ds.coords
    ]
    return ds.drop_vars(names, errors="ignore")


def _child_batch_coords(
    ds: xr.Dataset,
    *,
    child: xr.DataTree,
    specs: tuple[BatchCoordSpec, ...],
    batch_dim: str,
    batch_position: int,
    label: str,
    owner: str,
) -> dict[str, xr.Variable]:
    out: dict[str, xr.Variable] = {}
    for spec in specs:
        if not set(spec.template.dims).issubset(ds.dims):
            continue
        coord = _resolved_child_coord(
            child,
            ds=ds,
            spec=spec,
            batch_dim=batch_dim,
            batch_position=batch_position,
            label=label,
            owner=owner,
        )
        out[spec.name] = coord.expand_dims({batch_dim: [label]}).variable
    return out


def _resolved_child_coord(
    child: xr.DataTree,
    *,
    ds: xr.Dataset,
    spec: BatchCoordSpec,
    batch_dim: str,
    batch_position: int,
    label: str,
    owner: str,
) -> xr.DataArray:
    local = child.to_dataset(inherit=False).coords.get(spec.name)
    if local is not None:
        _require_compatible_coord_dims(local, spec=spec, label=label, owner=owner)
        return _coord_in_template_order(local, spec=spec)
    coord = ds.coords.get(spec.name)
    if coord is not None:
        ordered = _coord_in_template_order(coord, spec=spec)
        if ordered is not None:
            return ordered
    parent = child.parent
    if parent is not None:
        coord = parent.to_dataset(inherit=False).coords.get(spec.name)
        if coord is not None and batch_dim in coord.dims:
            coord = coord.isel({batch_dim: batch_position}, drop=True)
        if coord is not None:
            ordered = _coord_in_template_order(coord, spec=spec)
            if ordered is not None:
                return ordered
    return _missing_child_coord(spec, label=label, owner=owner)


def _missing_child_coord(
    spec: BatchCoordSpec,
    *,
    label: str,
    owner: str,
) -> xr.DataArray:
    if getattr(spec.template.dtype, "fields", None) is not None:
        raise ValueError(
            f"{owner}: coordinate {spec.name!r} is missing from DataTree group {label!r}, "
            f"and structured dtype {spec.template.dtype!r} has no unambiguous missing value; "
            "define the coordinate in every group or provide a compatible root fallback."
        )
    template = _concat_safe_coord(spec.template)
    return template.where(xr.zeros_like(template, dtype=bool))


def _concat_safe_coord(coord: xr.DataArray) -> xr.DataArray:
    if isinstance(coord.dtype, np.dtype):
        return coord
    return coord.astype(object)


def _coord_in_template_order(
    coord: xr.DataArray,
    *,
    spec: BatchCoordSpec,
) -> xr.DataArray | None:
    if set(coord.dims) != set(spec.template.dims):
        return None
    if coord.dims == spec.template.dims:
        return _concat_safe_coord(coord)
    return _concat_safe_coord(coord.transpose(*spec.template.dims))


__all__ = [
    "BatchCoordSpec",
    "canonicalize_selected_payload_vars",
    "concat_datatree_rows",
    "expand_empty_datatree_template",
    "expand_datatree_rows",
    "plan_datatree_child_payloads",
    "structural_scalar_metadata_coord_names",
]
