from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from tal.core import AnalysisObject
from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.core.validity_finalize import set_left_packed_validity_or_prune_from_size_coord

from .backends import datatree_children, validate_batch_dim_role_compatibility
from .metadata_domain import collect_extract_metadata_columns, promote_extract_metadata
from .options import CatalogExtractOptions, CatalogQueryOptions
from .types import CatalogState


def ensure_grouping_context_for_query_extract(state: CatalogState, *, owner: str) -> None:
    if state.backend != "dataset":
        return
    ds = state.data if isinstance(state.data, xr.Dataset) else _invalid_dataset_payload(state.data)
    declared, sequence_dim, batch_dims, core_dims = read_roles(ds)
    if not declared:
        return
    validate_batch_dim_role_compatibility(
        batch_dim=state.batch_dim,
        sequence_dim=sequence_dim,
        core_dims=core_dims,
        owner=owner,
    )
    if len(batch_dims) > 1:
        raise ValueError(
            f"{owner}: ambiguous grouping context for query/extract; declared batch_dims={batch_dims!r}, "
            f"catalog batch_dim={state.batch_dim!r}. Reconstruct catalog with unambiguous grouping."
        )
    if batch_dims and batch_dims != (state.batch_dim,):
        raise ValueError(
            f"{owner}: explicit batch_dim {state.batch_dim!r} conflicts with declared batch_dims "
            f"{batch_dims!r}."
        )


def extract_catalog_to_analysis_object(
    state: CatalogState,
    *,
    variables: tuple[str, ...] | None,
    options: CatalogExtractOptions,
    owner: str,
) -> AnalysisObject:
    ensure_grouping_context_for_query_extract(state, owner=owner)
    if state.backend == "dataset":
        source = state.data if isinstance(state.data, xr.Dataset) else _invalid_dataset_payload(state.data)
        payload = _extract_dataset_payload(source, variables=variables, options=options, owner=owner)
        return _finalize_extract(payload=payload, source_schema=source, state=state, options=options, owner=owner)
    tree = state.data if isinstance(state.data, xr.DataTree) else _invalid_datatree_payload(state.data)
    payload, schema_source = _extract_datatree_payload(
        tree,
        state=state,
        variables=variables,
        options=options,
        owner=owner,
    )
    return _finalize_extract(
        payload=payload,
        source_schema=schema_source,
        state=state,
        options=options,
        owner=owner,
    )


def _extract_dataset_payload(
    source: xr.Dataset,
    *,
    variables: tuple[str, ...] | None,
    options: CatalogExtractOptions,
    owner: str,
) -> xr.Dataset:
    selected = _resolve_selected_vars(
        available=tuple(source.data_vars),
        requested=variables,
        ignore_missing=options.ignore_missing_vars,
        owner=owner,
    )
    return _select_vars(source, selected)


def _extract_datatree_payload(
    tree: xr.DataTree,
    *,
    state: CatalogState,
    variables: tuple[str, ...] | None,
    options: CatalogExtractOptions,
    owner: str,
) -> tuple[xr.Dataset, xr.Dataset]:
    children = datatree_children(tree)
    labels = tuple(children.keys())
    if not children:
        return _extract_empty_datatree_payload(
            state=state,
            variables=variables,
            options=options,
            owner=owner,
        )
    ordered_children = tuple(children[label].ds for label in labels)
    available = _shared_child_vars(ordered_children)
    selected = _resolve_selected_vars(
        available=available,
        requested=variables,
        ignore_missing=options.ignore_missing_vars,
        owner=owner,
    )
    rows = [
        _expand_child_row(
            _select_vars(ds, selected),
            batch_dim=state.batch_dim,
            label=label,
            owner=owner,
        )
        for ds, label in zip(ordered_children, labels, strict=True)
    ]
    merged = xr.concat(rows, dim=state.batch_dim, join="exact", compat="identical", combine_attrs="drop_conflicts")
    merged = merged.assign_coords({state.batch_dim: xr.DataArray(np.asarray(labels, dtype=object), dims=(state.batch_dim,))})
    merged = _strip_dataset_attrs(merged)
    metadata = collect_extract_metadata_columns(state, owner=owner, options=CatalogQueryOptions())
    promoted = promote_extract_metadata(
        merged,
        batch_dim=state.batch_dim,
        metadata=metadata,
        options=options.metadata_promotion,
        owner=owner,
    )
    return promoted, ordered_children[0]


def _extract_empty_datatree_payload(
    *,
    state: CatalogState,
    variables: tuple[str, ...] | None,
    options: CatalogExtractOptions,
    owner: str,
) -> tuple[xr.Dataset, xr.Dataset]:
    if state.template is None:
        if variables not in (None, ()):
            raise ValueError(
                f"{owner}: cannot extract requested variables from empty DataTree without template metadata."
            )
        out = xr.Dataset(coords={state.batch_dim: xr.DataArray(np.asarray([], dtype=object), dims=(state.batch_dim,))})
        return out, xr.Dataset()
    template = state.template
    selected = _resolve_selected_vars(
        available=tuple(template.data_vars),
        requested=variables,
        ignore_missing=options.ignore_missing_vars,
        owner=owner,
    )
    base = _select_vars(template, selected)
    expanded = base.expand_dims({state.batch_dim: np.asarray([], dtype=object)})
    size_name = read_sequence_size_coord_name(template)
    if size_name is not None and size_name in expanded.coords and expanded.coords[size_name].dims == ():
        expanded = expanded.drop_vars(size_name)
        expanded = expanded.assign_coords(
            {
                size_name: xr.DataArray(
                    np.asarray([], dtype=np.int64),
                    dims=(state.batch_dim,),
                )
            }
        )
    return expanded, template


def _shared_child_vars(children: Sequence[xr.Dataset]) -> tuple[str, ...]:
    first = tuple(children[0].data_vars)
    shared = set(first)
    for ds in children[1:]:
        shared.intersection_update(ds.data_vars.keys())
    return tuple(name for name in first if name in shared)


def _strip_dataset_attrs(ds: xr.Dataset) -> xr.Dataset:
    if not ds.attrs:
        return ds
    out = ds.copy(deep=False)
    out.attrs = {}
    return out


def _resolve_selected_vars(
    *,
    available: tuple[str, ...],
    requested: tuple[str, ...] | None,
    ignore_missing: bool,
    owner: str,
) -> tuple[str, ...]:
    if requested is None:
        return available
    unknown = tuple(name for name in requested if name not in available)
    if unknown and not ignore_missing:
        raise ValueError(
            f"{owner}: unknown extract variable(s) {unknown!r}; set ignore_missing_vars=True to ignore."
        )
    return tuple(name for name in requested if name in available)


def _expand_child_row(ds: xr.Dataset, *, batch_dim: str, label: str, owner: str) -> xr.Dataset:
    try:
        return ds.expand_dims({batch_dim: [label]})
    except ValueError as exc:
        raise ValueError(
            f"{owner}: explicit batch_dim {batch_dim!r} collides with child payload dims for group "
            f"{label!r}; choose a non-colliding constructor batch_dim."
        ) from exc


def _select_vars(ds: xr.Dataset, selected: tuple[str, ...]) -> xr.Dataset:
    if not selected:
        return ds.drop_vars(list(ds.data_vars), errors="ignore")
    return ds[list(selected)]


def _finalize_extract(
    *,
    payload: xr.Dataset,
    source_schema: xr.Dataset,
    state: CatalogState,
    options: CatalogExtractOptions,
    owner: str,
) -> AnalysisObject:
    source_ao = AnalysisObject._from_unvalidated(source_schema)
    spec = _resolve_extract_schema_spec(
        source_schema,
        payload=payload,
        batch_dim=state.batch_dim,
        require_sequence_size_coord=options.require_sequence_size_coord,
        owner=owner,
    )
    finalized = finalize_with_schema(
        source_ao,
        payload,
        spec=spec,
        validate=options.validate,
        owner=owner,
    )
    if spec.size_name is None:
        return finalized
    return set_left_packed_validity_or_prune_from_size_coord(
        finalized,
        size_name=spec.size_name,
        validate=options.validate,
        owner=owner,
    )


def _resolve_extract_schema_spec(
    source: xr.Dataset,
    *,
    payload: xr.Dataset,
    batch_dim: str,
    require_sequence_size_coord: bool,
    owner: str,
) -> CoreSchemaFinalizeSpec:
    roles_declared, sequence_dim, batch_dims, core_dims = read_roles(source)
    if not roles_declared or sequence_dim is None:
        if require_sequence_size_coord:
            raise ValueError(
                f"{owner}: require_sequence_size_coord=True requires declared roles/sequence_dim support."
            )
        return CoreSchemaFinalizeSpec(
            sequence_dim=None,
            batch_dims=(),
            core_dims=(),
            param_name=None,
            size_name=None,
        )
    final_batch_dims = _resolve_output_batch_dims(
        batch_dims=batch_dims,
        sequence_dim=sequence_dim,
        core_dims=core_dims,
        batch_dim=batch_dim,
        payload=payload,
        owner=owner,
    )
    param_name = _coord_if_present(payload, read_param_coord_name(source))
    size_name = _validity_coord_if_present(
        payload,
        name=read_sequence_size_coord_name(source),
        batch_dims=final_batch_dims,
    )
    if require_sequence_size_coord and size_name is None:
        raise ValueError(
            f"{owner}: require_sequence_size_coord=True but source did not provide usable sequence_size_coord support."
        )
    return CoreSchemaFinalizeSpec(
        sequence_dim=sequence_dim,
        batch_dims=final_batch_dims,
        core_dims=core_dims,
        param_name=param_name,
        size_name=size_name,
    )


def _resolve_output_batch_dims(
    *,
    batch_dims: tuple[str, ...],
    sequence_dim: str,
    core_dims: tuple[str, ...],
    batch_dim: str,
    payload: xr.Dataset,
    owner: str,
) -> tuple[str, ...]:
    if batch_dim not in payload.dims:
        raise ValueError(
            f"{owner}: constructor-resolved batch_dim {batch_dim!r} is not present in extract payload dims."
        )
    validate_batch_dim_role_compatibility(
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
        core_dims=core_dims,
        owner=owner,
    )
    if len(batch_dims) > 1:
        raise ValueError(
            f"{owner}: ambiguous grouping context for extract; source batch_dims={batch_dims!r}, catalog batch_dim={batch_dim!r}."
        )
    if batch_dims:
        if batch_dims != (batch_dim,):
            raise ValueError(
                f"{owner}: explicit batch_dim {batch_dim!r} conflicts with declared batch_dims {batch_dims!r}."
            )
        return batch_dims
    return (batch_dim,)


def _coord_if_present(ds: xr.Dataset, name: str | None) -> str | None:
    if name is None:
        return None
    return name if name in ds.coords else None


def _validity_coord_if_present(
    ds: xr.Dataset,
    *,
    name: str | None,
    batch_dims: tuple[str, ...],
) -> str | None:
    if name is None or name not in ds.coords:
        return None
    coord = ds.coords[name]
    return name if coord.dims == batch_dims else None


def _invalid_dataset_payload(payload: xr.Dataset | xr.DataTree) -> xr.Dataset:
    raise TypeError(f"catalog extract internal error: expected xr.Dataset; got {type(payload).__name__}.")


def _invalid_datatree_payload(payload: xr.Dataset | xr.DataTree) -> xr.DataTree:
    raise TypeError(f"catalog extract internal error: expected xr.DataTree; got {type(payload).__name__}.")


__all__ = [
    "ensure_grouping_context_for_query_extract",
    "extract_catalog_to_analysis_object",
]
