from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.base import is_dask_collection

from tal.core import AnalysisObject, SchemaError
from tal.core.dataset_utils import ensure_dataset
from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.core.schema import merge_schema

from .adapter_paths import (
    ExportDestinationPreflight, existing_filesystem_identity, filesystem_collision_key,
    normalize_export_label_path, resolve_export_root, snapshot_export_destinations,
    stringify_export_identity,
)
from .csv_commit import (
    CsvCommitPlan, commit_csv_export, discard_csv_commit,
    prepare_csv_commit,
)
from .csv_export_context import CsvExportContext, resolve_csv_export_context
from .csv_sizes import normalize_csv_export_sizes, require_csv_export_size_dtype
from .csv_validation import register_csv_export_name
from .options import CsvExportOptions


@dataclass(frozen=True)
class _CsvExportField:
    output_name: str
    source_name: object
    is_coord: bool


@dataclass(frozen=True)
class _CsvExportPlan:
    paths: tuple[str, ...]
    path_preflight: tuple[ExportDestinationPreflight, ...]
    fields: tuple[_CsvExportField, ...]
    batch_dim: str
    sequence_dim: str
    lengths: tuple[int, ...]
    float_format: str | None


def _external_csv_export_dataset(value: xr.Dataset | xr.DataArray) -> xr.Dataset:
    ds = ensure_dataset(value)
    if not isinstance(value, xr.DataArray) or "tal" not in value.attrs:
        return ds
    return merge_schema(ds, value.attrs["tal"], validate=False)


def coerce_csv_export_source(value: object, *, owner: str) -> AnalysisObject:
    """Coerce a CSV export source without copying external array payloads.

    Existing AnalysisObject instances retain their lifecycle and identity.
    External xarray inputs receive only a shallow dataset wrapper so export
    options can choose the effective validity coordinate before values are
    materialized.
    """
    if isinstance(value, AnalysisObject):
        return coerce_analysis_object_input(value, owner=owner)
    if not isinstance(value, (xr.Dataset, xr.DataArray)):
        expected = "AnalysisObject, xr.Dataset, or xr.DataArray"
        raise TypeError(f"{owner}: expected {expected}; got {type(value).__name__}.")
    try:
        ds = _external_csv_export_dataset(value)
        return AnalysisObject._from_unvalidated(ds)
    except SchemaError as exc:
        raise ValueError(f"{owner}: invalid AnalysisObject schema payload.") from exc
    except Exception as exc:
        raise ValueError(f"{owner}: invalid xarray AnalysisObject input.") from exc


def _materialize_batch_labels(
    ds: xr.Dataset, *, batch_dim: str, owner: str
) -> tuple[object, ...]:
    """Materialize labels at the boundary where Python output paths are required."""
    coord = ds.coords[batch_dim]
    if tuple(coord.dims) != (batch_dim,):
        raise ValueError(
            f"{owner}: batch coordinate {batch_dim!r} must have dims ({batch_dim!r},); "
            f"got {tuple(coord.dims)!r}."
        )
    try:
        values = np.asarray(coord.variable.compute().data)
    except Exception as exc:  # pragma: no cover - backend read failure envelope.
        raise ValueError(f"{owner}: failed reading batch coordinate {batch_dim!r}.") from exc
    return tuple(values)


def _require_unique_batch_labels(labels: Sequence[object], *, owner: str) -> None:
    try:
        duplicated = pd.Index(labels).duplicated()
    except Exception as exc:
        raise ValueError(f"{owner}: batch labels must support deterministic equality checks.") from exc
    indices = np.flatnonzero(duplicated)
    if indices.size:
        label = labels[int(indices[0])]
        normalized = stringify_export_identity(
            label,
            owner=owner,
            description="duplicate batch label",
        )
        raise ValueError(
            f"{owner}: duplicate batch label {normalized!r} is not supported for grouped CSV export."
        )


def _require_export_size_coord(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    size_name: str | None,
    owner: str,
) -> None:
    if size_name is None:
        return
    if size_name == batch_dim:
        raise ValueError(f"{owner}: sequence_size_coord cannot reuse batch coordinate {batch_dim!r}.")
    if size_name not in ds.coords:
        raise ValueError(f"{owner}: sequence_size_coord {size_name!r} was not found in dataset coords.")
    size = ds.coords[size_name]
    dims = tuple(size.dims)
    if dims != (batch_dim,):
        raise ValueError(
            f"{owner}: sequence_size_coord {size_name!r} must have dims ({batch_dim!r},); "
            f"got {dims!r}."
        )
    require_csv_export_size_dtype(size, size_name=size_name, owner=owner)


def _resolve_valid_lengths(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    sequence_dim: str,
    size_name: str | None,
    owner: str,
) -> tuple[int, ...]:
    width = int(ds.sizes[sequence_dim])
    batch_size = int(ds.sizes[batch_dim])
    if batch_size == 0:
        return ()
    if size_name is None:
        return (width,) * batch_size
    try:
        size_variable = ds.coords[size_name].variable.compute()
    except Exception as exc:  # pragma: no cover - backend read failure envelope.
        raise ValueError(
            f"{owner}: failed reading sequence_size_coord {size_name!r}."
        ) from exc
    return _normalize_valid_lengths(
        size_variable,
        width=width,
        size_name=size_name,
        owner=owner,
    )


def _normalize_valid_lengths(
    size_variable: xr.Variable,
    *,
    width: int,
    size_name: str,
    owner: str,
) -> tuple[int, ...]:
    normalized = normalize_csv_export_sizes(
        xr.DataArray(size_variable),
        width=width,
        size_name=size_name,
        owner=owner,
    )
    return tuple(int(value) for value in np.asarray(normalized.data).flat)


def _preflight_in_memory_valid_lengths(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    sequence_dim: str,
    size_name: str | None,
    owner: str,
) -> tuple[int, ...] | None:
    """Validate resident sizes now while deferring backend reads until paths exist."""
    width = int(ds.sizes[sequence_dim])
    if size_name is None:
        return (width,) * int(ds.sizes[batch_dim])
    variable = ds.coords[size_name].variable
    if not variable._in_memory:
        return None
    return _normalize_valid_lengths(variable, width=width, size_name=size_name, owner=owner)


def _append_export_field(
    fields: list[_CsvExportField],
    names: set[str],
    *,
    raw_name: object,
    is_coord: bool,
    owner: str,
) -> None:
    output_name = stringify_export_identity(
        raw_name,
        owner=owner,
        description="CSV column name",
    )
    register_csv_export_name(output_name, seen=names, owner=owner)
    fields.append(_CsvExportField(output_name, raw_name, is_coord))


def _exclude_export_coord(
    name: object,
    sequence_dim: str,
    param_name: str | None,
    declared_size_name: str | None,
    size_name: str | None,
) -> bool:
    is_overridden_validity = (
        name == declared_size_name
        and declared_size_name != size_name
        and name != param_name
    )
    return name == size_name or is_overridden_validity or (
        name == sequence_dim and name != param_name
    )


def _resolve_export_fields(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    sequence_dim: str,
    param_name: str | None,
    declared_size_name: str | None,
    size_name: str | None,
    owner: str,
) -> tuple[_CsvExportField, ...]:
    fields: list[_CsvExportField] = []
    names: set[str] = set()
    for name, coord in ds.coords.items():
        if _exclude_export_coord(
            name, sequence_dim, param_name, declared_size_name, size_name
        ):
            continue
        row_dims = tuple(dim for dim in coord.dims if dim != batch_dim)
        if row_dims == (sequence_dim,):
            _append_export_field(fields, names, raw_name=name, is_coord=True, owner=owner)
            continue
        if row_dims:
            output_name = stringify_export_identity(
                name,
                owner=owner,
                description="CSV coordinate name",
            )
            raise ValueError(
                f"{owner}: coordinate {output_name!r} is not representable for grouped CSV export; "
                f"dims={row_dims!r} after batch selection."
            )
    for name, arr in ds.data_vars.items():
        row_dims = tuple(dim for dim in arr.dims if dim != batch_dim)
        if row_dims != (sequence_dim,):
            output_name = stringify_export_identity(
                name,
                owner=owner,
                description="CSV variable name",
            )
            raise ValueError(
                f"{owner}: variable {output_name!r} is not representable for grouped CSV export; "
                f"dims={row_dims!r} after batch selection."
            )
        _append_export_field(fields, names, raw_name=name, is_coord=False, owner=owner)
    if not fields:
        raise ValueError(f"{owner}: grouped CSV export requires at least one exportable sequence field.")
    return tuple(fields)


def _resolve_export_paths(
    labels: Sequence[object],
    *,
    root: Path,
    owner: str,
) -> tuple[tuple[str, ...], tuple[ExportDestinationPreflight, ...]]:
    paths: list[str] = []
    seen: set[str] = set()
    existing_identities: set[tuple[int, int]] = set()
    for label in labels:
        path = normalize_export_label_path(label, root=root, owner=owner)
        key = filesystem_collision_key(path)
        if key in seen:
            raise ValueError(f"{owner}: normalized export path collision at path {path!r}.")
        identity = existing_filesystem_identity(path, owner=owner)
        if identity is not None and identity in existing_identities:
            raise ValueError(f"{owner}: existing export destinations alias one filesystem file at {path!r}.")
        seen.add(key)
        if identity is not None:
            existing_identities.add(identity)
        paths.append(path)
    resolved = tuple(paths)
    _require_export_plan_topology(resolved, owner=owner)
    preflight = snapshot_export_destinations(resolved, root=root, owner=owner)
    return resolved, preflight


def _planned_destination_parent(raw_path: str, *, destination_keys: set[str]) -> Path | None:
    return next(
        (
            parent
            for parent in Path(raw_path).parents
            if filesystem_collision_key(str(parent)) in destination_keys
        ),
        None,
    )


def _require_export_plan_topology(paths: Sequence[str], *, owner: str) -> None:
    destination_keys = {filesystem_collision_key(path) for path in paths}
    for raw_path in paths:
        parent = _planned_destination_parent(raw_path, destination_keys=destination_keys)
        if parent is None:
            continue
        raise ValueError(
            f"{owner}: export destination {raw_path!r} has another planned destination "
            f"as parent path {str(parent)!r}."
        )


def _lower_export_field(
    ds: xr.Dataset,
    *,
    field: _CsvExportField,
    batch_dim: str,
    sequence_dim: str,
    row: int,
    valid_length: int,
) -> xr.Variable:
    source = _export_field_source(ds, field=field)
    indexers: dict[str, object] = {sequence_dim: slice(0, valid_length)}
    if batch_dim in source.dims:
        indexers[batch_dim] = row
    return source.variable.isel(indexers)


def _export_field_source(ds: xr.Dataset, *, field: _CsvExportField) -> xr.DataArray:
    return ds.coords[field.source_name] if field.is_coord else ds[field.source_name]


def _materialize_export_variable(variable: xr.Variable) -> np.ndarray:
    """Realize one valid field slice inside the coordinated Dask graph."""
    return np.asarray(variable.data)


def _slice_export_value(value: np.ndarray, valid_length: int) -> np.ndarray:
    return value[:valid_length]


def _slice_export_task(value: object, valid_length: int) -> object:
    if is_dask_collection(value):
        return delayed(_slice_export_value)(value, valid_length)
    return _slice_export_value(np.asarray(value), valid_length)


def _export_value_task(
    ds: xr.Dataset,
    *,
    plan: _CsvExportPlan,
    field: _CsvExportField,
    row: int,
    valid_length: int,
) -> object:
    if valid_length == 0:
        return np.empty(0, dtype=object)
    variable = _lower_export_field(
        ds,
        field=field,
        batch_dim=plan.batch_dim,
        sequence_dim=plan.sequence_dim,
        row=row,
        valid_length=valid_length,
    )
    if is_dask_collection(variable) or variable._in_memory:
        return variable.data
    return delayed(_materialize_export_variable)(variable)


def _row_export_tasks(
    ds: xr.Dataset,
    *,
    plan: _CsvExportPlan,
    row: int,
    valid_length: int,
    invariant: dict[int, object],
    invariant_length: int,
) -> tuple[object, ...]:
    tasks: list[object] = []
    for index, field in enumerate(plan.fields):
        source = _export_field_source(ds, field=field)
        if plan.batch_dim in source.dims:
            tasks.append(
                _export_value_task(
                    ds,
                    plan=plan,
                    field=field,
                    row=row,
                    valid_length=valid_length,
                )
            )
            continue
        if index not in invariant:
            invariant[index] = _export_value_task(
                ds,
                plan=plan,
                field=field,
                row=0,
                valid_length=invariant_length,
            )
        task = invariant[index]
        tasks.append(
            task
            if valid_length == invariant_length
            else _slice_export_task(task, valid_length)
        )
    return tuple(tasks)


def _lower_export_rows(
    ds: xr.Dataset,
    *,
    plan: _CsvExportPlan,
) -> tuple[tuple[object, ...], ...]:
    invariant: dict[int, object] = {}
    invariant_length = max(plan.lengths, default=0)
    return tuple(
        _row_export_tasks(
            ds,
            plan=plan,
            row=row,
            valid_length=valid_length,
            invariant=invariant,
            invariant_length=invariant_length,
        )
        for row, valid_length in enumerate(plan.lengths)
    )


def _serialize_csv_row(
    values: tuple[object, ...],
    *,
    fields: tuple[_CsvExportField, ...],
    raw_path: str,
    float_format: str | None,
) -> str:
    columns = {
        field.output_name: np.asarray(value)
        for field, value in zip(fields, values, strict=True)
    }
    pd.DataFrame(columns).to_csv(
        raw_path, index=False, float_format=float_format, encoding="utf-8"
    )
    return raw_path


def _serialize_export_rows(
    ds: xr.Dataset,
    plan: _CsvExportPlan,
    *,
    temporary_paths: tuple[str, ...],
    owner: str,
) -> tuple[str, ...]:
    """Serialize eager rows directly or jointly realize one lazy graph."""
    try:
        rows = _lower_export_rows(ds, plan=plan)
        if len(rows) != len(temporary_paths):
            raise RuntimeError("CSV row/temporary-path count mismatch")
        items = tuple(zip(rows, temporary_paths, strict=True))
        if not any(is_dask_collection(value) for row in rows for value in row):
            return tuple(
                _serialize_csv_row(
                    values,
                    fields=plan.fields,
                    raw_path=raw_path,
                    float_format=plan.float_format,
                )
                for values, raw_path in items
            )
        realized_rows = tuple(compute(*rows))
        return tuple(
            _serialize_csv_row(
                values,
                fields=plan.fields,
                raw_path=raw_path,
                float_format=plan.float_format,
            )
            for values, raw_path in zip(
                realized_rows, temporary_paths, strict=True
            )
        )
    except Exception as exc:
        raise ValueError(f"{owner}: failed serializing CSV export rows.") from exc


def _prepare_csv_export_commits(
    ds: xr.Dataset,
    plan: _CsvExportPlan,
    *,
    owner: str,
) -> CsvCommitPlan:
    staging = prepare_csv_commit(
        plan.paths,
        preflight=plan.path_preflight,
        owner=owner,
    )
    try:
        _serialize_export_rows(
            ds,
            plan,
            temporary_paths=staging.temporary_paths,
            owner=owner,
        )
    except BaseException:
        discard_csv_commit(staging)
        raise
    return staging


def _plan_csv_export(
    context: CsvExportContext,
    *,
    root: Path,
    float_format: str | None,
    owner: str,
) -> _CsvExportPlan:
    ds = context.ds
    batch_dim = context.batch_dim
    sequence_dim = context.sequence_dim
    size_name = context.size_name
    _require_export_size_coord(ds, batch_dim=batch_dim, size_name=size_name, owner=owner)
    fields = _resolve_export_fields(
        ds,
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
        param_name=context.param_name,
        declared_size_name=context.declared_size_name,
        size_name=size_name,
        owner=owner,
    )
    lengths = _preflight_in_memory_valid_lengths(
        ds,
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
        size_name=size_name,
        owner=owner,
    )
    labels = _materialize_batch_labels(ds, batch_dim=batch_dim, owner=owner)
    _require_unique_batch_labels(labels, owner=owner)
    paths, path_preflight = _resolve_export_paths(labels, root=root, owner=owner)
    if lengths is None:
        lengths = _resolve_valid_lengths(
            ds,
            batch_dim=batch_dim,
            sequence_dim=sequence_dim,
            size_name=size_name,
            owner=owner,
        )
    return _CsvExportPlan(
        lengths=lengths,
        paths=paths,
        path_preflight=path_preflight,
        fields=fields,
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
        float_format=float_format,
    )


def execute_csv_export(
    ao: AnalysisObject,
    out_dir: str,
    *,
    options: CsvExportOptions,
    owner: str,
) -> tuple[str, ...]:
    root = resolve_export_root(out_dir, owner=owner)
    context = resolve_csv_export_context(
        ao,
        options=options,
        owner=owner,
    )
    plan = _plan_csv_export(
        context,
        root=root,
        float_format=options.float_format,
        owner=owner,
    )
    if not plan.paths:
        return plan.paths
    staging = _prepare_csv_export_commits(context.ds, plan, owner=owner)
    commit_csv_export(staging, owner=owner)
    return plan.paths


__all__ = ["coerce_csv_export_source", "execute_csv_export"]
