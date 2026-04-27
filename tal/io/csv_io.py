from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import xarray as xr

from tal.core.orchestration.context import DatasetContextOptions, resolve_dataset_context
from tal.core.schema import merge_schema

from .finalize import finalize_loaded_dataset
from .metadata import read_sidecar_metadata, write_sidecar_metadata
from .options import AOCsvReadOptions, AOCsvWriteOptions, coerce_csv_read_options, coerce_csv_write_options


def _python_scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _resolve_sequence_dim(ao: "AnalysisObject", *, owner: str) -> str:
    context = resolve_dataset_context(
        ao,
        owner=owner,
        options=DatasetContextOptions(require_roles=True, require_sequence_dim=True),
    )
    if context.sequence_dim is None:
        raise ValueError(f"{owner}: AO-direct CSV requires declared sequence_dim.")
    return context.sequence_dim


def _require_single_sequence_shape(ds: xr.Dataset, *, sequence_dim: str, owner: str) -> None:
    extra_dims = tuple(dim for dim in ds.dims if dim != sequence_dim)
    if extra_dims:
        raise ValueError(
            f"{owner}: AO-direct CSV supports a single sequence dimension; "
            f"found extra dims {extra_dims!r}."
        )


def _sequence_coord_values(ds: xr.Dataset, *, sequence_dim: str) -> Any:
    if sequence_dim in ds.coords:
        return ds.coords[sequence_dim].values
    return list(range(ds.sizes[sequence_dim]))


def _collect_csv_payload(
    ds: xr.Dataset,
    *,
    sequence_dim: str,
    owner: str,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    columns: dict[str, Any] = {sequence_dim: _sequence_coord_values(ds, sequence_dim=sequence_dim)}
    data_vars: list[str] = []
    coords: list[str] = []
    scalar_coords: dict[str, Any] = {}
    for name in ds.data_vars:
        arr = ds[name]
        if arr.dims != (sequence_dim,):
            raise ValueError(
                f"{owner}: data variable {name!r} is not representable as 1-D over {sequence_dim!r}; "
                f"dims={tuple(arr.dims)!r}."
            )
        columns[name] = arr.values
        data_vars.append(str(name))
    for name, coord in ds.coords.items():
        if name == sequence_dim or name in ds.data_vars:
            continue
        if coord.dims == (sequence_dim,):
            columns[name] = coord.values
            coords.append(str(name))
            continue
        if coord.dims == ():
            scalar_coords[str(name)] = _python_scalar(coord.values)
            continue
        raise ValueError(
            f"{owner}: coordinate {name!r} is not representable in single-object CSV; "
            f"dims={tuple(coord.dims)!r}."
        )
    return columns, tuple(data_vars), tuple(coords), scalar_coords


def _schema_payload(ds: xr.Dataset, *, owner: str) -> dict[str, Any]:
    payload = ds.attrs.get("tal")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{owner}: AO-direct CSV requires schema mapping in ds.attrs['tal'].")
    return dict(payload)


def _first_duplicate_name(names: tuple[str, ...]) -> str | None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            return name
        seen.add(name)
    return None


def _normalize_name_container(value: Any, *, field_name: str, owner: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{owner}: metadata {field_name} must be a list/tuple of non-empty strings.")
    names: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"{owner}: metadata {field_name}[{idx}] must be a non-empty string; got {item!r}."
            )
        names.append(item)
    normalized = tuple(names)
    duplicate = _first_duplicate_name(normalized)
    if duplicate is not None:
        raise ValueError(f"{owner}: metadata {field_name} contains duplicate name {duplicate!r}.")
    return normalized


def _normalize_scalar_coords(value: Any, *, owner: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner}: metadata scalar_coords must be a mapping.")
    out: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{owner}: metadata scalar_coords keys must be non-empty strings; got {key!r}.")
        out[key] = item
    return out


def write_analysis_object_csv(
    ao: "AnalysisObject",
    path: str,
    *,
    opts: AOCsvWriteOptions | None,
    owner: str,
) -> str:
    options = coerce_csv_write_options(opts, owner=owner)
    if not isinstance(path, str) or not path:
        raise TypeError(f"{owner}: path must be a non-empty string.")
    sequence_dim = _resolve_sequence_dim(ao, owner=owner)
    ds = ao.unsafe_data
    _require_single_sequence_shape(ds, sequence_dim=sequence_dim, owner=owner)
    columns, data_vars, coords, scalar_coords = _collect_csv_payload(
        ds,
        sequence_dim=sequence_dim,
        owner=owner,
    )
    pd.DataFrame(columns).to_csv(path, index=False, float_format=options.float_format)
    metadata = {
        "version": 1,
        "sequence_dim": sequence_dim,
        "data_vars": data_vars,
        "coords": coords,
        "scalar_coords": scalar_coords,
        "tal_schema": _schema_payload(ds, owner=owner),
    }
    write_sidecar_metadata(path, metadata, metadata_path=options.metadata_path, owner=owner)
    return path


def _require_read_metadata(payload: dict[str, Any], *, owner: str) -> dict[str, object]:
    required = ("sequence_dim", "data_vars", "coords", "scalar_coords", "tal_schema")
    missing = tuple(key for key in required if key not in payload)
    if missing:
        raise ValueError(f"{owner}: metadata sidecar is missing required keys {missing!r}.")
    sequence_dim = payload["sequence_dim"]
    if not isinstance(sequence_dim, str) or not sequence_dim:
        raise ValueError(f"{owner}: metadata sequence_dim must be a non-empty string.")
    data_vars = _normalize_name_container(payload["data_vars"], field_name="data_vars", owner=owner)
    coords = _normalize_name_container(payload["coords"], field_name="coords", owner=owner)
    if sequence_dim in data_vars:
        raise ValueError(f"{owner}: metadata data_vars cannot contain sequence_dim {sequence_dim!r}.")
    if sequence_dim in coords:
        raise ValueError(f"{owner}: metadata coords cannot contain sequence_dim {sequence_dim!r}.")
    overlap = tuple(sorted(set(data_vars).intersection(coords)))
    if overlap:
        raise ValueError(f"{owner}: metadata data_vars/coords overlap is not allowed: {overlap!r}.")
    scalar_coords = _normalize_scalar_coords(payload["scalar_coords"], owner=owner)
    scalar_overlap = tuple(sorted(set(scalar_coords).intersection({sequence_dim, *data_vars, *coords})))
    if scalar_overlap:
        raise ValueError(
            f"{owner}: metadata scalar_coords collides with sequence/data/coord names: {scalar_overlap!r}."
        )
    if not isinstance(payload["tal_schema"], Mapping):
        raise ValueError(f"{owner}: metadata tal_schema must be a mapping.")
    return {
        "sequence_dim": sequence_dim,
        "data_vars": data_vars,
        "coords": coords,
        "scalar_coords": scalar_coords,
        "tal_schema": dict(payload["tal_schema"]),
    }


def _require_column_set(frame: pd.DataFrame, *, metadata: dict[str, object], owner: str) -> None:
    expected = {
        metadata["sequence_dim"],
        *tuple(metadata["data_vars"]),  # type: ignore[arg-type]
        *tuple(metadata["coords"]),  # type: ignore[arg-type]
    }
    actual = set(frame.columns)
    missing = tuple(sorted(expected - actual))
    extra = tuple(sorted(actual - expected))
    if missing or extra:
        raise ValueError(
            f"{owner}: CSV/metadata column mismatch; missing={missing!r}, extra={extra!r}."
        )


def _dataset_from_csv(frame: pd.DataFrame, *, metadata: dict[str, object]) -> xr.Dataset:
    sequence_dim = metadata["sequence_dim"]  # type: ignore[assignment]
    coords: dict[str, Any] = {sequence_dim: (sequence_dim, frame[sequence_dim].to_numpy())}
    data_vars = {
        str(name): (sequence_dim, frame[str(name)].to_numpy()) for name in tuple(metadata["data_vars"])  # type: ignore[arg-type]
    }
    for name in tuple(metadata["coords"]):  # type: ignore[arg-type]
        coords[str(name)] = (sequence_dim, frame[str(name)].to_numpy())
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    scalar_coords = metadata["scalar_coords"]
    if isinstance(scalar_coords, Mapping) and scalar_coords:
        ds = ds.assign_coords({str(name): _python_scalar(value) for name, value in scalar_coords.items()})
    return merge_schema(ds, dict(metadata["tal_schema"]), validate=False)


def read_analysis_object_csv(
    cls: type["AnalysisObject"],
    path: str,
    *,
    opts: AOCsvReadOptions | None,
    validate: bool,
    owner: str,
) -> "AnalysisObject":
    options = coerce_csv_read_options(opts, owner=owner)
    if not isinstance(path, str) or not path:
        raise TypeError(f"{owner}: path must be a non-empty string.")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas/engine specific envelope.
        raise ValueError(f"{owner}: failed reading CSV file {path!r}.") from exc
    metadata = _require_read_metadata(
        read_sidecar_metadata(path, metadata_path=options.metadata_path, owner=owner),
        owner=owner,
    )
    _require_column_set(frame, metadata=metadata, owner=owner)
    ds = _dataset_from_csv(frame, metadata=metadata)
    return finalize_loaded_dataset(cls, ds, validate=validate, owner=owner)


__all__ = ["read_analysis_object_csv", "write_analysis_object_csv"]
