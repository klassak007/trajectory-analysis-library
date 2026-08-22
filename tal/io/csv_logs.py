from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from tal.core.orchestration.context import DatasetContextOptions, resolve_dataset_context
from tal.core.orchestration.inputs import coerce_analysis_object_input

from .adapter_finalize import finalize_adapter_dataset
from .adapter_metadata import (
    aggregate_batch_metadata,
    collect_csv_scalar_metadata,
    promote_adapter_metadata,
)
from .adapter_paths import normalize_export_label_path, resolve_ingest_inputs
from .options import (
    CsvExportOptions,
    CsvIngestOptions,
    coerce_csv_export_options,
    coerce_csv_ingest_options,
)

_TIME_INFER_CANDIDATES = {"time", "timestamp", "t", "time_s", "time_sec"}


@dataclass(frozen=True)
class _CsvRecord:
    label: str
    source_path: str
    resolved_path: str
    time_col: str
    time_values: np.ndarray
    value_columns: tuple[str, ...]
    values: dict[str, np.ndarray]
    metadata: dict[str, object]


def _is_monotonic(values: np.ndarray, *, order: str) -> bool:
    if values.size <= 1:
        return True
    diffs = np.diff(values)
    if order == "strict":
        return bool(np.all(diffs > 0))
    return bool(np.all(diffs >= 0))


def _resolve_time_column(frame: pd.DataFrame, *, opts: CsvIngestOptions, owner: str, path: str) -> str:
    if opts.time_col is not None:
        if opts.time_col not in frame.columns:
            raise ValueError(f"{owner}: time_col {opts.time_col!r} not found in {path!r}.")
        return opts.time_col
    if not opts.allow_time_infer:
        raise ValueError(
            f"{owner}: time_col must be explicitly provided by default; set allow_time_infer=True "
            "to enable inference."
        )
    matches = [name for name in frame.columns if str(name).lower() in _TIME_INFER_CANDIDATES]
    if len(matches) != 1:
        raise ValueError(
            f"{owner}: time inference requires exactly one candidate from {_TIME_INFER_CANDIDATES!r}; "
            f"found {tuple(matches)!r} in {path!r}."
        )
    return str(matches[0])


def _coerce_time_column(
    frame: pd.DataFrame,
    *,
    time_col: str,
    opts: CsvIngestOptions,
    owner: str,
    path: str,
) -> pd.DataFrame:
    numeric = pd.to_numeric(frame[time_col], errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float, copy=False))
    if not bool(np.all(finite)):
        bad = int(finite.size - np.count_nonzero(finite))
        if opts.invalid_time == "fail":
            raise ValueError(
                f"{owner}: time column {time_col!r} in {path!r} contains {bad} non-finite values."
            )
        frame = frame.loc[finite].copy(deep=False)
        numeric = numeric.loc[finite]
        if frame.empty:
            raise ValueError(f"{owner}: all rows were dropped in {path!r} due to non-finite time values.")
    out = frame.copy(deep=False)
    out[time_col] = numeric.astype(float)
    if opts.sort_time:
        out = out.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    return out


def _apply_monotonic_policy(
    frame: pd.DataFrame,
    *,
    time_col: str,
    opts: CsvIngestOptions,
    owner: str,
    path: str,
) -> pd.DataFrame:
    values = frame[time_col].to_numpy(dtype=float, copy=False)
    if _is_monotonic(values, order=opts.monotonic_order):
        return frame
    if not opts.allow_nonmonotonic_normalize:
        raise ValueError(
            f"{owner}: time column {time_col!r} in {path!r} violates monotonic_order={opts.monotonic_order!r}."
        )
    sorted_frame = frame.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    sorted_values = sorted_frame[time_col].to_numpy(dtype=float, copy=False)
    if not _is_monotonic(sorted_values, order=opts.monotonic_order):
        raise ValueError(
            f"{owner}: time normalization could not satisfy monotonic_order={opts.monotonic_order!r} in {path!r}."
        )
    return sorted_frame


def _resolve_value_columns(
    frame: pd.DataFrame,
    *,
    time_col: str,
    opts: CsvIngestOptions,
    owner: str,
    path: str,
) -> tuple[str, ...]:
    if opts.value_columns is not None:
        columns = opts.value_columns
    else:
        excluded = {time_col, *opts.metadata_columns}
        columns = tuple(
            str(name)
            for name in frame.columns
            if str(name) not in excluded and pd.api.types.is_numeric_dtype(frame[name])
        )
    if not columns:
        raise ValueError(f"{owner}: no numeric value columns were selected for {path!r}.")
    missing = tuple(name for name in columns if name not in frame.columns)
    if missing:
        raise ValueError(f"{owner}: value columns {missing!r} were not found in {path!r}.")
    collisions = tuple(
        name
        for name in columns
        if name in {opts.batch_dim, opts.sequence_dim, opts.sequence_size_coord, opts.param_coord}
    )
    if collisions:
        raise ValueError(
            f"{owner}: value columns collide with reserved semantic names: {collisions!r}."
        )
    return tuple(columns)


def _coerce_numeric_value_column(
    frame: pd.DataFrame,
    *,
    name: str,
    owner: str,
    path: str,
) -> np.ndarray:
    try:
        numeric = pd.to_numeric(frame[name], errors="raise")
    except Exception as exc:
        raise ValueError(f"{owner}: value column {name!r} in {path!r} contains non-numeric values.") from exc
    return numeric.to_numpy(dtype=float, copy=False)


def _read_csv_record(
    path_info: Any,
    *,
    opts: CsvIngestOptions,
    owner: str,
) -> _CsvRecord:
    path = str(path_info.path)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas engine envelope.
        raise ValueError(f"{owner}: failed reading CSV file {path!r}.") from exc
    time_col = _resolve_time_column(frame, opts=opts, owner=owner, path=path)
    frame = _coerce_time_column(frame, time_col=time_col, opts=opts, owner=owner, path=path)
    frame = _apply_monotonic_policy(frame, time_col=time_col, opts=opts, owner=owner, path=path)
    value_columns = _resolve_value_columns(frame, time_col=time_col, opts=opts, owner=owner, path=path)
    values = {name: _coerce_numeric_value_column(frame, name=name, owner=owner, path=path) for name in value_columns}
    metadata = collect_csv_scalar_metadata(
        frame,
        metadata_columns=opts.metadata_columns,
        owner=owner,
        source_path=path,
    )
    return _CsvRecord(
        label=path_info.label,
        source_path=path,
        resolved_path=path_info.resolved_path,
        time_col=time_col,
        time_values=frame[time_col].to_numpy(dtype=float, copy=False),
        value_columns=value_columns,
        values=values,
        metadata=metadata,
    )


def _records_to_dataset(
    records: Sequence[_CsvRecord],
    *,
    opts: CsvIngestOptions,
    owner: str,
) -> xr.Dataset:
    first_columns = records[0].value_columns
    for record in records[1:]:
        if record.value_columns != first_columns:
            raise ValueError(
                f"{owner}: value-column mismatch across inputs; expected {first_columns!r}, "
                f"got {record.value_columns!r} for {record.source_path!r}."
            )
    labels = [record.label for record in records]
    sizes, time_grid, var_arrays = _build_record_grids(records, value_columns=first_columns)
    width = int(time_grid.shape[1])
    ds = xr.Dataset(
        data_vars={
            name: ((opts.batch_dim, opts.sequence_dim), values)
            for name, values in var_arrays.items()
        },
        coords={
            opts.batch_dim: np.asarray(labels, dtype=object),
            opts.sequence_dim: np.arange(width, dtype=np.int64),
            opts.param_coord: ((opts.batch_dim, opts.sequence_dim), time_grid),
            opts.sequence_size_coord: (opts.batch_dim, sizes),
        },
    )
    batch_metadata = _build_batch_metadata(records)
    return promote_adapter_metadata(
        ds,
        batch_dim=opts.batch_dim,
        metadata=batch_metadata,
        options=opts.metadata_promotion,
        owner=owner,
    )


def _build_record_grids(
    records: Sequence[_CsvRecord],
    *,
    value_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    batch = len(records)
    width = max(record.time_values.size for record in records)
    sizes = np.asarray([record.time_values.size for record in records], dtype=np.int64)
    time_grid = np.full((batch, width), np.nan, dtype=float)
    var_arrays = {
        name: np.full((batch, width), np.nan, dtype=float)
        for name in value_columns
    }
    for row, record in enumerate(records):
        size = int(record.time_values.size)
        time_grid[row, :size] = record.time_values
        for name in value_columns:
            var_arrays[name][row, :size] = record.values[name]
    return sizes, time_grid, var_arrays


def _build_batch_metadata(records: Sequence[_CsvRecord]) -> dict[str, object]:
    metadata = aggregate_batch_metadata([record.metadata for record in records])
    time_cols = [record.time_col for record in records]
    if all(time_cols[0] == value for value in time_cols[1:]):
        metadata["io_time_source"] = time_cols[0]
    else:
        metadata["io_time_source"] = list(time_cols)
    metadata["io_source_paths"] = [record.resolved_path for record in records]
    return metadata


def read_csv_logs(
    inputs: str | Sequence[str] | Mapping[str, str],
    *,
    opts: CsvIngestOptions | None = None,
    validate: bool = True,
):
    """Read CSV log files into a TAL AnalysisObject.

    Parameters
    ----------
    inputs : str | Sequence[str] | Mapping[str, str]
        Input paths/mapping consumed by ingestion.
    opts : tal.io.CsvIngestOptions or None
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``CsvIngestOptions`` key fields: ``time_col`` (default None), ``allow_time_infer`` (default False), ``sort_time`` (default True), ``monotonic_order`` (default 'nondecreasing').
    validate : bool
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    object
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from tal.io import CsvIngestOptions, read_csv_logs
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir) / "run.csv"
    ...     _ = path.write_text("time,value\\n0.0,1.0\\n1.0,2.0\\n", encoding="utf-8")
    ...     ao = read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))
    >>> ao.unsafe_data["value"].values.tolist()
    [[1.0, 2.0]]
    """
    owner = "tal.io.read_csv_logs"
    options = coerce_csv_ingest_options(opts, owner=owner)
    path_infos = resolve_ingest_inputs(inputs, owner=owner)
    records = [_read_csv_record(path_info, opts=options, owner=owner) for path_info in path_infos]
    ds = _records_to_dataset(records, opts=options, owner=owner)
    return finalize_adapter_dataset(
        ds,
        batch_dim=options.batch_dim,
        sequence_dim=options.sequence_dim,
        size_name=options.sequence_size_coord,
        param_name=options.param_coord,
        validate=validate,
        owner=owner,
    )


def _raw_label_key(label: object) -> tuple[str, object]:
    try:
        if bool(np.isnan(label)):  # type: ignore[arg-type]
            return ("nan", "nan")
    except Exception:
        pass
    return ("value", repr(label))


def _require_unique_batch_labels(labels: Sequence[object], *, owner: str) -> None:
    seen: set[tuple[str, object]] = set()
    for label in labels:
        key = _raw_label_key(label)
        if key in seen:
            raise ValueError(
                f"{owner}: duplicate batch label {label!r} is not supported for grouped CSV export."
            )
        seen.add(key)


def _resolve_export_context(ao: Any, *, owner: str) -> tuple[str, str, str | None, xr.Dataset]:
    context = resolve_dataset_context(
        ao,
        owner=owner,
        options=DatasetContextOptions(require_roles=True, require_sequence_dim=True),
    )
    if context.sequence_dim is None:
        raise ValueError(f"{owner}: grouped CSV export requires declared sequence_dim.")
    if len(context.batch_dims) != 1:
        raise ValueError(
            f"{owner}: grouped CSV export requires exactly one batch dim; got {context.batch_dims!r}."
        )
    batch_dim = context.batch_dims[0]
    if batch_dim not in context.ds.coords:
        raise ValueError(
            f"{owner}: grouped CSV export derives labels from batch coordinate values; "
            f"coord {batch_dim!r} is missing."
        )
    return batch_dim, context.sequence_dim, context.sequence_size_coord, context.ds


def _resolve_effective_export_size_name(
    *,
    options: CsvExportOptions,
    context_size_name: str | None,
) -> str | None:
    if options.sequence_size_coord is not None:
        return options.sequence_size_coord
    return context_size_name


def _resolve_valid_length(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    row: int,
    sequence_dim: str,
    size_name: str | None,
    owner: str,
) -> int:
    width = int(ds.sizes[sequence_dim])
    if size_name is None:
        return width
    if size_name not in ds.coords:
        raise ValueError(f"{owner}: sequence_size_coord {size_name!r} was not found in dataset coords.")
    coord = ds.coords[size_name]
    if coord.dims != (batch_dim,):
        raise ValueError(
            f"{owner}: sequence_size_coord {size_name!r} must have dims ({batch_dim!r},); "
            f"got {tuple(coord.dims)!r}."
        )
    raw_value = np.asarray(coord.isel({batch_dim: row}).values).item()
    if isinstance(raw_value, (bool, np.bool_)):
        raise ValueError(f"{owner}: sequence_size_coord {size_name!r} value {raw_value!r} is not an integer.")
    try:
        numeric = float(raw_value)
    except Exception as exc:
        raise ValueError(
            f"{owner}: sequence_size_coord {size_name!r} value {raw_value!r} is not numeric."
        ) from exc
    if not np.isfinite(numeric):
        raise ValueError(
            f"{owner}: sequence_size_coord {size_name!r} value {raw_value!r} must be finite."
        )
    if not float(numeric).is_integer():
        raise ValueError(
            f"{owner}: sequence_size_coord {size_name!r} value {raw_value!r} must be integer-valued."
        )
    value = int(numeric)
    if value < 0 or value > width:
        raise ValueError(
            f"{owner}: sequence_size_coord {size_name!r} value {value} is out of bounds for width {width}."
        )
    return value


def _append_csv_column(
    columns: dict[str, np.ndarray],
    column_origins: dict[str, object],
    *,
    raw_name: object,
    values: np.ndarray,
    owner: str,
) -> None:
    key = str(raw_name)
    if key in columns:
        first = column_origins[key]
        raise ValueError(
            f"{owner}: duplicate CSV column name {key!r} after string normalization from "
            f"{first!r} and {raw_name!r}."
        )
    columns[key] = values
    column_origins[key] = raw_name


def _row_frame_columns(
    row_ds: xr.Dataset,
    *,
    sequence_dim: str,
    valid_length: int,
    size_name: str | None,
    owner: str,
) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    column_origins: dict[str, object] = {}
    for name, coord in row_ds.coords.items():
        if name in {sequence_dim, size_name}:
            continue
        if coord.dims == (sequence_dim,):
            _append_csv_column(
                columns,
                column_origins,
                raw_name=name,
                values=coord.values[:valid_length],
                owner=owner,
            )
            continue
        if coord.dims != ():
            raise ValueError(
                f"{owner}: coordinate {name!r} is not representable for grouped CSV export; "
                f"dims={tuple(coord.dims)!r}."
            )
    for name, arr in row_ds.data_vars.items():
        if arr.dims != (sequence_dim,):
            raise ValueError(
                f"{owner}: variable {name!r} is not representable for grouped CSV export; "
                f"dims={tuple(arr.dims)!r}."
            )
        _append_csv_column(
            columns,
            column_origins,
            raw_name=name,
            values=arr.values[:valid_length],
            owner=owner,
        )
    return columns


def write_csv_logs(
    value: Any,
    out_dir: str,
    *,
    opts: CsvExportOptions | None = None,
) -> tuple[str, ...]:
    """Export a grouped AnalysisObject to one CSV file per batch label.

    Parameters
    ----------
    value : Any
        Input value consumed by this operation.
    out_dir : str
        Output directory where exported files are written.
    opts : CsvExportOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``CsvExportOptions`` key fields: ``float_format`` (default None), ``sequence_size_coord`` (default None).

    Returns
    -------
    tuple[str, ...]
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import tempfile
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.io import CsvExportOptions, write_csv_logs
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0]])}, coords={"run": ["run_a"], "sample": [0, 1], "time": ("sample", [0.0, 1.0])}),
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=(),
    ...     param_coord="time",
    ...     validate=True,
    ... )
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     paths = write_csv_logs(ao, tmpdir, opts=CsvExportOptions(float_format="%.1f"))
    >>> len(paths)
    1
    """
    owner = "tal.io.write_csv_logs"
    options = coerce_csv_export_options(opts, owner=owner)
    ao = coerce_analysis_object_input(value, owner=owner)
    batch_dim, sequence_dim, context_size_name, ds = _resolve_export_context(ao, owner=owner)
    size_name = _resolve_effective_export_size_name(options=options, context_size_name=context_size_name)
    labels = list(ds.coords[batch_dim].values)
    _require_unique_batch_labels(labels, owner=owner)
    paths: list[str] = []
    seen_paths: set[str] = set()
    for row, label in enumerate(labels):
        path = normalize_export_label_path(label, out_dir=out_dir, owner=owner)
        if path in seen_paths:
            raise ValueError(
                f"{owner}: normalized export path collision for label {label!r} at path {path!r}."
            )
        seen_paths.add(path)
        valid_length = _resolve_valid_length(
            ds,
            batch_dim=batch_dim,
            row=row,
            sequence_dim=sequence_dim,
            size_name=size_name,
            owner=owner,
        )
        row_ds = ds.isel({batch_dim: row}, drop=True)
        columns = _row_frame_columns(
            row_ds,
            sequence_dim=sequence_dim,
            valid_length=valid_length,
            size_name=size_name,
            owner=owner,
        )
        pd.DataFrame(columns).to_csv(path, index=False, float_format=options.float_format)
        paths.append(path)
    return tuple(paths)


__all__ = ["read_csv_logs", "write_csv_logs"]
