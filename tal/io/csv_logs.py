from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .adapter_finalize import _finalize_owned_adapter_dataset
from .adapter_metadata import (
    aggregate_batch_metadata,
    promote_adapter_metadata,
    require_generated_metadata_preflight,
)
from .adapter_paths import resolve_ingest_inputs
from .adapter_spool import AdapterArraySpool, fill_spooled_adapter_row, spool_adapter_arrays
from .adapter_temp import owned_temporary_directory
from .adapter_time import is_monotonic
from .csv_export import coerce_csv_export_source, execute_csv_export
from .csv_metadata import collect_csv_scalar_metadata
from .csv_time import normalize_csv_time_tokens
from .csv_validation import require_valid_csv_header
from .options import (
    CsvExportOptions,
    CsvIngestOptions,
    coerce_csv_export_options,
    coerce_csv_ingest_options,
)

_TIME_INFER_CANDIDATES = ("time", "timestamp", "t", "time_s", "time_sec")
_CSV_GENERATED_METADATA_NAMES = ("io_time_source", "io_source_paths")
_CSV_SCALAR_GENERATED_METADATA_NAMES = ("io_time_source",)


@dataclass(frozen=True)
class _CsvRecord:
    label: str
    resolved_path: str
    time_col: str
    time_values: np.ndarray
    value_columns: tuple[str, ...]
    values: dict[str, np.ndarray]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _CsvSpoolRecord:
    label: str
    resolved_path: str
    time_col: str
    time_dtype: str
    integer_time_bounds: tuple[int, int] | None
    metadata: dict[str, object]
    arrays: AdapterArraySpool


def _require_csv_ingest_preflight(
    opts: CsvIngestOptions,
    *,
    value_columns: Sequence[str],
    owner: str,
) -> None:
    if opts.value_columns is not None and not value_columns:
        raise ValueError(f"{owner}: value_columns must contain at least one name when provided.")
    layout_names = {
        opts.batch_dim,
        opts.sequence_dim,
        opts.sequence_size_coord,
        opts.param_coord,
    }
    collisions = tuple(name for name in value_columns if name in layout_names)
    if collisions:
        raise ValueError(
            f"{owner}: value columns collide with reserved semantic names: {collisions!r}."
        )
    occupied = (
        opts.batch_dim,
        opts.sequence_dim,
        opts.sequence_size_coord,
        opts.param_coord,
        *value_columns,
    )
    require_generated_metadata_preflight(
        options=opts.metadata_promotion,
        generated_names=_CSV_GENERATED_METADATA_NAMES,
        scalar_generated_names=_CSV_SCALAR_GENERATED_METADATA_NAMES,
        user_metadata_names=opts.metadata_columns,
        occupied_names=occupied,
        owner=owner,
    )


def _resolve_time_column(
    columns: Sequence[object],
    *,
    opts: CsvIngestOptions,
    owner: str,
    path: str,
) -> str:
    if opts.time_col is not None:
        if opts.time_col not in columns:
            raise ValueError(f"{owner}: time_col {opts.time_col!r} not found in {path!r}.")
        return opts.time_col
    if not opts.allow_time_infer:
        raise ValueError(
            f"{owner}: time_col must be explicitly provided by default; set allow_time_infer=True "
            "to enable inference."
        )
    matches = [
        name
        for candidate in _TIME_INFER_CANDIDATES
        for name in columns
        if str(name).lower() == candidate
    ]
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
    normalized = normalize_csv_time_tokens(
        frame[time_col],
        owner=owner,
        label=f"time column {time_col!r} in {path!r}",
    )
    if normalized.invalid_count:
        if opts.invalid_time == "fail":
            raise ValueError(
                f"{owner}: time column {time_col!r} in {path!r} contains "
                f"{normalized.invalid_count} non-finite values."
            )
        frame = frame.loc[normalized.valid_mask].copy(deep=False)
        if frame.empty:
            raise ValueError(f"{owner}: all rows were dropped in {path!r} due to non-finite time values.")
    out = frame.copy(deep=False)
    out[time_col] = normalized.values
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
    values = frame[time_col].to_numpy(copy=False)
    if is_monotonic(values, order=opts.monotonic_order):
        return frame
    if not opts.allow_nonmonotonic_normalize:
        raise ValueError(
            f"{owner}: time column {time_col!r} in {path!r} violates monotonic_order={opts.monotonic_order!r}."
        )
    sorted_frame = frame.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    sorted_values = sorted_frame[time_col].to_numpy(copy=False)
    if not is_monotonic(sorted_values, order=opts.monotonic_order):
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
    _require_csv_ingest_preflight(opts, value_columns=columns, owner=owner)
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


def _read_csv_frame(path: str, *, time_col: str, owner: str) -> pd.DataFrame:
    """Delegate complete record parsing to pandas under the public owner."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            return pd.read_csv(
                path,
                converters={time_col: str},
                index_col=False,
                on_bad_lines="error",
            )
    except Exception as exc:  # pragma: no cover - pandas engine envelope.
        raise ValueError(f"{owner}: failed reading CSV file {path!r}.") from exc


def _read_csv_record(
    path_info: Any,
    *,
    opts: CsvIngestOptions,
    owner: str,
) -> _CsvRecord:
    path = str(path_info.resolved_path)
    columns = require_valid_csv_header(path, owner=owner)
    time_col = _resolve_time_column(columns, opts=opts, owner=owner, path=path)
    frame = _read_csv_frame(path, time_col=time_col, owner=owner)
    if frame.empty:
        raise ValueError(f"{owner}: CSV file {path!r} contains no data rows.")
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
        resolved_path=path_info.resolved_path,
        time_col=time_col,
        time_values=frame[time_col].to_numpy(copy=False),
        value_columns=value_columns,
        values=values,
        metadata=metadata,
    )


def _integer_time_bounds(values: np.ndarray) -> tuple[int, int] | None:
    if values.dtype.kind not in "iu":
        return None
    return int(values.min()), int(values.max())


def _spool_csv_records(
    path_infos: Sequence[Any],
    *,
    spool_dir: str,
    opts: CsvIngestOptions,
    owner: str,
) -> tuple[tuple[_CsvSpoolRecord, ...], tuple[str, ...]]:
    plans: list[_CsvSpoolRecord] = []
    value_columns: tuple[str, ...] | None = None
    for row, path_info in enumerate(path_infos):
        record = _read_csv_record(path_info, opts=opts, owner=owner)
        if value_columns is None:
            value_columns = record.value_columns
        elif set(record.value_columns) != set(value_columns):
            raise ValueError(
                f"{owner}: value-column mismatch across inputs; expected {value_columns!r}, "
                f"got {record.value_columns!r} for {record.resolved_path!r}."
            )
        arrays = spool_adapter_arrays(
            spool_dir,
            row=row,
            time_values=record.time_values,
            field_values=tuple(record.values[name] for name in value_columns),
            owner=owner,
        )
        plans.append(
            _CsvSpoolRecord(
                label=record.label,
                resolved_path=record.resolved_path,
                time_col=record.time_col,
                time_dtype=record.time_values.dtype.str,
                integer_time_bounds=_integer_time_bounds(record.time_values),
                metadata=record.metadata,
                arrays=arrays,
            )
        )
        del record
    if value_columns is None:  # pragma: no cover - non-empty input invariant.
        raise RuntimeError("CSV spool planning received no inputs")
    return tuple(plans), value_columns


def _spooled_csv_records_to_dataset(
    records: Sequence[_CsvSpoolRecord],
    *,
    value_columns: tuple[str, ...],
    opts: CsvIngestOptions,
    owner: str,
) -> xr.Dataset:
    labels = [record.label for record in records]
    sizes, time_grid, var_arrays = _build_spooled_record_grids(
        records,
        value_columns=value_columns,
        owner=owner,
    )
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


def _build_spooled_record_grids(
    records: Sequence[_CsvSpoolRecord],
    *,
    value_columns: tuple[str, ...],
    owner: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    batch = len(records)
    width = max(record.arrays.size for record in records)
    sizes = np.asarray([record.arrays.size for record in records], dtype=np.int64)
    time_dtype = _resolve_csv_time_dtype(records, owner=owner)
    time_grid = np.full(
        (batch, width),
        _csv_time_padding_value(time_dtype),
        dtype=time_dtype,
    )
    var_arrays = {
        name: np.full((batch, width), np.nan, dtype=float)
        for name in value_columns
    }
    for row, record in enumerate(records):
        fill_spooled_adapter_row(
            record.arrays,
            row=row,
            time_target=time_grid,
            field_targets=tuple(var_arrays[name] for name in value_columns),
            owner=owner,
        )
    return sizes, time_grid, var_arrays


def _resolve_csv_time_dtype(
    records: Sequence[_CsvSpoolRecord],
    *,
    owner: str,
) -> np.dtype[Any]:
    dtypes = tuple(np.dtype(record.time_dtype) for record in records)
    integer_records = tuple(record for record in records if record.integer_time_bounds is not None)
    if len(integer_records) == len(records):
        return _resolve_integer_time_dtype(dtypes, integer_records, owner=owner)
    dtype = np.result_type(*dtypes)
    if integer_records and dtype.kind == "f":
        exact_limit = 2 ** (np.finfo(dtype).nmant + 1)
        if any(
            max(abs(bounds[0]), abs(bounds[1])) > exact_limit
            for record in integer_records
            if (bounds := record.integer_time_bounds) is not None
        ):
            raise ValueError(
                f"{owner}: integer CSV time values cannot be combined with floating-point "
                "time values without precision loss."
            )
    return dtype


def _resolve_integer_time_dtype(
    dtypes: tuple[np.dtype[Any], ...],
    records: Sequence[_CsvSpoolRecord],
    *,
    owner: str,
) -> np.dtype[Any]:
    dtype = np.result_type(*dtypes)
    if dtype.kind in "iu":
        return dtype
    bounds = tuple(record.integer_time_bounds for record in records)
    minimum = min(bound[0] for bound in bounds if bound is not None)
    maximum = max(bound[1] for bound in bounds if bound is not None)
    if minimum >= 0:
        return np.dtype(np.uint64)
    if maximum <= np.iinfo(np.int64).max:
        return np.dtype(np.int64)
    raise ValueError(
        f"{owner}: integer CSV time values span no lossless NumPy integer dtype."
    )


def _csv_time_padding_value(dtype: np.dtype[Any]) -> int | float:
    if dtype.kind == "f":
        return np.nan
    if dtype.kind == "u":
        return 0
    return int(np.iinfo(dtype).min)


def _build_batch_metadata(records: Sequence[_CsvSpoolRecord]) -> dict[str, object]:
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
    This lossy adapter constructs a new AO from explicit ingest options and
    input columns. It is not the inverse of :func:`write_csv_logs`. Duplicate
    source headers fail before pandas can assign normalized identities. Python's
    CSV reader owns logical-header grammar; Python/pandas own complete-record
    grammar, payload NUL behavior, field-size limits, and parser security.
    Normalized records are temporarily spooled so final padded grids do not
    coexist with every source record in memory.

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
    _require_csv_ingest_preflight(
        options,
        value_columns=options.value_columns if options.value_columns is not None else (),
        owner=owner,
    )
    path_infos = resolve_ingest_inputs(inputs, owner=owner)
    with owned_temporary_directory(
        prefix="tal-csv-ingest-",
        owner=owner,
        purpose="CSV ingest spool directory",
    ) as spool_dir:
        records, value_columns = _spool_csv_records(
            path_infos,
            spool_dir=spool_dir,
            opts=options,
            owner=owner,
        )
        ds = _spooled_csv_records_to_dataset(
            records,
            value_columns=value_columns,
            opts=options,
            owner=owner,
        )
    return _finalize_owned_adapter_dataset(
        ds,
        batch_dim=options.batch_dim,
        sequence_dim=options.sequence_dim,
        size_name=options.sequence_size_coord,
        param_name=options.param_coord,
        validate=validate,
        owner=owner,
    )


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
    This is a lossy tabular projection, not AO persistence. It omits padded
    tails using declared validity, may eagerly materialize selected rows, and
    does not preserve arbitrary schema, dtype, attrs, encoding, indexes,
    chunks, or subclass identity. Resident rows serialize without Dask; lazy
    rows share one graph under the configured scheduler and are jointly realized
    before serialization. The writer serializes each row directly to a
    same-directory temporary file and commits through per-destination atomic
    replacement. The batch write is non-transactional. Export assumes a trusted
    filesystem whose path topology remains stable.

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
    ao = coerce_csv_export_source(value, owner=owner)
    return execute_csv_export(ao, out_dir, options=options, owner=owner)


__all__ = ["read_csv_logs", "write_csv_logs"]
