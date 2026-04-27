from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MetadataChannel = Literal["sidecar_json"]
PromotionScalarTarget = Literal["attrs", "batch_coord", "none"]
PromotionNonScalarTarget = Literal["none", "attrs"]
TimeMonotonicOrder = Literal["nondecreasing", "strict"]
InvalidTimePolicy = Literal["fail", "drop"]
RosTimestampSource = Literal["auto", "header", "receive"]


@dataclass(frozen=True)
class AOZarrWriteOptions:
    mode: str | None = None
    consolidated: bool | None = None


@dataclass(frozen=True)
class AOZarrReadOptions:
    consolidated: bool | None = None
    chunks: object | None = None


@dataclass(frozen=True)
class AOCsvWriteOptions:
    metadata_channel: MetadataChannel = "sidecar_json"
    metadata_path: str | None = None
    float_format: str | None = None


@dataclass(frozen=True)
class AOCsvReadOptions:
    metadata_channel: MetadataChannel = "sidecar_json"
    metadata_path: str | None = None


@dataclass(frozen=True)
class AdapterMetadataPromotionOptions:
    scalar_target: PromotionScalarTarget = "attrs"
    nonscalar_target: PromotionNonScalarTarget = "none"


@dataclass(frozen=True)
class CsvIngestOptions:
    """Options for CSV log ingestion.

    Notes
    -----
    ``time_col`` identifies the input column used as the AO parameter
    coordinate. ``value_columns`` can restrict which numeric columns become AO
    data variables.

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile
    >>> from tal.io import CsvIngestOptions, read_csv_logs
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir) / "run.csv"
    ...     _ = path.write_text("time,value\\n0.0,1.0\\n1.0,2.0\\n", encoding="utf-8")
    ...     ao = read_csv_logs(path, opts=CsvIngestOptions(time_col="time"))
    >>> ao.unsafe_data.sizes["sample"]
    2
    """

    time_col: str | None = None
    allow_time_infer: bool = False
    sort_time: bool = True
    monotonic_order: TimeMonotonicOrder = "nondecreasing"
    allow_nonmonotonic_normalize: bool = False
    invalid_time: InvalidTimePolicy = "fail"
    value_columns: tuple[str, ...] | None = None
    metadata_columns: tuple[str, ...] = ()
    batch_dim: str = "trial"
    sequence_dim: str = "sample"
    sequence_size_coord: str = "sequence_size"
    param_coord: str = "time"
    metadata_promotion: AdapterMetadataPromotionOptions = field(
        default_factory=AdapterMetadataPromotionOptions
    )


@dataclass(frozen=True)
class CsvExportOptions:
    """Options for CSV log export.

    Notes
    -----
    ``sequence_size_coord`` controls which validity coordinate is exported for
    ragged left-packed sequences.

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.io import CsvExportOptions, write_csv_logs
    >>> ao = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     write_csv_logs(ao, tmpdir, opts=CsvExportOptions(float_format="%.1f"))
    ...     exported = sorted(Path(tmpdir).glob("*.csv"))
    >>> bool(exported)
    True
    """

    float_format: str | None = None
    sequence_size_coord: str | None = None


@dataclass(frozen=True)
class RosIngestOptions:
    """Options for ROS log ingestion.

    Notes
    -----
    These options describe topic and timestamp handling for ROS-style adapters.
    Readers consume the options at the ingestion boundary.

    Examples
    --------
    >>> from tal.io import RosIngestOptions
    >>> opts = RosIngestOptions(topic="/odom", timestamp_source="header")
    >>> (opts.topic, opts.timestamp_source)
    ('/odom', 'header')
    """

    topic: str | None = None
    message_type: str | None = None
    timestamp_source: RosTimestampSource = "auto"
    sort_time: bool = True
    monotonic_order: TimeMonotonicOrder = "nondecreasing"
    allow_nonmonotonic_normalize: bool = False
    invalid_time: InvalidTimePolicy = "fail"
    batch_dim: str = "trial"
    sequence_dim: str = "sample"
    sequence_size_coord: str = "sequence_size"
    param_coord: str = "time"
    metadata_promotion: AdapterMetadataPromotionOptions = field(
        default_factory=AdapterMetadataPromotionOptions
    )


def _coerce_metadata_channel(channel: str, *, owner: str) -> MetadataChannel:
    if channel != "sidecar_json":
        raise ValueError(f"{owner}: metadata_channel must be 'sidecar_json'; got {channel!r}.")
    return channel


def _coerce_optional_path(path: str | None, *, owner: str, field_name: str) -> str | None:
    if path is None:
        return None
    if not isinstance(path, str) or not path:
        raise ValueError(f"{owner}: {field_name} must be a non-empty string when provided.")
    return path


def _coerce_non_empty_name(value: str, *, owner: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner}: {field_name} must be a non-empty string.")
    return value


def _coerce_optional_name(value: str | None, *, owner: str, field_name: str) -> str | None:
    if value is None:
        return None
    return _coerce_non_empty_name(value, owner=owner, field_name=field_name)


def _coerce_name_tuple(
    names: tuple[str, ...] | list[str] | None,
    *,
    owner: str,
    field_name: str,
) -> tuple[str, ...] | None:
    if names is None:
        return None
    if not isinstance(names, (tuple, list)):
        raise TypeError(f"{owner}: {field_name} must be a tuple/list of non-empty strings or None.")
    out: list[str] = []
    for idx, name in enumerate(names):
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{owner}: {field_name}[{idx}] must be a non-empty string; got {name!r}."
            )
        out.append(name)
    duplicate = _first_duplicate(out)
    if duplicate is not None:
        raise ValueError(f"{owner}: {field_name} contains duplicate name {duplicate!r}.")
    return tuple(out)


def _first_duplicate(names: list[str]) -> str | None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            return name
        seen.add(name)
    return None


def _coerce_scalar_target(value: str, *, owner: str) -> PromotionScalarTarget:
    if value not in {"attrs", "batch_coord", "none"}:
        raise ValueError(
            f"{owner}: scalar_target must be one of 'attrs', 'batch_coord', or 'none'; got {value!r}."
        )
    return value


def _coerce_nonscalar_target(value: str, *, owner: str) -> PromotionNonScalarTarget:
    if value not in {"none", "attrs"}:
        raise ValueError(f"{owner}: nonscalar_target must be 'none' or 'attrs'; got {value!r}.")
    return value


def _coerce_monotonic_order(value: str, *, owner: str) -> TimeMonotonicOrder:
    if value not in {"nondecreasing", "strict"}:
        raise ValueError(
            f"{owner}: monotonic_order must be one of 'nondecreasing' or 'strict'; got {value!r}."
        )
    return value


def _coerce_invalid_time(value: str, *, owner: str) -> InvalidTimePolicy:
    if value not in {"fail", "drop"}:
        raise ValueError(f"{owner}: invalid_time must be 'fail' or 'drop'; got {value!r}.")
    return value


def _coerce_ros_timestamp_source(value: str, *, owner: str) -> RosTimestampSource:
    if value not in {"auto", "header", "receive"}:
        raise ValueError(
            f"{owner}: timestamp_source must be one of 'auto', 'header', or 'receive'; got {value!r}."
        )
    return value


def coerce_zarr_write_options(
    opts: AOZarrWriteOptions | None,
    *,
    owner: str,
) -> AOZarrWriteOptions:
    if opts is None:
        return AOZarrWriteOptions()
    if not isinstance(opts, AOZarrWriteOptions):
        raise TypeError(f"{owner}: opts must be AOZarrWriteOptions or None.")
    return opts


def coerce_zarr_read_options(
    opts: AOZarrReadOptions | None,
    *,
    owner: str,
) -> AOZarrReadOptions:
    if opts is None:
        return AOZarrReadOptions()
    if not isinstance(opts, AOZarrReadOptions):
        raise TypeError(f"{owner}: opts must be AOZarrReadOptions or None.")
    return opts


def coerce_csv_write_options(
    opts: AOCsvWriteOptions | None,
    *,
    owner: str,
) -> AOCsvWriteOptions:
    if opts is None:
        return AOCsvWriteOptions()
    if not isinstance(opts, AOCsvWriteOptions):
        raise TypeError(f"{owner}: opts must be AOCsvWriteOptions or None.")
    return AOCsvWriteOptions(
        metadata_channel=_coerce_metadata_channel(opts.metadata_channel, owner=owner),
        metadata_path=_coerce_optional_path(opts.metadata_path, owner=owner, field_name="metadata_path"),
        float_format=opts.float_format,
    )


def coerce_csv_read_options(
    opts: AOCsvReadOptions | None,
    *,
    owner: str,
) -> AOCsvReadOptions:
    if opts is None:
        return AOCsvReadOptions()
    if not isinstance(opts, AOCsvReadOptions):
        raise TypeError(f"{owner}: opts must be AOCsvReadOptions or None.")
    return AOCsvReadOptions(
        metadata_channel=_coerce_metadata_channel(opts.metadata_channel, owner=owner),
        metadata_path=_coerce_optional_path(opts.metadata_path, owner=owner, field_name="metadata_path"),
    )


def coerce_adapter_metadata_promotion_options(
    opts: AdapterMetadataPromotionOptions | None,
    *,
    owner: str,
) -> AdapterMetadataPromotionOptions:
    if opts is None:
        return AdapterMetadataPromotionOptions()
    if not isinstance(opts, AdapterMetadataPromotionOptions):
        raise TypeError(f"{owner}: metadata_promotion must be AdapterMetadataPromotionOptions or None.")
    return AdapterMetadataPromotionOptions(
        scalar_target=_coerce_scalar_target(opts.scalar_target, owner=owner),
        nonscalar_target=_coerce_nonscalar_target(opts.nonscalar_target, owner=owner),
    )


def coerce_csv_ingest_options(
    opts: CsvIngestOptions | None,
    *,
    owner: str,
) -> CsvIngestOptions:
    if opts is None:
        return CsvIngestOptions()
    if not isinstance(opts, CsvIngestOptions):
        raise TypeError(f"{owner}: opts must be CsvIngestOptions or None.")
    return CsvIngestOptions(
        time_col=_coerce_optional_name(opts.time_col, owner=owner, field_name="time_col"),
        allow_time_infer=bool(opts.allow_time_infer),
        sort_time=bool(opts.sort_time),
        monotonic_order=_coerce_monotonic_order(opts.monotonic_order, owner=owner),
        allow_nonmonotonic_normalize=bool(opts.allow_nonmonotonic_normalize),
        invalid_time=_coerce_invalid_time(opts.invalid_time, owner=owner),
        value_columns=_coerce_name_tuple(opts.value_columns, owner=owner, field_name="value_columns"),
        metadata_columns=_coerce_name_tuple(opts.metadata_columns, owner=owner, field_name="metadata_columns")
        or (),
        batch_dim=_coerce_non_empty_name(opts.batch_dim, owner=owner, field_name="batch_dim"),
        sequence_dim=_coerce_non_empty_name(opts.sequence_dim, owner=owner, field_name="sequence_dim"),
        sequence_size_coord=_coerce_non_empty_name(
            opts.sequence_size_coord, owner=owner, field_name="sequence_size_coord"
        ),
        param_coord=_coerce_non_empty_name(opts.param_coord, owner=owner, field_name="param_coord"),
        metadata_promotion=coerce_adapter_metadata_promotion_options(
            opts.metadata_promotion, owner=owner
        ),
    )


def coerce_csv_export_options(
    opts: CsvExportOptions | None,
    *,
    owner: str,
) -> CsvExportOptions:
    if opts is None:
        return CsvExportOptions()
    if not isinstance(opts, CsvExportOptions):
        raise TypeError(f"{owner}: opts must be CsvExportOptions or None.")
    size_name = _coerce_optional_name(
        opts.sequence_size_coord,
        owner=owner,
        field_name="sequence_size_coord",
    )
    return CsvExportOptions(float_format=opts.float_format, sequence_size_coord=size_name)


def coerce_ros_ingest_options(
    opts: RosIngestOptions | None,
    *,
    owner: str,
) -> RosIngestOptions:
    if opts is None:
        return RosIngestOptions()
    if not isinstance(opts, RosIngestOptions):
        raise TypeError(f"{owner}: opts must be RosIngestOptions or None.")
    return RosIngestOptions(
        topic=_coerce_optional_name(opts.topic, owner=owner, field_name="topic"),
        message_type=_coerce_optional_name(opts.message_type, owner=owner, field_name="message_type"),
        timestamp_source=_coerce_ros_timestamp_source(opts.timestamp_source, owner=owner),
        sort_time=bool(opts.sort_time),
        monotonic_order=_coerce_monotonic_order(opts.monotonic_order, owner=owner),
        allow_nonmonotonic_normalize=bool(opts.allow_nonmonotonic_normalize),
        invalid_time=_coerce_invalid_time(opts.invalid_time, owner=owner),
        batch_dim=_coerce_non_empty_name(opts.batch_dim, owner=owner, field_name="batch_dim"),
        sequence_dim=_coerce_non_empty_name(opts.sequence_dim, owner=owner, field_name="sequence_dim"),
        sequence_size_coord=_coerce_non_empty_name(
            opts.sequence_size_coord, owner=owner, field_name="sequence_size_coord"
        ),
        param_coord=_coerce_non_empty_name(opts.param_coord, owner=owner, field_name="param_coord"),
        metadata_promotion=coerce_adapter_metadata_promotion_options(
            opts.metadata_promotion, owner=owner
        ),
    )


__all__ = [
    "AOCsvReadOptions",
    "AOCsvWriteOptions",
    "AdapterMetadataPromotionOptions",
    "AOZarrReadOptions",
    "AOZarrWriteOptions",
    "CsvExportOptions",
    "CsvIngestOptions",
    "InvalidTimePolicy",
    "PromotionNonScalarTarget",
    "PromotionScalarTarget",
    "RosIngestOptions",
    "RosTimestampSource",
    "TimeMonotonicOrder",
    "coerce_adapter_metadata_promotion_options",
    "coerce_csv_read_options",
    "coerce_csv_export_options",
    "coerce_csv_ingest_options",
    "coerce_csv_write_options",
    "coerce_ros_ingest_options",
    "coerce_zarr_read_options",
    "coerce_zarr_write_options",
]
