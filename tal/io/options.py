from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

PromotionScalarTarget = Literal["attrs", "batch_coord", "none"]
PromotionNonScalarTarget = Literal["none", "attrs"]
TimeMonotonicOrder = Literal["nondecreasing", "strict"]
InvalidTimePolicy = Literal["fail", "drop"]
RosTimestampSource = Literal["auto", "header", "receive"]
ZarrWriteMode = Literal["w", "w-"]


@dataclass(frozen=True)
class AOZarrWriteOptions:
    mode: ZarrWriteMode | None = None
    consolidated: bool | None = None


@dataclass(frozen=True)
class AOZarrReadOptions:
    consolidated: bool | None = None
    chunks: object | None = None


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
    data variables. Ingestion constructs a new AO from the selected columns; it
    is not an inverse of CSV log export.

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile
    >>> from tal.io import CsvIngestOptions, read_csv_logs
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir) / "run.csv"
    ...     _ = path.write_text("time,value\\n0.0,1.0\\n1.0,2.0\\n", encoding="utf-8")
    ...     ao = read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))
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
    ``sequence_size_coord`` selects the validity coordinate whose values
    determine the exported row count for each ragged left-packed sequence. The
    validity coordinate itself is not exported. Export is a lossy tabular
    projection and does not preserve arbitrary xarray or TAL metadata.

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.io import CsvExportOptions, write_csv_logs
    >>> ds = xr.Dataset(
    ...     {"value": (("trial", "sample"), [[1.0]])},
    ...     coords={"trial": ["run"], "sample": [0]},
    ... )
    >>> ao = AnalysisObject.from_data(
    ...     ds,
    ...     sequence_dim="sample",
    ...     batch_dims=("trial",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     _ = write_csv_logs(ao, tmpdir, opts=CsvExportOptions(float_format="%.1f"))
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
    The selected timestamp source is normalized exactly to signed int64 Unix
    nanoseconds at the ingestion boundary.

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


def _coerce_float_format(value: str | None, *, owner: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{owner}: float_format must be a built-in string or None.")
    return value


def _coerce_bool(value: object, *, owner: str, field_name: str) -> bool:
    if type(value) is not bool:
        value_type = type(value)
        type_name = value_type.__name__
        if value_type.__module__ != "builtins":
            type_name = f"{value_type.__module__}.{type_name}"
        raise TypeError(f"{owner}: {field_name} must be bool (built-in); got {type_name}.")
    return value is True


def _coerce_optional_bool(
    value: object,
    *,
    owner: str,
    field_name: str,
) -> bool | None:
    if value is None:
        return None
    return _coerce_bool(value, owner=owner, field_name=field_name)


def _coerce_zarr_write_mode(value: object, *, owner: str) -> ZarrWriteMode | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{owner}: mode must be a string or None.")
    if value not in {"w", "w-"}:
        raise ValueError(
            f"{owner}: mode must be 'w', 'w-', or None; got {value!r}. "
            "Incremental Zarr mutation is not supported by AO-direct persistence."
        )
    return cast(ZarrWriteMode, value)


def _coerce_non_empty_name(value: str, *, owner: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner}: {field_name} must be a non-empty string.")
    return value


def _coerce_optional_name(value: str | None, *, owner: str, field_name: str) -> str | None:
    if value is None:
        return None
    return _coerce_non_empty_name(value, owner=owner, field_name=field_name)


def _coerce_adapter_layout_names(
    *,
    batch_dim: str,
    sequence_dim: str,
    sequence_size_coord: str,
    param_coord: str,
    owner: str,
) -> tuple[str, str, str, str]:
    raw_names = (
        ("batch_dim", batch_dim),
        ("sequence_dim", sequence_dim),
        ("sequence_size_coord", sequence_size_coord),
        ("param_coord", param_coord),
    )
    names = tuple(
        (field_name, _coerce_non_empty_name(value, owner=owner, field_name=field_name))
        for field_name, value in raw_names
    )
    owners: dict[str, str] = {}
    for field_name, name in names:
        if name in owners:
            raise ValueError(
                f"{owner}: {owners[name]} and {field_name} must be distinct; both are {name!r}."
            )
        owners[name] = field_name
    return names[0][1], names[1][1], names[2][1], names[3][1]


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


def _coerce_choice(
    value: object,
    *,
    choices: tuple[str, ...],
    owner: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{owner}: {field_name} must be a string; got {type(value).__name__}.")
    if value not in choices:
        raise ValueError(f"{owner}: {field_name} must be one of {choices!r}; got {value!r}.")
    return value


def _coerce_scalar_target(value: object, *, owner: str) -> PromotionScalarTarget:
    return cast(
        PromotionScalarTarget,
        _coerce_choice(
            value,
            choices=("attrs", "batch_coord", "none"),
            owner=owner,
            field_name="scalar_target",
        ),
    )


def _coerce_nonscalar_target(value: object, *, owner: str) -> PromotionNonScalarTarget:
    return cast(
        PromotionNonScalarTarget,
        _coerce_choice(
            value,
            choices=("none", "attrs"),
            owner=owner,
            field_name="nonscalar_target",
        ),
    )


def _coerce_monotonic_order(value: object, *, owner: str) -> TimeMonotonicOrder:
    return cast(
        TimeMonotonicOrder,
        _coerce_choice(
            value,
            choices=("nondecreasing", "strict"),
            owner=owner,
            field_name="monotonic_order",
        ),
    )


def _coerce_invalid_time(value: object, *, owner: str) -> InvalidTimePolicy:
    return cast(
        InvalidTimePolicy,
        _coerce_choice(
            value,
            choices=("fail", "drop"),
            owner=owner,
            field_name="invalid_time",
        ),
    )


def _coerce_ros_timestamp_source(value: object, *, owner: str) -> RosTimestampSource:
    return cast(
        RosTimestampSource,
        _coerce_choice(
            value,
            choices=("auto", "header", "receive"),
            owner=owner,
            field_name="timestamp_source",
        ),
    )


def coerce_zarr_write_options(
    opts: AOZarrWriteOptions | None,
    *,
    owner: str,
) -> AOZarrWriteOptions:
    if opts is None:
        return AOZarrWriteOptions()
    if not isinstance(opts, AOZarrWriteOptions):
        raise TypeError(f"{owner}: opts must be AOZarrWriteOptions or None.")
    return AOZarrWriteOptions(
        mode=_coerce_zarr_write_mode(opts.mode, owner=owner),
        consolidated=_coerce_optional_bool(
            opts.consolidated,
            owner=owner,
            field_name="consolidated",
        ),
    )


def coerce_zarr_read_options(
    opts: AOZarrReadOptions | None,
    *,
    owner: str,
) -> AOZarrReadOptions:
    if opts is None:
        return AOZarrReadOptions()
    if not isinstance(opts, AOZarrReadOptions):
        raise TypeError(f"{owner}: opts must be AOZarrReadOptions or None.")
    return AOZarrReadOptions(
        consolidated=_coerce_optional_bool(
            opts.consolidated,
            owner=owner,
            field_name="consolidated",
        ),
        chunks=opts.chunks,
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
    batch_dim, sequence_dim, size_name, param_name = _coerce_adapter_layout_names(
        batch_dim=opts.batch_dim,
        sequence_dim=opts.sequence_dim,
        sequence_size_coord=opts.sequence_size_coord,
        param_coord=opts.param_coord,
        owner=owner,
    )
    return CsvIngestOptions(
        time_col=_coerce_optional_name(opts.time_col, owner=owner, field_name="time_col"),
        allow_time_infer=_coerce_bool(
            opts.allow_time_infer, owner=owner, field_name="allow_time_infer"
        ),
        sort_time=_coerce_bool(opts.sort_time, owner=owner, field_name="sort_time"),
        monotonic_order=_coerce_monotonic_order(opts.monotonic_order, owner=owner),
        allow_nonmonotonic_normalize=_coerce_bool(
            opts.allow_nonmonotonic_normalize,
            owner=owner,
            field_name="allow_nonmonotonic_normalize",
        ),
        invalid_time=_coerce_invalid_time(opts.invalid_time, owner=owner),
        value_columns=_coerce_name_tuple(opts.value_columns, owner=owner, field_name="value_columns"),
        metadata_columns=_coerce_name_tuple(opts.metadata_columns, owner=owner, field_name="metadata_columns")
        or (),
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
        sequence_size_coord=size_name,
        param_coord=param_name,
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
    return CsvExportOptions(
        float_format=_coerce_float_format(opts.float_format, owner=owner),
        sequence_size_coord=size_name,
    )


def coerce_ros_ingest_options(
    opts: RosIngestOptions | None,
    *,
    owner: str,
) -> RosIngestOptions:
    if opts is None:
        return RosIngestOptions()
    if not isinstance(opts, RosIngestOptions):
        raise TypeError(f"{owner}: opts must be RosIngestOptions or None.")
    batch_dim, sequence_dim, size_name, param_name = _coerce_adapter_layout_names(
        batch_dim=opts.batch_dim,
        sequence_dim=opts.sequence_dim,
        sequence_size_coord=opts.sequence_size_coord,
        param_coord=opts.param_coord,
        owner=owner,
    )
    return RosIngestOptions(
        topic=_coerce_optional_name(opts.topic, owner=owner, field_name="topic"),
        message_type=_coerce_optional_name(opts.message_type, owner=owner, field_name="message_type"),
        timestamp_source=_coerce_ros_timestamp_source(opts.timestamp_source, owner=owner),
        sort_time=_coerce_bool(opts.sort_time, owner=owner, field_name="sort_time"),
        monotonic_order=_coerce_monotonic_order(opts.monotonic_order, owner=owner),
        allow_nonmonotonic_normalize=_coerce_bool(
            opts.allow_nonmonotonic_normalize,
            owner=owner,
            field_name="allow_nonmonotonic_normalize",
        ),
        invalid_time=_coerce_invalid_time(opts.invalid_time, owner=owner),
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
        sequence_size_coord=size_name,
        param_coord=param_name,
        metadata_promotion=coerce_adapter_metadata_promotion_options(
            opts.metadata_promotion, owner=owner
        ),
    )


__all__ = [
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
    "ZarrWriteMode",
    "coerce_adapter_metadata_promotion_options",
    "coerce_csv_export_options",
    "coerce_csv_ingest_options",
    "coerce_ros_ingest_options",
    "coerce_zarr_read_options",
    "coerce_zarr_write_options",
]
