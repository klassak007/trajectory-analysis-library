(api-io)=
# I/O

TAL provides two I/O layers:

- AO-direct persistence for existing `AnalysisObject` instances.
- Log-oriented adapters for ingesting CSV or ROS-style recordings.

AO-direct persistence is the canonical round-trip path. Catalog bridge helpers
are optional convenience.

```{contents}
:local:
:depth: 2
```

## AO-Direct Surface

```python
ao.io.to_zarr(...)
ao.io.to_csv(...)
AnalysisObject.from_zarr(...)
AnalysisObject.from_csv(...)
```

Zarr is the richest AO round-trip format. It preserves schema roles,
parameter-coordinate metadata, validity metadata, and ordinary xarray payload
structure.

CSV is intentionally narrower. It is a flat single-object round trip for
representable one-dimensional row payloads, with required TAL reconstruction
metadata carried in an explicit metadata channel.

Loaders return the class they were called on when the persisted payload
validates for that class. CSV sidecar metadata is validated before column
reconstruction proceeds.

## Log Readers and Writers

```python
from tal.io import read_csv_logs, read_csv_logs_catalog, read_ros_logs, read_ros_logs_catalog
```

CSV and ROS readers ingest external logs into TAL objects or catalog views. CSV
export is available through `write_csv_logs(...)` when grouped AOs should be
written back to per-run tabular files.

`read_csv_logs_catalog(...)` and `read_ros_logs_catalog(...)` are convenience
constructors over the same adapter inputs. ROS readers use message timestamps
when available before falling back to receive/log time.

## Deterministic Input Policy

Log ingestion accepts:

- explicit `Mapping[str, str]` label-to-path inputs
- `Sequence[str]` path inputs
- glob strings

Sequences preserve caller order. Glob inputs are expanded and sorted
lexicographically by normalized resolved path. Duplicate paths and derived label
collisions fail closed.

## CSV and Timestamp Policy

CSV default behavior requires explicit `time_col`. Optional inference is
available only for a small known candidate set and succeeds only when exactly
one candidate exists.

ROS timestamp precedence is deterministic: `header.stamp` first, then
receive/log time. Empty `0/0` header stamps are treated as missing.

## Metadata Promotion

Metadata promotion is controlled by options. Scalar metadata defaults to attrs;
non-scalar metadata is omitted unless explicitly enabled.

The `tal` metadata key is reserved for schema and cannot be used as ordinary
promoted metadata.

CSV export uses declared sequence-size metadata when present, so ragged
validity is not guessed from padded rows. Explicit value-column selections must
match available numeric columns.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/io
   :nosignatures:

   tal.io.AnalysisObjectIOAccessor.to_zarr
   tal.io.AnalysisObjectIOAccessor.to_csv
   tal.AnalysisObject.from_zarr
   tal.AnalysisObject.from_csv
   tal.io.CsvIngestOptions
   tal.io.CsvExportOptions
   tal.io.RosIngestOptions
   tal.io.read_csv_logs
   tal.io.read_csv_logs_catalog
   tal.io.write_csv_logs
   tal.io.read_ros_logs
   tal.io.read_ros_logs_catalog
```

## See Also

- {doc}`analysis-object`
- {doc}`catalog`
