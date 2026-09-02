(api-io)=
# I/O

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

TAL separates canonical persistence from tabular interchange:

- Zarr is the canonical AO-direct persistence format.
- CSV and ROS APIs are log-oriented adapters.

```{contents}
:local:
:depth: 2
```

## AO-Direct Surface

```python
ao.io.to_zarr(...)
AnalysisObject.from_zarr(...)
```

Zarr is TAL's canonical AnalysisObject persistence boundary. TAL preserves validated
TAL schema and decoded array values, dimensions, coordinate relationships, parameter
semantics, and validity semantics within xarray/Zarr's serialization model. It does
not promise representation identity: Python attribute container types may normalize,
an index may load as a different equivalent index class, encoding dictionaries need
not round-trip, and eager/Dask backing or chunk topology may depend on storage and read
options. Before writing, TAL structurally validates the AO and materializes only a
declared non-resident sequence-size coordinate so invalid validity values fail before
store mutation. This includes Dask and other xarray backend arrays, and the coordinate
is realized exactly once under the public I/O error owner. On read, TAL structurally validates persisted
metadata and uses the same narrow materialization boundary. Other chunked payload
variables remain lazy until the requested storage or caller computation. The loader returns the class it was called on
when the payload satisfies that class's invariants. A successful lazy load transfers
the opened store's close ownership to the returned dataset; call
`loaded.close()` when the AO is no longer needed to release backend
resources. A subclass-provided dataset close callback is composed with backend
cleanup rather than replaced. If validation or materialization has already
failed, a cleanup interruption cannot replace that active primary failure.

AO-direct writes persist one complete snapshot. `AOZarrWriteOptions.mode` accepts
the backend default, `"w"`, or `"w-"`; incremental `"a"`, `"a-"`, and `"r+"`
modes are intentionally rejected because they can retain unrelated arrays from an
older store. Incremental or regional Zarr mutation needs a separate result-store
contract and is not part of this persistence surface.

TAL intentionally has no `ao.io.to_csv(...)` or `AnalysisObject.from_csv(...)` API.
CSV cannot generally preserve an arbitrary xarray Dataset or TAL schema, so presenting
it as an AO round trip would promise semantics the format cannot represent.

## Log Readers and Writers

```python
from tal.io import read_csv_logs, read_ros_logs, write_csv_logs
```

CSV and ROS readers ingest external logs directly into TAL objects. CSV export
is available through `write_csv_logs(...)` when grouped AOs should be written
back to per-run tabular files. ROS readers use message timestamps when
available before falling back to receive/log time.

`read_csv_logs(...)` and `write_csv_logs(...)` are deliberately lossy and are
not inverse operations. CSV output contains tabular observation columns only: there is
no sidecar and no promise to preserve TAL schema, subclass identity, exact dtype,
datetime unit, attributes, encoding, indexes, or chunk topology. Ragged padded tails
are omitted using declared sequence-size metadata. Writing may eagerly materialize the
batch labels needed for path derivation, the effective size coordinate needed for
validation, and the selected valid-region fields.
Reading independently constructs a new AO from the explicit ingest options and the
columns present in the log.

CSV export validates representable fields before eager access, resolves and validates
the export root once even for an empty batch, preflights every destination, validates
the effective sequence-size values, and slices export fields to their valid regions
before realizing one coherent snapshot. An explicit `sequence_size_coord` override wins
before TAL validates the superseded validity block or its declared coordinate; every
other schema block is still validated normally, and the effective coordinate is
validated once. The
overridden declared validity coordinate is omitted from the export projection and is
not executed unless clearing its invalid former validity role leaves it as the declared
sequence parameter. Ignored metadata coordinates and fully excluded padded-tail chunks
are not part of that realization. An empty batch validates its structure, effective
size-coordinate dtype, and export root without realizing non-resident validity values.
The declared parameter coordinate is always part of the projection when it has
sequence semantics, including when it is also the sequence dimension coordinate and
when no data variable is selected.
Exported column names must be non-empty, contain a non-whitespace character, and be
NUL- and BOM-free after string normalization. CSV ingestion applies the same identity
rules to the effective header before pandas can normalize it. TAL opens the file with
UTF-8 BOM handling and Python's newline-aware CSV reader, skips only leading unquoted
space/tab-only physical lines, and reads one logical header, including quoted commas or
newlines. Pandas then parses the complete file once with implicit-index inference
disabled and malformed-line errors enabled. Parser warnings, parser errors, decoding
failures, and backend failures retain the public reader owner.

Fully in-memory exports serialize directly without constructing a Dask graph. Lazy
exports use one shared graph, honor the configured Dask scheduler, and jointly realize
the selected valid rows before filesystem serialization begins. The writer then
serializes each result directly to a short same-directory temporary file. This ordering
prevents fail-fast schedulers from racing temporary-file cleanup, but it may retain the
complete valid export projection in memory.
Destination and prospective-parent collision checks use the same Unicode-normalized,
case-folded filesystem key, and fail before creating any export directory.
The commit owner replaces destinations sequentially in batch order with `os.replace`.
Each replacement is atomic, so a failure preserves the destination currently being
replaced and retains the public writer error owner. The batch is not transactional: a
commit failure may leave an already-written complete prefix. Uncommitted temporary files
are removed, but newly created empty directories may remain.

This is a trusted, stable-filesystem log adapter, not a hardened CSV parser or
multi-file transaction system. Python/pandas own CSV grammar, malformed-record details,
payload-NUL behavior, oversized-field limits, and parser security. Concurrent path or
symlink mutation, rollback of already committed files, and adversarial filesystem changes
are unsupported user/upstream boundaries.

## Deterministic Input Policy

Log ingestion accepts:

- explicit `Mapping[str, str]` label-to-path inputs
- `Sequence[str]` path inputs
- glob strings

Sequences preserve caller order. Glob inputs are expanded and sorted
lexicographically by normalized resolved path. Duplicate paths and derived label
collisions fail closed. CSV sources must be files. ROS sources may also be ROS 2
bag directories, and readers consume the resolved path rather than an unexpanded
input spelling.

Export destinations also fail closed when final paths collide under deterministic
Unicode normalization and case folding, including on case-insensitive filesystems.
Pre-existing destinations that alias one inode and planned file/parent-directory
topology conflicts also fail during preflight.

## CSV and Timestamp Policy

CSV default behavior requires explicit `time_col`. Optional inference is
available only for a small known candidate set and succeeds only when exactly
one candidate exists.

Empty or duplicate CSV headers fail before pandas can normalize their identities;
whitespace-only and later-field BOM identities fail at the same boundary. Validation
skips leading unquoted space/tab-only blank lines and inspects only the effective logical
header record. Quoted delimiters and quoted newlines are delegated to Python's CSV
reader; complete-record parsing is delegated to pandas.
Boolean policy fields require actual `bool` values; strings such as `"false"`
are rejected, and choice-policy fields require strings. `CsvExportOptions.float_format`
accepts a built-in string or `None` and otherwise follows pandas formatting semantics.

CSV integer timestamps remain exact through sorting, record spooling, and final padded
grid construction. Raw timestamp tokens are classified before pandas dtype inference,
including when invalid rows are dropped. Boolean and non-real timestamp columns fail
closed. If integer and floating timestamps are combined within or across files, integers
outside the floating dtype's exact range are rejected instead of being silently rounded.

ROS timestamp precedence is deterministic: `header.stamp` first, then
receive/log time. Empty `0/0` header stamps are treated as missing. The selected
source becomes a signed `int64` Unix-nanosecond parameter coordinate; paired
seconds and nanoseconds never pass through floating point, and ignored sources
are not parsed.
Canonical ROS 2 stamp fields are resolved before ROS 1 aliases, and malformed
backend conversion failures retain the public reader owner.

CSV and ROS readers spool one normalized record at a time to temporary array
files before allocating their final padded grids. They therefore retain at
most one decoded record during normalization rather than every record plus the
complete padded result. Schema finalization adopts those freshly owned grid
buffers without duplicating them; ordinary external `Dataset` construction
still uses isolation copies. Temporary cleanup failures retain the public
reader owner and never replace an already-active primary failure. Cleanup
interruptions such as `KeyboardInterrupt` likewise cannot erase the original
serialization, iteration, or schema failure. Under the default ROS metadata
policy, varying frame identities are detected without
retaining per-message frame strings that will ultimately be omitted. When both metadata
targets are disabled, optional frame fields are not accessed at all.

Custom ROS batch, sequence, validity, and time-coordinate names cannot reuse a
fixed pose payload field such as `translation_x`; those collisions fail before
path resolution or backend access.

## Metadata Promotion

Metadata promotion is controlled by options. Scalar metadata defaults to attrs;
non-scalar metadata is omitted unless explicitly enabled.

The `tal` attr key is reserved for schema and cannot be used as ordinary metadata
promoted to attrs. Attrs occupy a separate xarray namespace, so a batch coordinate
named `tal` remains valid when it has no array-namespace collision. Generated attrs
may likewise use the same name as a data variable, coordinate, or dimension. For example,
a CSV value column named `io_time_source` and a ROS batch dimension named
`ros_timestamp_unit` remain valid under the default attr policy.

CSV metadata columns cannot claim adapter-generated identities such as
`io_time_source` or `io_source_paths`, because the adapter owns those values.
When scalar metadata is configured for batch-coordinate promotion instead of
attrs, potentially scalar generated identities are preflighted against configured
layout and explicit value names. Such collisions fail before path resolution or ROS
backend work rather than after dataset construction.

CSV export uses declared sequence-size metadata when present, so ragged
validity is not guessed from padded rows. Explicit size overrides must contain
real numeric values rather than numeric-looking strings. Explicit value-column
selections must match available numeric columns.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/io
   :nosignatures:

   tal.io.AnalysisObjectIOAccessor.to_zarr
   tal.AnalysisObject.from_zarr
   tal.io.CsvIngestOptions
   tal.io.CsvExportOptions
   tal.io.RosIngestOptions
   tal.io.read_csv_logs
   tal.io.write_csv_logs
   tal.io.read_ros_logs
```

## See Also

- {doc}`analysis-object`
