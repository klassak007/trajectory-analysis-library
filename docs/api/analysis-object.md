(api-analysis-object)=
# `AnalysisObject`

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

`AnalysisObject` is TAL's core container: an `xarray.Dataset` with schema
metadata stored in `ds.attrs["tal"]`. It keeps the base object small and exposes
domain behavior through accessors such as `ao.param`, `ao.events`, `ao.group`,
`ao.combine`, `ao.components`, `ao.frames`, `ao.io`, and `ao.viz`.

Use this page for the AO contract itself. Parameter operations, events,
reducers, frame graph utilities, and typed wrappers have dedicated reference
pages.

```{contents}
:local:
:depth: 2
```

## Construction

```python
AnalysisObject(data: xr.Dataset | xr.DataArray)
```

`xr.Dataset` inputs are accepted directly. `xr.DataArray` inputs are promoted
to single-variable datasets. Other input types raise `TypeError`.

The constructor isolates the input dataset for mutation safety and validates an
existing TAL schema when one is present.

## Schema-Aware Construction

```python
AnalysisObject.from_data(
    data,
    *,
    sequence_dim=None,
    batch_dims=(),
    core_dims=(),
    param_coord=None,
    sequence_size_coord=None,
    layout="left_packed",
    validate=True,
)
```

`from_data(...)` is the preferred boundary when data enters TAL. It applies
roles, parameter coordinate metadata, and validity metadata before the final
validation pass.

Key rules:

- `core_dims` and `batch_dims` may be declared without `sequence_dim`.
- `param_coord` and `sequence_size_coord` require `sequence_dim`.
- When no roles are declared, runtime consumers treat the object as core-only
  by default.

## Data Access

| Attribute or Method | Contract |
| --- | --- |
| `ao.data` | Mutation-safe deep copy of the underlying dataset. |
| `ao.as_dataset()` | Mutation-safe dataset copy. |
| `ao.unsafe_data` | Internal backing dataset reference for low-level inspection. |
| `ao.to_dataarray(name=None)` | Convert a single-variable AO to a `DataArray`; multi-variable datasets fail. |

Prefer `ao.data` in notebooks and user code. Use `ao.unsafe_data` only when you
need to inspect the exact backing store or schema consumed by operations.

## Accessors

| Accessor | Purpose |
| --- | --- |
| `ao.param` | Parameter-domain indexing, interpolation, resampling, and synchronization helpers. |
| `ao.events` | Condition masks, boundary tables, intervals, and event windows. |
| `ao.group` | Grouped layouts and grouped reducers. |
| `ao.combine` | Schema-aware concat, merge, align, and core assembly operations. |
| `ao.components` | Component registry definition, extraction, patching, and composition. |
| `ao.frames` | Frame ID metadata and runtime graph bridge operations. |
| `ao.io` | Canonical AO-direct persistence to Zarr. |
| `ao.viz` | Optional plotting and explorer helpers. |

## Structural Operations

Schema-safe structural wrappers return new AOs and repair metadata after
dataset shape or coordinate changes:

```python
ao.isel(...)
ao.sel(...)
ao.where(...)
ao.drop_vars(...)
ao.rename(...)
ao.transpose(...)
```

Role references, parameter metadata, validity metadata, and component registry
entries are preserved, rewritten, or pruned so the output schema remains
truthful.

## Reducers

`AnalysisObject` exposes:

```text
mean, sum, std, var, median, min, max, count, any, all
```

Reducers preserve TAL semantics where possible and remove schema metadata that
is no longer truthful after a dimension is reduced. See {doc}`reducers`.

## Operand Intent Helpers

`.a(...)` attaches alignment intent to an operand:

```python
left = ao.a(on="sequence", sequence_join="inner", batch_join="exact")
```

`.b()` marks an operand as willing to broadcast under semantic broadcast rules:

```python
out = ao + bias.b()
```

Both helpers return new AO views and do not mutate source data. The intent is
consumed later by supported arithmetic, ufunc, linalg, or spatial planners.

## Schema Writers

Each writer delegates to the core schema API and returns a new AO:

```python
ao.set_roles(...)
ao.set_param_coord(...)
ao.set_validity(...)
ao.merge_schema(...)
ao.validate_schema()
```

Use these when semantics are discovered after construction. Prefer
`from_data(...)` when the semantics are already known.

## Persistence Loaders

```python
AnalysisObject.from_zarr(...)
```

Zarr is the AO-direct persistence format. CSV is available only through the
lossy log adapters described in {doc}`io`.

## Operator Overloads

Arithmetic and ordering comparisons are routed through TAL's ufunc owners:

- arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`, unary `+/-`, `abs(...)`
- ordering comparisons: `<`, `<=`, `>`, `>=`, returning `Condition`
- expression equality: use `tal.ufuncs.equal(...)` and
  `tal.ufuncs.not_equal(...)`

Object equality (`==`, `!=`) keeps identity-style Python semantics for object
safety and hashability.

## Errors

- `SchemaError` for schema validation or writer failures.
- `TypeError` for invalid constructor inputs.
- `ValueError` for invalid structural, conversion, alignment, or option
  choices.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/analysis_object
   :nosignatures:

   tal.AnalysisObject
   tal.AnalysisObject.as_dataset
   tal.AnalysisObject.to_dataarray
   tal.AnalysisObject.isel
   tal.AnalysisObject.sel
   tal.AnalysisObject.where
   tal.AnalysisObject.drop_vars
   tal.AnalysisObject.rename
   tal.AnalysisObject.transpose
   tal.AnalysisObject.from_data
   tal.AnalysisObject.set_roles
   tal.AnalysisObject.set_param_coord
   tal.AnalysisObject.set_validity
   tal.AnalysisObject.merge_schema
   tal.AnalysisObject.validate_schema
   tal.AnalysisObject.a
   tal.AnalysisObject.b
   tal.AnalysisObject.events
   tal.AnalysisObject.components
   tal.AnalysisObject.group
   tal.AnalysisObject.combine
   tal.AnalysisObject.frames
   tal.AnalysisObject.viz
```

## See Also

- {doc}`schema`
- {doc}`timebase`
- {doc}`events`
- {doc}`components`
- {doc}`reducers`
