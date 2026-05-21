(api-timebase)=
# Parameter Operations

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

TAL parameter operations evaluate, select, resample, and synchronize AOs on a
semantic coordinate. The coordinate is often time, but the API is intentionally
generic: any declared ordered coordinate can be used as the operation domain.

```{contents}
:local:
:depth: 2
```

## Param Accessor

```python
ao.param.index(query, *, on=None, opts=None, ...)
ao.param.sel(query, *, on=None, opts=None, validate=True, ...)
ao.param.at(query, *, on=None, opts=None, validate=True, ...)
ao.param.resample_to(grid, *, on=None, opts=None, validate=True, ...)
ao.param.interp_like(other, *, on=None, batch_join="exact", validate=True, ...)
```

`on` selects the coordinate used as the parameter domain. If `on` is omitted,
TAL uses the AO's declared `param_coord`.

## Selection vs Evaluation

| Method | Contract |
| --- | --- |
| `index(...)` | Build nearest sample-index mappings for parameter queries. |
| `sel(...)` | Select existing samples nearest to parameter query values. |
| `at(...)` | Evaluate or interpolate values on query points. |
| `resample_to(...)` | Evaluate the AO on an explicit target grid. |
| `interp_like(...)` | Evaluate one AO on another AO's parameter grid. |

Use `sel(...)` when you want recorded samples. Use `at(...)` or
`resample_to(...)` when query values may fall between recorded samples.

## Synchronization

```python
from tal.core import ParamSyncOptions, synchronize

left_sync, right_sync = synchronize(
    [left, right],
    on="time_s",
    opts=ParamSyncOptions(join="domain", how="interp", batch_join="inner"),
)
```

`synchronize(...)` resolves a shared parameter grid and evaluates each input AO
according to the selected policy.

Common `ParamSyncOptions` choices:

- `join="exact"` requires matching parameter domains.
- `join="domain"` builds a shared domain grid.
- `how="interp"` interpolates onto the resolved grid.
- `batch_join` controls how batch labels are joined.

## Invariants

- Parameter operations require sequence semantics.
- Interpolation requires monotonic parameter values on each active batch row.
- Ragged validity metadata limits the active domain.
- Batch labels must satisfy the requested join policy.
- Ambiguous or missing parameter coordinates fail before numeric work starts.

## Typed Spatial Behavior

Typed spatial objects may override interpolation defaults while keeping the same
accessor shape. `Rotation.param.at(...)` can use rotation-aware interpolation;
`Pose.param.at(...)` can interpolate translation and rotation with different
policies.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/timebase
   :nosignatures:

   tal.core.param_ops.accessor.ParamAccessor.index
   tal.core.param_ops.accessor.ParamAccessor.sel
   tal.core.param_ops.accessor.ParamAccessor.at
   tal.core.param_ops.accessor.ParamAccessor.resample_to
   tal.core.param_ops.accessor.ParamAccessor.interp_like
   tal.core.synchronize
   tal.core.ParamSelectOptions
   tal.core.ParamEvalOptions
   tal.core.ParamSyncOptions
```

## See Also

- {doc}`schema`
- {doc}`analysis-object`
- User guide: {doc}`../user-guide/time`
