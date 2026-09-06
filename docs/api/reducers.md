(api-reducers)=
# Reducers

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

Reducers summarize AO data while keeping schema metadata truthful. They are
available on `AnalysisObject` and on grouped views returned by
`ao.group.groupby(...)` or `ao.group.groupby_bins(...)`.

## Surface

```text
mean, sum, std, var, median, min, max, count, any, all
```

All reducers accept `dim` and `validate`.

- `dim=None` reduces over all reducible non-component dimensions.
- Explicit reduction over required typed component dimensions fails closed.
- Non-participating variables are excluded from output.

## Variable Domains

- `mean`, `sum`, `std`, `var`, `median`, `min`, `max`, and `count` operate on
  numeric variables.
- `any` and `all` operate on numeric and boolean variables.
- Reducers fail closed when no eligible variables exist.

## Ragged Validity

When left-packed validity metadata is present, padded tails are excluded
structurally before reduction. This is independent of `skipna`.

`skipna=False` still matters inside the valid prefix: missing values inside the
active domain poison reducers according to xarray-style semantics.

## Empty Reductions

`min` and `max` return NaN when a dimension being reduced has length zero, with
either `skipna` setting. Remaining dimensions, coordinates, and variable
attributes are preserved. Floating and complex dtypes are retained; integer
inputs promote to `float64`. Dask-backed results remain lazy.

This also applies to explicitly included empty bins in grouped reductions.
When no groups remain, the result has a zero-length group dimension. An empty
dimension that is not being reduced does not trigger this rule, and `dim=()`
remains a no-op.

## Weighted Reductions

Weighted support is intentionally narrow:

- supported: `mean`, `sum`
- unsupported: `std`, `var`, `median`, `min`, `max`, `count`, `any`, `all`

Accepted weights for weighted `mean` and `sum`:

- aligned `xr.DataArray`
- 1-D `ndarray` when eligible payload variables collectively use exactly one
  active reduced dimension
- per-dimension mapping for multi-dimension reduction

Indexed `xr.DataArray` weights require exact label alignment. One-dimensional
`ndarray` weights are positional and size-checked without manufacturing an
xarray index. Mapping factors follow the rule for each factor's representation.
Weights fail closed when no eligible payload variable uses a reduced dimension;
they are never silently ignored by a no-op or coordinate-only reduction.
Negative weights are rejected, including each mapping factor before factors
are multiplied. Another negative or zero factor cannot hide an invalid sign.
Factor entries used only in structurally invalid padded tails are excluded.
Empty payload domains have no participating factor entries; mapping completeness,
factor lengths, and index alignment are still checked, but factors are not
value-validated or multiplied.
These value checks run during reduction, after grouped key planning.
Non-finite weights are handled according to
`skipna`.

```python
import numpy as np
import xarray as xr
from tal.core import AnalysisObject

values = AnalysisObject.from_data(
    xr.DataArray([1.0, 2.0, 3.0], dims="trial",
                 coords={"trial": ["a", "b", "c"]}, name="value"),
    batch_dims=("trial",),
)
positional = np.array([1.0, 2.0, 1.0])
indexed = xr.DataArray(positional, dims="trial",
                       coords={"trial": ["a", "b", "c"]})
for weights in (positional, indexed):
    result = values.mean(dim="trial", weights=weights)
    np.testing.assert_allclose(result.as_dataset()["value"], 2.0)
```

## Grouped Reducers

`ao.group.groupby(...)` chooses the topology from the source AO:

- A source with a sequence role returns `GroupedView`, even when it also has
  batch roles. It supports
  `materialize`, `padded`, `stacked`, and reducers. Reducer options use
  `GroupMaterializeOptions`.
- A source without a sequence role and with a batch role returns
  `BatchGroupedView`. It exposes reducers only, groups the primary batch lane,
  and replaces that lane with the group dimension. Reducer options use
  `BatchGroupReduceOptions`.
- A source with neither role fails before grouping-key lookup or realization.

`GroupedView` and `BatchGroupedView` are return-only result types. Construct
them through `ao.group.groupby(...)`; their direct constructors are not public
entrypoints.

For batch-only grouping, `dim=None` reduces the primary batch lane. Explicit
dimensions must include that lane. Supplemental batch/core dimensions and
primary-independent variables are retained. Group keys are matched exactly by
xarray index when indexed; purely unindexed keys use positional size matching.
Batch-only grouping requires `preserve_batch=False`; customized `member_dim`
and `sequence_index_coord` values are sequence-only and are rejected.

```python
import numpy as np
import xarray as xr
from tal.core import AnalysisObject, BatchGroupReduceOptions

distance = AnalysisObject.from_data(
    xr.Dataset(
        {"distance": (("trial", "sample"), [[4., 2.], [3., 1.], [8., 6.]])},
        coords={"trial": ["a", "b", "c"], "sample": [0, 1],
                "outcome": ("trial", ["win", "loss", "win"])},
    ),
    sequence_dim="sample", batch_dims=("trial",),
)
per_trial = distance.min(dim="sample")
by_outcome = per_trial.group.groupby("outcome").mean(
    dim="trial",
    opts=BatchGroupReduceOptions(group_dim="outcome_group"),
)
result = by_outcome.as_dataset()["distance"]
assert result.dims == ("outcome_group",)
assert result.coords["outcome_group"].data.tolist() == ["win", "loss"]
np.testing.assert_allclose(result, [4.0, 1.0])
```

Grouped reducers reuse the ordinary AO reducer and finalization owners. They do
not return an xarray `DatasetGroupBy`.

## Typed Reducers

Typed math stays domain-local. Explicitly owned typed reducers, such as
`Rotation.mean(...)`, preserve typed outputs when invariants hold. Inherited AO
reducers on typed subclasses demote to base `AnalysisObject` when the typed
contract no longer applies.

`Rotation.mean` preserves lazy Dask arrays, including empty groups. Its numerical
kernel follows xarray's gufunc requirement that each reduced dimension and the
quaternion component dimension occupy one chunk. Group assembly can produce
multiple chunks on the group dimension; rechunk that dimension before a further
Rotation mean over groups.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/reducers
   :nosignatures:

   tal.AnalysisObject.mean
   tal.AnalysisObject.sum
   tal.AnalysisObject.std
   tal.AnalysisObject.var
   tal.AnalysisObject.median
   tal.AnalysisObject.min
   tal.AnalysisObject.max
   tal.AnalysisObject.count
   tal.AnalysisObject.any
   tal.AnalysisObject.all
   tal.core.BatchGroupedView
   tal.core.BatchGroupReduceOptions
```

## See Also

- {doc}`analysis-object`
- User guide: {doc}`../user-guide/indexing`
