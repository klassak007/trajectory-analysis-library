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

## Weighted Reductions

Weighted support is intentionally narrow:

- supported: `mean`, `sum`
- unsupported: `std`, `var`, `median`, `min`, `max`, `count`, `any`, `all`

Accepted weights for weighted `mean` and `sum`:

- aligned `xr.DataArray`
- 1-D `ndarray` for a single-dimension reduction
- per-dimension mapping for multi-dimension reduction

Weights require exact label alignment. Negative weights are rejected. Non-finite
weights are handled according to `skipna`.

## Grouped Reducers

Grouped reducers reuse grouped materialization and AO reducer owners. By
default, grouped `dim=None` reduces over the grouped member axis. Grouped
outputs remain AOs with grouped topology metadata finalized after reduction.

## Typed Reducers

Typed math stays domain-local. Explicitly owned typed reducers, such as
`Rotation.mean(...)`, preserve typed outputs when invariants hold. Inherited AO
reducers on typed subclasses demote to base `AnalysisObject` when the typed
contract no longer applies.

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
```

## See Also

- {doc}`analysis-object`
- User guide: {doc}`../user-guide/indexing`
