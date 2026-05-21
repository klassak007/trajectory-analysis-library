(api-catalog)=
# Catalog

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

`Catalog` is TAL's browse/query/extract layer for grouped log-style data. It is
useful before analysis: select records, filter by metadata, and extract
analysis-ready AOs from a larger collection.

`Catalog` does not replace AO-direct I/O. Numeric kernels belong on
`AnalysisObject` and typed objects.

```{contents}
:local:
:depth: 2
```

## Constructor

```python
from tal.catalog import Catalog

catalog = Catalog(value, *, backend="auto", batch_dim=None)
```

Accepted inputs:

- `AnalysisObject`
- `xarray.Dataset`
- `xarray.DataArray`
- `xarray.DataTree`

Grouping is resolved at construction. An explicit `batch_dim` wins when it is
semantically compatible; otherwise TAL can use declared batch roles or backend
defaults. There is no post-construction regrouping API.

## Browse Surface

```python
catalog.sel(...)
catalog.isel(...)
catalog.head(...)
catalog.tail(...)
```

Browse operations return `Catalog` objects. Empty selections return empty
catalogs. Unsupported selector kinds fail closed rather than falling back to
backend-specific behavior.

Read-only properties:

- `backend`
- `batch_dim`
- `group_labels`
- `data`

## Query Surface

```python
catalog.query(where=None, *, opts=None, **filters)
```

The query language is finite and portable:

- structured predicate dictionaries via `where={...}`
- keyword equality shorthand via `**filters`
- operators: `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, `and`, `or`, `not`
- metadata namespaces: `attr.<name>`, `coord.<name>`, `batch.<name>`

String expressions, callables, regex predicates, and backend-specific query
languages are intentionally rejected.

Unqualified metadata names are accepted only when unique across namespaces.
Ambiguous names fail closed.

## Extract Surface

```python
ao = catalog.extract(variables=None, *, opts=None)
```

`extract(...)` materializes an `AnalysisObject` and finalizes schema through
core owners.

Variable semantics:

- `variables=None` extracts all extractable payload variables.
- Unknown variable names fail by default.
- `ignore_missing_vars=True` is opt-in.
- `variables=[]` is deterministic and does not mean "all."

Metadata promotion is options-controlled. Scalar metadata can be promoted to
attrs or batch coordinates; non-scalar metadata is omitted unless explicitly
enabled.

## Guardrails

`Catalog` blocks arithmetic, NumPy coercion, and ufunc dispatch. This preserves
a clean boundary: browse/query/extract in `Catalog`, numeric work in AO or typed
objects.

## API Summary

```{eval-rst}
.. autosummary::
   :toctree: _generated/catalog
   :nosignatures:

   tal.catalog.Catalog
   tal.catalog.Catalog.sel
   tal.catalog.Catalog.isel
   tal.catalog.Catalog.head
   tal.catalog.Catalog.tail
   tal.catalog.Catalog.query
   tal.catalog.Catalog.extract
```
