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

For DataTree catalogs, root batch metadata and other root data that depend on
the resolved batch dimension are selected positionally in child order. Root
coordinate values do not need to equal child labels; reordered, subset, and
empty selections keep the root dataset aligned with the selected children.
Catalog construction fails closed when the root batch length differs from the
number of child groups, before any root coordinate is read positionally.
After a non-empty selection, a later empty selection retains the first selected
child as its extraction template rather than reverting to the source catalog's
first child.

Read-only properties:

- `backend`
- `batch_dim`
- `group_labels`
- `data`

`Catalog.data` returns an independently owned public payload. Eager arrays and
nested metadata are copied immediately. Object-bearing Dask arrays receive
lazy deep-copy tasks, so accessing the property does not compute them and
mutating a normally computed public copy cannot alter Catalog-owned values.
Pandas extension-dtype indexes, including categorical indexes, retain their
dtype and category dtype across the ownership boundary. Registered auxiliary
xarray indexes are rebuilt over detached coordinate variables and remain
registered rather than degrading to plain coordinates; mutable Python objects
stored in pandas-backed indexes are recursively detached from Catalog-owned
values. Registered coordinate attributes and encodings are retained and
independently owned. Non-index categorical coordinates and payload variables retain their
categorical dtype while recursively detaching mutable category objects.

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

DataTree payload variables and empty-selection templates share a child-payload
view. Child-local variables and dimensions remain authoritative, while root
coordinates on those payload dimensions (for example a shared sample or time
coordinate) are retained. Compatible inherited xarray index groups are retained
with those coordinates. Root coordinates spanning the batch dimension plus
payload dimensions are selected row-by-row in child order. Child-local
coordinates with the same name take precedence. Pure root batch coordinates
and unrelated root metadata remain catalog metadata rather than becoming child
payload dimensions. Child-local auxiliary coordinates are classified as
batch-dependent from structure before concat, so Dask-backed coordinate values
are not computed merely to compare children. Dimension coordinates remain
invariant even without a registered xarray index, and extraction does not
synthesize an index for an originally unindexed coordinate. When an auxiliary
coordinate is absent from one child, root metadata with the same name supplies
its fallback when compatible; otherwise a lazy missing-value coordinate
preserves the output topology. Same-name child-local coordinates are checked
before variable filtering can hide conflicts. Coordinates with the same named
dimensions are transposed to one canonical order; different named-dimension
sets fail closed rather than broadcasting into a Cartesian coordinate. Selected payload
variables follow the same rule: dimension-order permutations are canonicalized,
while different named-dimension sets fail before concat can Cartesian-expand
them. Constant child-local scalar coordinates are not promoted a second time
after they become batch-dependent result coordinates. Nested constant metadata
uses missing-aware equality, including pandas missing scalars inside sequences.
Empty and non-empty
extraction outputs own their invariant coordinate buffers independently of
internally shared Catalog datasets. They also own mutable object-valued
batch-coordinate elements and mutable objects stored in payload variables,
including structured dtypes with object fields. Numeric payload variables are
not copied, and Dask-backed object isolation remains lazy. Mutable dataset
attributes and dataset encodings are detached. Variable attributes are detached.
All variable encodings are also detached at the result boundary, including
registered index-coordinate encodings; they remain present, and editing an
extracted object cannot modify an internal template.

Dataset-backed extraction provides the same result-ownership guarantee. Eager
payload and coordinate buffers are copied, while Dask-backed arrays remain
lazy. Object-value or metadata isolation failures retain the calling Catalog
operation in their error message, including failures raised during later Dask
computation. Structured coordinates containing object fields are supported
when each group or a compatible root defines them; a sparse structured
coordinate fails closed because TAL cannot invent an unambiguous structured
missing value.

Metadata promotion is options-controlled. Scalar metadata can be promoted to
attrs or batch coordinates; non-scalar metadata is omitted unless explicitly
enabled. Default promotion preserves structurally represented child coordinates
without comparing their values. Selecting a non-default target reconciles those
coordinates with the requested attr/none policy; this may realize Dask-backed
scalar metadata when constant-versus-varying classification is required.
Schema-owned parameter and sequence-size coordinates are always preserved and
are never treated as promotable metadata. When both metadata targets are
`none`, extraction drops eligible metadata coordinates structurally without
projecting or computing their values. Empty DataTree selections apply the same
target policy to retained-template metadata: default scalar metadata becomes a
zero-length batch coordinate, while `attr` and `none` move or omit it exactly as
they do for non-empty extraction. Sequence- and array-valued child attributes
remain one metadata cell per group rather than acquiring an accidental array
dimension. Pandas and NumPy missing scalars compare as missing metadata values,
so a consistently missing field remains scalar rather than being silently
classified as varying.

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
