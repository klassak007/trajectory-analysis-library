(viewing-debugging)=
# Viewing and Debugging

TAL is easiest to debug when you inspect the data carrier and the semantic
carrier together. Shapes and coordinates tell you what xarray sees; the
`tal` schema tells you what TAL operations will preserve, align, reduce, or
reject.

## Minimal Example

<!-- example-id: UG-VIEWING-SCHEMA -->
```python
import numpy as np
import xarray as xr
from tal.core import AnalysisObject
from tal.core.schema_read import read_param_coord_name, read_roles

sample = np.arange(3)
ao = AnalysisObject.from_data(
    xr.Dataset(
        {"value": ("sample", np.array([1.0, 2.0, 3.0]))},
        coords={"sample": sample, "time_s": ("sample", np.array([0.0, 0.1, 0.2]))},
    ),
    sequence_dim="sample",
    core_dims=(),
    param_coord="time_s",
    validate=True,
)
out = ao.param.at([0.05, 0.15], on="time_s")

safe_snapshot = ao.data
backing_store = ao.unsafe_data
roles = read_roles(backing_store)
param_name = read_param_coord_name(backing_store)
before_schema = ao.unsafe_data.attrs["tal"]
after_schema = out.unsafe_data.attrs["tal"]
```

Use the safe copy for notebook inspection and exploratory display. Use
`unsafe_data` when you need to inspect the exact backing dataset or schema
being consumed by an operation.

## Debugging Flow

1. Inspect `ao.unsafe_data.dims`, `ao.unsafe_data.sizes`, and
   `ao.unsafe_data.coords`.
2. Inspect `read_roles(ao.unsafe_data)` and
   `read_param_coord_name(ao.unsafe_data)`.
3. Compare `ao.unsafe_data.attrs["tal"]` before and after the operation.
4. Check `ao.frames.ids()` before frame-aware spatial operations.

In notebooks, `display(ao.unsafe_data)` and `display(out.unsafe_data)` are
often enough to spot missing coordinates or unexpected topology changes.

## Visualization

When optional visualization backends are installed, use `ao.viz.line(...)`,
`ao.viz.scatter(...)`, or `ao.viz.component(...)` for fast inspection. Prefer
explicit `var`, `x`, and `group_key` options when an AO has multiple variables
or batch dimensions.

## What Usually Goes Wrong

- The AO has no resolved `sequence_dim` for sequence-dependent APIs.
- The `on=` coordinate differs from the declared `param_coord`.
- A reduction removed a dimension and schema roles changed accordingly.
- Frame-dependent spatial operations are missing parent or child frame tags.
- Optional visualization backends are not installed.

## See Also

- {doc}`core_concepts`
- {doc}`indexing`
- {doc}`frames`
- API: {doc}`../api/viz`
