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

safe_snapshot = ao.as_dataset()
out_snapshot = out.as_dataset()
roles = read_roles(safe_snapshot)
param_name = read_param_coord_name(safe_snapshot)
before_schema = safe_snapshot.attrs["tal"]
after_schema = out_snapshot.attrs["tal"]
```

Use the safe copy for notebook inspection and exploratory display. When a large
payload makes copying undesirable and inspection is strictly read-only, request
one shallow snapshot explicitly and keep the AO open while using lazy data.

## Debugging Flow

1. Capture one `snapshot = ao.as_dataset()`.
2. Inspect `snapshot.dims`, `snapshot.sizes`, and `snapshot.coords`.
3. Inspect `read_roles(snapshot)` and `read_param_coord_name(snapshot)`.
4. Compare `snapshot.attrs["tal"]` with the output snapshot.
5. Check `ao.frames.ids()` before frame-aware spatial operations.

In notebooks, `display(ao.as_dataset())` and `display(out.as_dataset())` are
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
