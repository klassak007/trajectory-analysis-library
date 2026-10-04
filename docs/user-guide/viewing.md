(viewing-debugging)=
# Viewing and Debugging

Inspect an AnalysisObject directly with `print(ao)` or by displaying `ao` in a
notebook. Its display combines xarray's dimensions, coordinates, variables, and
ordinary attributes with a compact summary of stored TAL declarations.
Typed objects show their concrete class; spatial types also summarize their
stored frames and representation.

The **Role** column under Coordinates identifies declared `sequence`, `batch`,
and `core` dimension coordinates and the `parameter` coordinate. A coordinate
can have multiple roles, such as `sequence, parameter`. Auxiliary coordinates
do not inherit the roles of their dimensions. Xarray's index indicators retain
their usual meaning; no extra symbol is needed. Dimensions without coordinate
labels remain described in Dimensions and the TAL summary.

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

print(ao)
print(out)
with xr.set_options(display_width=90, display_max_rows=8):
    preview = repr(out)

safe_snapshot = ao.as_dataset()
out_snapshot = out.as_dataset()
roles = read_roles(safe_snapshot)
param_name = read_param_coord_name(safe_snapshot)
before_schema = safe_snapshot.attrs["tal"]
after_schema = out_snapshot.attrs["tal"]
```

Direct display shares the existing data without copying numerical buffers or
executing Dask tasks. In notebooks, the collapsed **TAL schema** section reveals
bounded stored metadata. Expanding it does not validate or compute the object.
Depth, item, and text limits have explicit omission markers; custom schema values
show a type placeholder. User attributes retain xarray's normal display behavior.

Transform-backed coordinates show their names, index type, and an explicit
omitted-values marker. This includes xarray's transform-backed `RangeIndex`.
Displaying them never invokes a coordinate transform. Ordinary indexes and
coordinates use xarray's native previews.

The display describes stored declarations, including deliberately unvalidated
ones; it is not a validation certificate. Unreadable metadata rows are marked
unavailable while readable rows and data remain visible. Use a safe Dataset
snapshot when you need programmatic inspection or exploratory mutation. Keep
the AO open while working with lazy data.

## Debugging Flow

1. Inspect `ao` directly, then capture `snapshot = ao.as_dataset()` if needed.
2. Compare the displayed topology with the declared TAL roles.
3. Inspect `read_roles(snapshot)` and `read_param_coord_name(snapshot)`.
4. Compare `snapshot.attrs["tal"]` with the output snapshot.
5. Check `ao.frames.ids()` before frame-aware spatial operations.

In notebooks, display `ao` and `out` directly to compare results. Xarray's
`display_style="text"` option selects the text preview; rich-formatting
incompatibility also falls back to escaped text. No separate TAL display
configuration is required.

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
