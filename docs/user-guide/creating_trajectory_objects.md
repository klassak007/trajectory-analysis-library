(creating-trajectory-objects)=
# Creating Trajectory Objects

An `AnalysisObject` is the boundary where ordinary labeled data becomes
trajectory-aware. The best time to declare TAL semantics is when data enters
your analysis: name the sequence axis, batch axes, payload axes, parameter
coordinate, and any validity coordinate while the layout is still obvious.

## Construction Pattern

Start with an `xarray.Dataset` or `xarray.DataArray`, then call
`AnalysisObject.from_data(...)` with the roles you know. TAL validates the
schema and returns an object whose later operations can preserve those
semantics.

## Minimal Example

<!-- example-id: UG-CREATING-SEQUENCE-AO -->
```python
import numpy as np
import xarray as xr
from tal.core import AnalysisObject

core_only = AnalysisObject.from_data(
    xr.Dataset({"value": ("axis", np.array([1.0, 2.0, 3.0]))}, coords={"axis": ["x", "y", "z"]}),
    core_dims=("axis",),
    validate=True,
)

batch_core = AnalysisObject.from_data(
    xr.Dataset(
        {"value": (("trial", "axis"), [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])},
        coords={"trial": ["a", "b"], "axis": ["x", "y", "z"]},
    ),
    batch_dims=("trial",),
    core_dims=("axis",),
    validate=True,
)

sample = np.arange(5)
trial = ["a", "b"]
time_s = np.linspace(0.0, 0.4, sample.size)
values = np.stack([
    np.column_stack([time_s, time_s**2, np.zeros_like(time_s)]),
    np.column_stack([time_s + 1.0, time_s**2, np.ones_like(time_s)]),
])

full = AnalysisObject.from_data(
    xr.Dataset(
        {"position": (("trial", "sample", "axis"), values)},
        coords={
            "trial": trial,
            "sample": sample,
            "axis": ["x", "y", "z"],
            "time_s": (("trial", "sample"), np.broadcast_to(time_s, (2, sample.size))),
            "group_size": ("trial", np.array([5, 4], dtype=np.int64)),
        },
    ),
    sequence_dim="sample",
    batch_dims=("trial",),
    core_dims=("axis",),
    param_coord="time_s",
    sequence_size_coord="group_size",
    validate=True,
)

updated = AnalysisObject(
    xr.Dataset(
        {"position": (("trial", "sample", "axis"), values)},
        coords={
            "trial": trial,
            "sample": sample,
            "axis": ["x", "y", "z"],
            "time_s": (("trial", "sample"), np.broadcast_to(time_s, (2, sample.size))),
            "group_size": ("trial", np.array([5, 4], dtype=np.int64)),
        },
    )
)
updated = updated.set_roles(sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",), validate=True)
updated = updated.set_param_coord(name="time_s", validate=True)
updated = updated.set_validity(sequence_size_coord="group_size", validate=True)
```

The three examples show the same constructor pattern at different levels of
structure: payload-only data, batched payload data, and full trajectory data
with sequence, batch, parameter, and validity metadata.

## When To Set Semantics Later

Prefer `AnalysisObject.from_data(...)` when you already know the roles. Use
`ao.set_roles(...)`, `ao.set_param_coord(...)`, and `ao.set_validity(...)` when
the data arrives before its semantics are known or when a workflow discovers
metadata after loading.

Delayed schema writes are still explicit and validated. They return new AOs
rather than mutating the source object in place.

## `ao.data` vs `ao.unsafe_data`

- `ao.data` and `ao.as_dataset()` return mutation-safe deep copies.
- `ao.unsafe_data` returns the backing dataset for low-level inspection and
  advanced debugging.
- Prefer `ao.data` in notebooks unless you intentionally need direct backing
  store access.

## What Usually Goes Wrong

- `param_coord` without a declared `sequence_dim`.
- `sequence_size_coord` without a declared `sequence_dim`.
- A role name that is a coordinate but not a dimension.
- Multi-variable data passed to an operation that needs one unambiguous numeric
  payload.

Construction-time mistakes are cheaper to fix than downstream alignment or
interpolation errors, so it is worth declaring semantics early.

## Quick Checks

- Inspect `full.unsafe_data`.
- Check `full.unsafe_data.attrs["tal"]`.
- Use `full.to_dataarray(name="position")` only when one data variable should
  become the payload boundary.

## See Also

- {doc}`core_concepts`
- {doc}`time`
- {doc}`events`
- API: {doc}`../api/analysis-object`
