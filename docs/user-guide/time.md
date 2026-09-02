(timebase-resampling)=
# Time and Parameter Operations

TAL's parameter APIs evaluate, resample, and synchronize trajectories on a
declared coordinate. The coordinate is often time, but the same surface works
for distance along a path, gait phase, optimization iteration, or any ordered
domain that should drive interpolation.

## Minimal Example

<!-- example-id: UG-TIME-SYNCHRONIZE -->
```python
import numpy as np
import xarray as xr
from tal.core import AnalysisObject, ParamEvalOptions, ParamSyncOptions, synchronize

imu_t = np.linspace(0.0, 1.0, 11)
gps_t = np.linspace(0.0, 1.0, 3)

imu = AnalysisObject.from_data(
    xr.Dataset(
        {"accel": (("trial", "sample"), np.sin(2.0 * np.pi * imu_t)[None, :])},
        coords={
            "trial": ["run_0"],
            "sample": np.arange(imu_t.size),
            "time_s": (("trial", "sample"), imu_t[None, :]),
            "group_size": ("trial", np.array([imu_t.size], dtype=np.int64)),
        },
    ),
    sequence_dim="sample",
    batch_dims=("trial",),
    core_dims=(),
    param_coord="time_s",
    sequence_size_coord="group_size",
    validate=True,
)

gps = AnalysisObject.from_data(
    xr.Dataset(
        {"speed": (("trial", "sample"), (0.5 + 0.1 * np.cos(2.0 * np.pi * gps_t))[None, :])},
        coords={
            "trial": ["run_0"],
            "sample": np.arange(gps_t.size),
            "time_s": (("trial", "sample"), gps_t[None, :]),
            "group_size": ("trial", np.array([gps_t.size], dtype=np.int64)),
        },
    ),
    sequence_dim="sample",
    batch_dims=("trial",),
    core_dims=(),
    param_coord="time_s",
    sequence_size_coord="group_size",
    validate=True,
)

query = np.array([0.05, 0.25, 0.75])
imu_at = imu.param.at(query, on="time_s", opts=ParamEvalOptions(method="linear", query_dim="query"))
imu_rs = imu.param.resample_to(np.linspace(0.0, 1.0, 6), on="time_s")
gps_on_imu = gps.param.interp_like(imu, on="time_s", batch_join="inner")
synced_imu, synced_gps = synchronize(
    [imu, gps],
    on="time_s",
    opts=ParamSyncOptions(join="domain", how="interp", batch_join="inner", query_dim="query"),
)
```

The example keeps the fast IMU stream and slower GPS stream in their original
sample layouts until a query asks for a common parameter grid.

## Parameter Operations

| Operation | Use When |
| --- | --- |
| `ao.param.at(query, on=...)` | You have explicit query points. |
| `ao.param.resample_to(grid, on=...)` | You want a new regular or irregular grid. |
| `left.param.interp_like(right, on=...)` | One AO should be evaluated on another AO's parameter grid. |
| `tal.core.synchronize([...], on=..., opts=...)` | Several AOs should land on one resolved parameter grid. |

Parameter semantics are explicit rather than guessed. The same public surface
works for clocks, distances, phases, or experiment indices as long as the
chosen domain is ordered and declared. Numeric and `datetime64` parameter
coordinates are supported. Datetime64 queries can use NumPy datetime64 values,
pandas timestamps, Python datetimes, or labeled xarray arrays; synchronization
tolerance for datetime64 params must be timedelta-like.

## What Usually Goes Wrong

- Missing `param_coord` or an incorrect `on=` name raises before
  interpolation.
- Non-monotonic parameter rows fail on interpolation paths.
- Batch labels must satisfy the requested `batch_join` policy.
- `ParamSyncOptions(join="exact")` requires matching domains;
  `join="domain"` builds a shared domain grid.

## Quick Checks

- Inspect `imu_at.as_dataset().coords["time_s"]`.
- Inspect `imu_rs.as_dataset().sizes`.
- Compare `synced_imu.as_dataset().coords["time_s"]` and
  `synced_gps.as_dataset().coords["time_s"]`.

## See Also

- {doc}`indexing`
- {doc}`events`
- API: {doc}`../api/timebase`
