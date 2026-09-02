(indexing-selection)=
# Indexing and Selection

TAL keeps ordinary xarray selection and parameter-domain selection separate.
Use `.sel()` and `.isel()` when you mean labels or integer positions. Use
`ao.param` when you mean a semantic progression coordinate such as time,
distance, phase, or iteration.

## Minimal Example

<!-- example-id: UG-INDEXING-PARAM-QUERY -->
```python
import numpy as np
import xarray as xr
from tal.core import AnalysisObject, ParamEvalOptions

sample = np.arange(6)
time_s = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.7])

ao = AnalysisObject.from_data(
    xr.Dataset(
        {"signal": (("trial", "sample"), [[0.0, 1.0, 0.0, 2.0, 1.0, 0.0]])},
        coords={
            "trial": ["t0"],
            "sample": sample,
            "time_s": (("trial", "sample"), time_s[None, :]),
            "group_size": ("trial", np.array([sample.size], dtype=np.int64)),
        },
    ),
    sequence_dim="sample",
    batch_dims=("trial",),
    core_dims=(),
    param_coord="time_s",
    sequence_size_coord="group_size",
    validate=True,
)

head = ao.isel(sample=slice(0, 3))
trial_t0 = ao.sel(trial="t0")
positive = ao.where(ao.as_dataset()["signal"] > 0.0)
index = ao.param.index([0.18, 0.52], on="time_s")
nearest = ao.param.sel([0.18, 0.52], on="time_s")
interp = ao.param.at([0.18, 0.52], on="time_s", opts=ParamEvalOptions(method="linear"))
```

The first three operations are structural xarray-style operations. The last
three ask TAL to interpret `time_s` as the query domain.

## Choosing The Right Selection API

| API | Query space | Typical use |
| --- | --- | --- |
| `ao.isel(...)` | integer positions | take the first samples, select row offsets |
| `ao.sel(...)` | coordinate labels | select a named trial, scenario, or label |
| `ao.where(...)` | boolean masks | mask values while preserving schema |
| `ao.param.index(...)` | parameter values | debug or reuse nearest sample locations |
| `ao.param.sel(...)` | parameter values | select recorded samples nearest to query values |
| `ao.param.at(...)` | parameter values | interpolate or evaluate on a query grid |

## What Usually Goes Wrong

- Parameter APIs require sequence semantics and a parameter coordinate.
- Interpolation requires valid monotonic parameter domains inside each batch
  row.
- Shared dimensions align by labels, not by raw position.
- `ao.param.sel(...)` selects recorded samples; use `ao.param.at(...)` when you
  need interpolation.

## Quick Checks

- Inspect `ao.as_dataset().sizes`.
- Inspect `ao.as_dataset().coords["time_s"]`.
- Compare `index`, `nearest.as_dataset()`, and `interp.as_dataset()` to confirm
  whether you wanted sample lookup or interpolation.

## See Also

- {doc}`time`
- {doc}`events`
- {doc}`viewing`
