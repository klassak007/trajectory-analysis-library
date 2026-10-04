(numpy-interop)=
# NumPy Interop

TAL supports NumPy-style elementwise work through `tal.ufuncs` and selected AO
operators. The point is not to bypass xarray; it is to keep xarray's
label-aware alignment while preserving TAL schema metadata on the result.

## Minimal Example

<!-- example-id: UG-NUMPY-UFUNCS -->
```python
import numpy as np
import xarray as xr
from tal import ufuncs
from tal.core import AnalysisObject

x = AnalysisObject.from_data(
    xr.Dataset(
        {"value": (("sample", "axis"), [[0.0, 1.0], [2.0, 3.0]])},
        coords={"sample": [0, 1], "axis": ["a", "b"]},
    ),
    sequence_dim="sample",
    core_dims=("axis",),
    validate=True,
)

y = AnalysisObject.from_data(
    xr.Dataset(
        {"value": (("sample", "axis"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"sample": [0, 1], "axis": ["a", "b"]},
    ),
    sequence_dim="sample",
    core_dims=("axis",),
    validate=True,
)

bias = AnalysisObject.from_data(
    xr.Dataset({"value": ("axis", [10.0, 20.0])}, coords={"axis": ["a", "b"]}),
    core_dims=("axis",),
    validate=True,
)

sin_x = ufuncs.sin(x)
exp_x = ufuncs.exp(x)
sum_xy = x + y
aligned_xy = x.a(on="sequence", sequence_join="inner") + y
broadcast_bias = x + bias.b()
condition = ufuncs.greater(x, 1.0)
```

Unary ufuncs return AOs. Ordering comparisons return `Condition` objects that
can feed the events system.

## Alignment Intent

`.a(...)` attaches operand-local alignment intent. It does not eagerly rewrite
data; it gives an operation planner enough policy to decide how shared named
dimensions should line up.

Use it when an elementwise expression should align by sequence labels or by a
declared parameter key rather than relying on a default.

## Broadcast Intent

`.b()` marks an operand as willing to expand under TAL's semantic broadcast
rules. That is useful for payload-like bias terms or lower-rank values that are
meaningful across a more structured AO.

`.b()` is not an escape hatch. Strict linalg and frame-sensitive spatial
operations still enforce their own contracts.

## What Usually Goes Wrong

- TAL does not rely on `np.sin(ao)` dispatch; use `tal.ufuncs.sin(ao)`.
- Multi-variable AOs need unambiguous numeric variable selection.
- Shared non-core dimensions align by labels, not array position.
- Conflicting chained alignment intents fail closed.

## Quick Checks

- Inspect `sum_xy.as_dataset()`.
- Inspect `sin_x.as_dataset().attrs["tal"]`.
- Inspect `condition` directly before feeding it to `ao.events.mask(...)`.

## See Also

- {doc}`linalg`
- {doc}`indexing`
- API: {doc}`../api/ufuncs`
