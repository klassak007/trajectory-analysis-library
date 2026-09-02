(core-concepts)=
# Core Concepts

TAL's core idea is simple: keep the data in `xarray`, and store the meaning of
the dimensions in `ds.attrs["tal"]`. That metadata is small, but it lets public
operations distinguish ordered samples from independent batches and payload
axes without guessing from array shape.

## The Role Model

Most TAL objects are organized around three dimension roles:

| Role | Meaning | Examples |
| --- | --- | --- |
| `sequence_dim` | Ordered samples along a trajectory. | `sample`, `step`, `frame` |
| `batch_dims` | Independent lanes of analysis. | `trial`, `run`, `robot`, `scenario` |
| `core_dims` | Payload structure consumed as a value. | `axis`, `row`, `col`, `quat` |

`core_dims` is always explicit when roles are declared. It may be empty for a
scalar payload, which is still useful because TAL then knows that non-core dims
are topology rather than value structure.

## Parameter And Validity Semantics

The sequence dimension is the structural axis. A `param_coord` is the semantic
coordinate used to ask questions such as "what was the value at time 2.5?" or
"resample this trajectory onto this distance grid." The parameter is often time,
but TAL treats it as a general ordered coordinate.

Ragged batches use `sequence_size_coord` to record the valid prefix length for
each batch lane. This lets rectangular arrays carry variable-length trajectories
without letting padded tails participate in reducers, events, or interpolation.

## Components And Extensions

Some payload axes contain named pieces. A pose may contain rotation and
position; a six-vector may contain linear and angular components; a marker array
may contain named marker groups. TAL records those mappings through the
component registry behind `ao.components`.

Other extension metadata, such as frame IDs and spatial representation, lives
under the same schema root. The important rule is that operations must keep
metadata truthful after structural changes.

## Minimal Example

<!-- example-id: UG-CORE-CONCEPTS-ROLES -->
```python
import numpy as np
import xarray as xr
from tal.core import AnalysisObject
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name

sample = np.arange(3)
trial = ["a", "b"]
time_s = np.array([0.0, 0.1, 0.2])

ao = AnalysisObject.from_data(
    xr.Dataset(
        {"speed": (("trial", "sample"), [[0.0, 1.0, 2.0], [0.5, 1.5, 0.0]])},
        coords={
            "trial": trial,
            "sample": sample,
            "time_s": (("trial", "sample"), np.broadcast_to(time_s, (2, 3))),
            "group_size": ("trial", np.array([3, 2], dtype=np.int64)),
        },
    ),
    sequence_dim="sample",
    batch_dims=("trial",),
    core_dims=(),
    param_coord="time_s",
    sequence_size_coord="group_size",
    validate=True,
)

snapshot = ao.as_dataset()
roles = read_roles(snapshot)
param_name = read_param_coord_name(snapshot)
size_name = read_sequence_size_coord_name(snapshot)
```

This object has two trials, three structural sample slots, a per-trial
parameter coordinate, and valid-length metadata that marks the second trial as
having only two valid samples.

## How Operations Use The Schema

Structural methods such as `ao.rename(...)`, `ao.drop_vars(...)`, `ao.isel(...)`,
`ao.sel(...)`, and `ao.where(...)` repair or prune schema metadata after the
dataset topology changes. Reducers remove roles for dimensions they reduce.
Parameter operations use `param_coord`, and grouped or ragged operations respect
`sequence_size_coord`.

The same role metadata is also consumed by `tal.ufuncs`, `tal.linalg`, and
`tal.spatial`, so numeric and typed operations can preserve sequence and batch
topology instead of collapsing everything into anonymous arrays.

`tal.ufuncs` provides AO-aware NumPy-style arithmetic while keeping role
metadata attached to the result.

## What Usually Goes Wrong

- A role references a dimension that no longer exists.
- `param_coord` or `sequence_size_coord` is declared without a `sequence_dim`.
- A dimension is accidentally declared in more than one role.
- A component registry mentions labels or variables that were renamed or
  dropped.

When one of these happens, TAL fails closed. The fix is usually to inspect the
schema and make the role declaration match the dataset you actually have.

## Quick Checks

- Capture one `snapshot = ao.as_dataset()`, then inspect `snapshot.sizes` and
  `snapshot.coords`.
- Read `roles`, `param_name`, and `size_name` after construction.
- Inspect `snapshot.attrs["tal"]` when an operation does not seem to
  honor the dimensions you expected.

## See Also

- {doc}`creating_trajectory_objects`
- {doc}`indexing`
- {doc}`viewing`
- API: {doc}`../api/schema`
