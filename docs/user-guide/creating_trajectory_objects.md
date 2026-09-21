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

## Build spatial objects from named scalar fields

Start from a schema-bearing AO when possible. The one-shot form is the shortest
workflow, while a recipe is useful when several sources expose the same field
suffixes.

<!-- example-id: UG-CREATING-SPATIAL-FIELDS -->
```python
import xarray as xr
from tal.core import AnalysisLayoutSpec
from tal.spatial import Pose

source = xr.Dataset(
    {
        "camera.position.x": ("sample", [1.0]),
        "camera.position.y": ("sample", [2.0]),
        "camera.position.z": ("sample", [3.0]),
        "camera.rotation.x": ("sample", [0.0]),
        "camera.rotation.y": ("sample", [0.0]),
        "camera.rotation.z": ("sample", [0.0]),
        "camera.rotation.w": ("sample", [1.0]),
    },
    coords={"sample": [0]},
)
layout = AnalysisLayoutSpec(sequence_dim="sample")
declared = layout.wrap(source)

# Shortest AO one-shot workflow.
camera = Pose.from_fields(
    declared,
    position="camera.position.{x,y,z}",
    rotation="camera.rotation.{x,y,z,w}",
)

# Reuse one immutable selector recipe with a literal source-name prefix.
pose_fields = Pose.fields(
    position="position.{x,y,z}",
    rotation="rotation.{x,y,z,w}",
)
camera_again = pose_fields.build(declared, prefix="camera.")

# An untagged Dataset supplies exactly one explicit layout authority.
camera_from_raw = pose_fields.build(
    source,
    prefix="camera.",
    source_layout=layout,
)
```

Brace selectors expand exact field names; dots and `prefix=` are literal. A
mapping from target labels to exact source names is the alternative when names
do not share a compact pattern. TAL does not infer a layout, representation,
quaternion order, or frame from field spelling.

Field assembly preserves applicable Dataset and coordinate metadata, parameter
and validity declarations, and native indexes. The newly assembled `position`
and `rotation` variables intentionally have empty ordinary attributes and
storage encodings, so per-channel units or storage settings are never chosen
implicitly.

## Build from CSV and ROS reader results

CSV and ROS readers already return schema-bearing AOs, so their declared batch,
sequence, parameter, and validity layout is inherited directly. Do not pass
`source_layout` for a reader result.

<!-- example-id: UG-CREATING-SPATIAL-FIELDS-FROM-READERS -->
```python
from tal.io import CsvIngestOptions, RosIngestOptions, read_csv_logs, read_ros_logs
from tal.spatial import Pose

csv_source = read_csv_logs(
    "camera.csv",
    opts=CsvIngestOptions(time_col="time"),
)
csv_pose = Pose.from_fields(
    csv_source,
    position="camera.position.{x,y,z}",
    rotation="camera.rotation.{x,y,z,w}",
)

ros_source = read_ros_logs(
    "camera.mcap",
    opts=RosIngestOptions(topic="/camera/pose"),
)
ros_pose = Pose.from_fields(
    ros_source,
    position="translation_{x,y,z}",
    rotation="quaternion_{x,y,z,w}",
)
```

Field factories consume the returned AO, not a path, bag, or DataFrame. They do
not reopen the reader, widen a CSV `value_columns` selection, push projection
into ingestion, or infer frames from ROS field names. Install the `ros` extra to
read ROS recordings; ordinary Dataset/AO field construction does not require
that optional dependency.

## Reuse a complete layout and select variables

<!-- example-id: UG-CREATING-REUSABLE-LAYOUT -->
```python
import xarray as xr
from tal.core import AnalysisLayoutSpec
from tal.spatial import Position

layout = AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",))
dataset = xr.Dataset(
    {
        "position": (("sample", "axis"), [[1.0, 2.0, 3.0]]),
        "quality": ("sample", [1]),
    },
    coords={"sample": [0], "axis": ["x", "y", "z"]},
)
selected = layout.wrap(dataset, data_vars="position")
position = Position(selected)
ordered = layout.wrap(dataset).select_vars(("quality", "position"))
assert list(ordered.as_dataset().data_vars) == ["quality", "position"]
assert list(position.as_dataset().data_vars) == ["position"]
```

The spec is a complete declaration: default fields remove source role and
optional-coordinate metadata. `AnalysisObject.from_data(...)` instead treats
its defaults as an overlay. Selection on an existing AO retains its subtype
only when the selected schema still satisfies that subtype; it does not
silently downgrade. Neither path infers roles or registers frame providers.

## Dataset exposure

- `ao.as_dataset()` returns a mutation-safe deep snapshot.
- `ao.as_dataset(copy="shallow")` isolates xarray structure and metadata while
  sharing payload buffers or lazy graphs.
- `ao.as_dataset(copy="none")` returns the backing Dataset and is reserved for
  explicit expert ownership crossings.
- Keep the AO open while using lazy deep/shallow views, and call `ao.close()`
  when a lazily loaded AO is no longer needed.

## What Usually Goes Wrong

- `param_coord` without a declared `sequence_dim`.
- `sequence_size_coord` without a declared `sequence_dim`.
- A role name that is a coordinate but not a dimension.
- Multi-variable data passed to an operation that needs one unambiguous numeric
  payload.

Construction-time mistakes are cheaper to fix than downstream alignment or
interpolation errors, so it is worth declaring semantics early.

## Quick Checks

- Capture one `snapshot = full.as_dataset()`, inspect it, then check
  `snapshot.attrs["tal"]`.
- Use `full.to_dataarray(name="position")` only when one data variable should
  become the payload boundary.

## See Also

- {doc}`core_concepts`
- {doc}`time`
- {doc}`events`
- API: {doc}`../api/analysis-object`
