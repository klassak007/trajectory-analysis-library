(illustrated-example)=
# Working with the Illustrated Trajectories

This example builds the object in the {ref}`AnalysisObject diagram <analysis-object-diagram>`.
Trials A, B, and C have six storage slots each, valid lengths of six, four, and
five, and different timestamps. Each valid sample holds a position and a
quaternion rotation.

```{figure} ../images/AnalysisObject.svg
:alt: Three Pose trajectories with separate position and rotation variables, shaded padding, and a table of per-trial timestamps.
:width: 100%
:align: center

The same sample index can refer to different times in different trials.
```

## Build the object and try the operations

The diagram specifies the layout, not numerical poses. To make the operations
easy to check, use the same simple motion in each trial: position `(t, 2t, 0)`
and identity rotation `(0, 0, 0, 1)`, sampled at the times shown. Times are in
seconds; the position values are in metres. These are example conventions,
not unit inference by TAL. No frame graph is needed for this example.

<!-- example-id: UG-ILLUSTRATED-WORKFLOW -->
```python
import numpy as np
import xarray as xr

from tal.core import AnalysisLayoutSpec
from tal.spatial import Pose

time = np.array([
    [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
    [0.02, 0.17, 0.31, 0.48, np.nan, np.nan],
    [0.00, 0.08, 0.23, 0.39, 0.57, np.nan],
])
lengths = np.array([6, 4, 5])
valid = np.arange(6) < lengths[:, None]
zero = np.where(valid, 0.0, np.nan)
one = np.where(valid, 1.0, np.nan)
source = xr.Dataset(
    {
        "position.x": (("trial", "sample"), time),
        "position.y": (("trial", "sample"), 2 * time),
        "position.z": (("trial", "sample"), zero),
        "rotation.x": (("trial", "sample"), zero),
        "rotation.y": (("trial", "sample"), zero),
        "rotation.z": (("trial", "sample"), zero),
        "rotation.w": (("trial", "sample"), one),
    },
    coords={
        "trial": ["A", "B", "C"],
        "sample": np.arange(6),
        "time": (("trial", "sample"), time),
        "sequence_size": ("trial", lengths),
    },
)
layout = AnalysisLayoutSpec(
    sequence_dim="sample", batch_dims=("trial",),
    param_coord="time", sequence_size_coord="sequence_size",
)
poses = Pose.from_fields(
    layout.wrap(source),
    position="position.{x,y,z}", rotation="rotation.{x,y,z,w}",
)

trial_b = poses.sel(trial="B")
first_two = poses.isel(sample=slice(0, 2))
positions, rotations = poses.decompose()
slot_two = positions.isel(sample=2)
at_time = poses.param.at([0.20])
mean_position = positions.mean(dim="sample")
```

`Pose.from_fields` assembles the two component variables and their core labels:
`position(trial, sample, axis)` and `rotation(trial, sample, quat)`. The source
layout supplies batch, sequence, parameter, and validity declarations.

## Read the results against the diagram

| Result | What changes in the diagram |
| --- | --- |
| `trial_b` | Select the green layer. Its two padded slots remain padding. |
| `first_two` | Keep slots 0 and 1 in every trial. All retained samples are valid. |
| `positions`, `rotations` | Separate the three-component and four-component slabs into typed objects. |
| `slot_two` | Select slot 2, whose times are 0.20, 0.31, and 0.23 seconds. |
| `at_time` | Evaluate all three trials at 0.20 seconds using each trial's own clock. |
| `mean_position` | Reduce the valid samples in each position slab to one position per trial. |

The selected slot therefore has x positions **0.20, 0.31, and 0.23**. The common
time query gives position **(0.20, 0.40, 0)** and identity rotation in all three
trials, using interpolation where needed. These operations return new objects;
the source retains its original layout and values.

The mean x positions are **0.250, 0.245, and 0.254**. This is a mean over recorded
samples, so it reflects each trial's sampling pattern. It is not a time-weighted
average. Padding contributes no observations.

## Continue with

- {doc}`creating_trajectory_objects` for layouts and reusable field recipes.
- {doc}`indexing` for positional, label, and parameter selection.
- {doc}`time` for resampling and synchronization.
- {doc}`spatial` for spatial operations and frame semantics.
