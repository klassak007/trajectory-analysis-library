(overview)=
# TAL Overview

*TAL* - the **Trajectory Analysis Library** - is an xarray-native Python
library for trajectory-shaped data: robot logs, simulation runs, motion traces,
time-indexed vectors, poses, velocities, accelerations, and other data where
the meaning of each dimension matters.

TAL keeps data in ordinary `xarray.Dataset` objects and adds a small semantic
schema so analysis code can stay labeled, validated, and frame-aware.

```{figure} ../images/AnalysisObject.png
:alt: An AnalysisObject wraps an xarray Dataset with TAL metadata for sequence, batch, core, parameter, validity, components, and frames.
:width: 90%
:align: center

An `AnalysisObject` is an `xarray.Dataset` plus TAL metadata.
```

## What TAL Means By Trajectory

In TAL, a **trajectory** is an ordered sequence of samples along a meaningful
progression axis. That axis is often time, but it can also be sample number,
simulation step, distance along a path, optimization iteration, gait phase, or
another ordered parameter.

The value at each sample may be a scalar, vector, matrix, pose, velocity,
acceleration, or another structured quantity. TAL names the important roles:

| Role | Meaning |
| --- | --- |
| `sequence_dim` | Ordered sample axis, such as time samples or path steps. |
| `batch_dims` | Independent groups such as runs, trials, robots, scenarios, or logs. |
| `core_dims` | Payload axes such as vector components, matrix rows/cols, or quaternion labels. |
| `param_coord` | Coordinate used for parameter-domain selection, interpolation, or resampling. |
| `sequence_size_coord` | Optional valid-length coordinate for ragged, padded trajectories. |

For the full schema contract, see {doc}`core_concepts`.

## Why TAL Exists

A raw array shape can hide the semantics users need for analysis:

```text
(20, 1000, 3)
```

That shape alone does not say whether `20` means robots or trials, whether
`1000` is time or distance, whether `3` is an xyz vector or something else, or
whether two datasets should align by sample index, timestamp, batch label, or
interpolation. TAL attaches those answers to the dataset instead of relying on
comments, variable names, or memory.

The goal is not to replace `xarray`; it is to make `xarray` more convenient and
explicit for trajectory analysis.

## The Core Model

The center of TAL is the `AnalysisObject`, usually shortened to **AO**. An AO
stores a dataset and records trajectory semantics in `ds.attrs["tal"]`.

```{code-block} python
from tal import AnalysisObject
```

The underlying dataset remains labeled scientific data. TAL adds metadata that
its public operations can consume deliberately:

| Concept | Used For |
| --- | --- |
| dimension roles | Distinguishing batch axes, sequence axes, and payload axes. |
| parameter coordinates | Interpolation, resampling, synchronization, and event clocks. |
| validity metadata | Ignoring padded tails in ragged trajectories. |
| component registries | Extracting and recomposing named payload pieces. |
| frame metadata | Tracking parent/child frame IDs for spatial quantities. |
| spatial metadata | Recording representation and expression-frame semantics. |

For construction patterns and `ao.data` vs `ao.unsafe_data`, see
{doc}`creating_trajectory_objects` and {doc}`viewing`.

## A First Example

<!-- example-id: UG-OVERVIEW-BASIC-WORKFLOW -->
```python
import numpy as np
import xarray as xr

from tal import AnalysisObject

time = np.linspace(0.0, 4.0, 5)
values = np.zeros((2, 5, 3))
values[:, :, 0] = time
values[:, :, 1] = time**2
values[:, :, 2] = 1.0

ds = xr.Dataset(
    {"position": (("run", "sample", "axis"), values)},
    coords={
        "run": ["run_a", "run_b"],
        "sample": np.arange(time.size),
        "time": ("sample", time),
        "axis": ["x", "y", "z"],
    },
)

ao = AnalysisObject.from_data(
    ds,
    sequence_dim="sample",
    batch_dims=("run",),
    core_dims=("axis",),
    param_coord="time",
    validate=True,
)

resampled = ao.param.at([0.5, 1.5, 2.5], on="time")
mean_position = ao.mean(dim="sample")
```

This AO knows that `sample` is the ordered trajectory axis, `run` is an
independent batch axis, `axis` is the vector payload axis, and `time` is the
coordinate for parameter-domain operations.

## Everyday Operations

Use xarray-style methods when the question is about labels, integer positions,
or structural changes:

```{code-block} python
first_run = ao.sel(run="run_a")
first_three_samples = ao.isel(sample=slice(0, 3))
renamed = ao.rename({"sample": "step"})
```

Use reducers when a dimension should be summarized:

```{code-block} python
mean_position = ao.mean(dim="sample")
max_position = ao.max(dim="sample")
count = ao.count(dim="sample")
```

Available AO reducers include `mean`, `sum`, `std`, `var`, `median`, `min`,
`max`, `count`, `any`, and `all`. See {doc}`../api/reducers` for validity and
typed-wrapper details.

Use `ao.param` when the query is about time, distance, phase, or another
declared parameter coordinate:

```{code-block} python
nearest = ao.param.sel([0.5, 1.5], on="time")
resampled = ao.param.at([0.5, 1.5, 2.5], on="time")
regular = ao.param.resample_to(np.linspace(0.0, 4.0, 9), on="time")
```

Use normal `.sel()` and `.isel()` for ordinary xarray labels or integer
positions. Use `ao.param.*` for parameter-domain lookup, interpolation, and
resampling. See {doc}`indexing` and {doc}`time`.

## Feature Map

This overview names the main surfaces without duplicating their full guides.

| Task | Surface | Details |
| --- | --- | --- |
| Create and inspect AOs | `AnalysisObject.from_data(...)`, `ao.data`, `ao.unsafe_data` | {doc}`creating_trajectory_objects`, {doc}`viewing`, {doc}`../api/analysis-object` |
| Select, mask, and reduce | `ao.sel(...)`, `ao.isel(...)`, `ao.where(...)`, AO reducers | {doc}`indexing`, {doc}`../api/reducers` |
| Interpolate or resample by a parameter | `ao.param.index(...)`, `ao.param.sel(...)`, `ao.param.at(...)`, `ao.param.resample_to(...)`, `ao.param.interp_like(...)` | {doc}`time`, {doc}`../api/timebase` |
| Combine or align AOs | `ao.combine.concat_batch(...)`, `ao.combine.concat_sequence(...)`, `ao.combine.merge(...)`, `ao.combine.align(...)` | {doc}`numpy`, {doc}`../api/analysis-object` |
| Use AO-aware arithmetic | `tal.ufuncs`, AO operators, `.a(...)`, `.b(...)` | {doc}`numpy`, {doc}`../api/ufuncs` |
| Do labeled linear algebra | `tal.linalg.Array`, `Vector`, `Vector3`, `Matrix` | {doc}`linalg`, {doc}`../api/types/index` |
| Work with poses and kinematics | `tal.spatial.Position`, `Rotation`, `Pose`, velocity and acceleration types | {doc}`spatial`, {doc}`../api/types/index` |
| Attach or resolve frames | `ao.frames`, `tal.frames.FrameGraph`, `find_path(...)` | {doc}`frames`, {doc}`../api/frames` |
| Extract events and windows | `ao.events.mask(...)`, `events(...)`, `intervals(...)`, `when(...)`, `around(...)` | {doc}`events`, {doc}`../api/events` |
| Group runs or bins | `ao.group.groupby(...)`, `ao.group.groupby_bins(...)`, grouped reducers | {doc}`../api/analysis-object` |
| Manage named components | `ao.components.define(...)`, `registry(...)`, `extract(...)`, `patch(...)`, `compose(...)` | {doc}`../api/components` |
| Persist and ingest data | `ao.io.to_zarr(...)`, `AnalysisObject.from_zarr(...)`, CSV and log readers | {doc}`../api/io` |
| Visualize trajectories | `ao.viz.line(...)`, `ao.viz.scatter(...)`, `ao.viz.explorer(...)` | {doc}`viewing`, {doc}`../api/viz` |

## Dimension Roles In Practice

A dataset shaped like:

```text
run x sample x axis
```

can mean:

```text
batch x sequence x xyz-vector
```

That distinction changes how operations should behave. A reducer over `sample`
summarizes each trajectory over its ordered progression. A reducer over `run`
summarizes independent trials. A linear-algebra operation over `axis` treats
that dimension as payload structure.

TAL's role metadata lets operations make those choices deliberately instead of
guessing from dimension order.

## Spatial And Frame Semantics

TAL's spatial layer builds on the same AO schema but adds typed meaning for
robotics and physics data: positions, rotations, poses, velocities, and
accelerations.

```{code-block} python
from tal.spatial import Pose, Position, Rotation
```

Spatial types distinguish two concepts that are easy to conflate:

| Operation | Meaning |
| --- | --- |
| `to_frame(...)` | Change the parent/child frame relationship of a quantity. |
| `express_in(...)` | Change the coordinate basis used to express the quantity. |

Moving a pose from `camera` to `world` is not the same as merely expressing a
vector in a different basis. See {doc}`spatial` and {doc}`frames`.

## Recommended Learning Path

1. Start with {doc}`creating_trajectory_objects`.
2. Learn the role model in {doc}`core_concepts`.
3. Practice selection and reducers with {doc}`indexing`.
4. Add parameter-domain interpolation with {doc}`time`.
5. Use typed wrappers in {doc}`linalg` and {doc}`spatial`.
6. Add frame metadata and graph topology with {doc}`frames`.
7. Use conditions, events, and windows in {doc}`events`.
8. Keep {doc}`viewing` nearby when inspecting schema or debugging topology.

## How To Think In TAL

Before writing analysis code, answer these questions:

1. What is the `sequence_dim`?
2. What are the `batch_dims`?
3. What are the `core_dims`?
4. Is there a `param_coord`, such as time?
5. Are trajectories all the same length, or do they need validity metadata?
6. Does this data live in a coordinate frame?
7. Is the payload ordinary numeric data, linear algebra data, or spatial data?
8. Should two objects align by labels, sample index, parameter coordinate, or interpolation?

Good TAL code is explicit about those answers.

## Glossary

| Term | Meaning |
| --- | --- |
| trajectory | Ordered samples along time, sample number, distance, iteration, phase, or another progression variable. |
| AO | Short for `AnalysisObject`. |
| sequence | Ordered samples along a trajectory. |
| batch | Independent groups such as runs, trials, robots, scenarios, or logs. |
| core | Payload axes such as vector, matrix, quaternion, or component dimensions. |
| parameter coordinate | Coordinate used for selection, interpolation, resampling, or synchronization. |
| ragged sequence | A set of sequences with unequal valid lengths. |
| frame | Coordinate frame such as `world`, `map`, `odom`, `base_link`, or `camera`. |
| representation | Storage form such as quaternion, rotation matrix, component pose, or matrix pose. |

TAL is useful when data is more than an anonymous array: dimensions have roles,
trajectories need alignment, samples live on a parameter grid, vectors and
matrices have payload semantics, and spatial quantities belong to coordinate
frames.
