(spatial-types)=
# Spatial

TAL's spatial layer gives trajectory data robotics-style meaning. Positions,
rotations, poses, velocities, and accelerations are still AOs underneath, but
they carry representation, relation, frame, and temporal semantics that plain
arrays cannot express safely.

Use spatial types when the payload is not just "three numbers," but a vector,
rotation, transform, or kinematic quantity with a coordinate-frame contract.

## Minimal Example

<!-- example-id: UG-SPATIAL-POSE -->
```python
import numpy as np
import xarray as xr
from tal.core import AnalysisObject
from tal.frames import FrameGraph
from tal.spatial import Pose, Position, Rotation

sample = np.arange(3)
time_s = np.array([0.0, 0.5, 1.0])

pos = Position(AnalysisObject.from_data(
    xr.Dataset(
        {"position": (("sample", "axis"), [[1.0, 0.0, 0.0], [1.5, 0.5, 0.0], [2.0, 1.0, 0.0]])},
        coords={"sample": sample, "axis": ["x", "y", "z"], "time_s": ("sample", time_s)},
    ),
    sequence_dim="sample",
    core_dims=("axis",),
    param_coord="time_s",
    validate=True,
))

rot = Rotation(AnalysisObject.from_data(
    xr.Dataset(
        {
            "rotation": (
                ("sample", "quat"),
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.3826834, 0.9238795], [0.0, 0.0, 0.7071068, 0.7071068]],
            )
        },
        coords={"sample": sample, "quat": ["x", "y", "z", "w"], "time_s": ("sample", time_s)},
    ),
    sequence_dim="sample",
    core_dims=("quat",),
    param_coord="time_s",
    validate=True,
))

graph = FrameGraph()
pose = Pose.from_components(
    rot,
    pos,
    parent="world",
    child="body",
    graph=graph,
)
pose.register()
body_samples = Position(pos, parent="body", child="probe", graph=graph)
world_samples = body_samples.to_frame("world")
detached = pose.with_graph(None)
out_pos, out_rot = pose.decompose()
identity_like = pose.compose(pose.inverse())
rotated = rot.apply(pos)
transformed = pose.apply(pos)
rot_m = rot.as_matrix()
rot_q = rot_m.as_quat()
pose_m = pose.as_matrix()
rot_at = rot.param.at([0.25], on="time_s")
pose_rs = pose.param.resample_to(np.linspace(0.0, 1.0, 5), on="time_s")
```

This example builds position and rotation trajectories, composes them into a
pose, applies transforms, converts representations, and evaluates typed
rotations/poses on a parameter grid. Registration retains the Pose's native
sampling; `to_frame` evaluates that provider on `body_samples`' parameter grid
without repeating `graph=`.

Frame declarations are schema metadata. The optional `graph` is a passive,
wrapper-local association: construction and `with_graph(...)` never create
frames or register providers. Spatial results retain a shared association;
Dataset/DataArray conversion and deliberate demotion do not. Use
`pose.register()` for an associated static or native-rate Pose, or
`bind_pose(...)` for a callable or exact unparameterized provider.

## Spatial Types

| Type | Meaning |
| --- | --- |
| `Position` | Cartesian `x`, `y`, `z` position or displacement. |
| `Rotation` | Quaternion or matrix rotation. |
| `Pose` | Rigid transform represented as components or a matrix. |
| `LinearVelocity`, `AngularVelocity`, `Velocity` | Kinematic velocity payloads. |
| `LinearAcceleration`, `AngularAcceleration`, `Acceleration` | Kinematic acceleration payloads. |

Combined kinematic payloads can use components or vector6 representation.

Spatial constructors validate representation labels. For example, position axes
must be `x`, `y`, `z`; quaternions must be `x`, `y`, `z`, `w`; matrix layouts
must use the expected row and column labels.

Direct spatial constructors and the approved Pose, Velocity, and Acceleration
factories accept `parent=`, `child=`, `expressed_in=`, and `graph=`. Omission
inherits existing declarations, while explicit `None` clears the corresponding
declaration or association. `Rotation.from_data(...)` remains a schema-ingress
factory without these keywords; associate its result with
`rotation.with_graph(graph)`. `obj.graph` is read-only, and
`obj.with_graph(graph)` returns a distinct metadata-isolated alias.

## Transform Algebra

Use `compose(...)`, `inverse()`, and `apply(...)` for local transform algebra.
Use `Pose.from_components(...)`, `pose.decompose()`, `pose.as_matrix()`,
`pose.as_components()`, and `Pose.from_matrix(...)` to move between pose
layouts. Inverse remains graph-free: a framed Pose or Rotation must already be
expressed in its parent basis. Re-express a third-frame value explicitly with
`value.express_in(parent).inverse()`. A missing parent denotes an unframed value
only when the child and expression basis are absent too; otherwise complete or
clear the partial framing before inversion.

Rotations can be converted with `Rotation.as_matrix()`, `Rotation.as_quat()`,
and `Rotation.to_rep(...)`.

## Parameter-Aware Spatial Operations

Spatial objects inherit `param` accessors, but typed objects can choose geometry
appropriate interpolation. `Rotation.param.at(...)` uses rotation-aware
interpolation by default; `Pose.param.at(...)` uses linear position
interpolation and rotation-aware interpolation for the rotational part.

Kinematic temporal methods are available on the relevant typed wrappers:

- `position.differentiate(...) -> LinearVelocity`
- `linear_velocity.differentiate(...) -> LinearAcceleration`
- `linear_velocity.integrate(...) -> Position`
- `velocity.differentiate(...) -> Acceleration`
- `acceleration.integrate(...) -> Velocity`
- `*.smooth(...)` for supported position, velocity, and acceleration families

## Frames: Relation vs Expression

Spatial frame semantics distinguish two operations:

| Operation | Meaning |
| --- | --- |
| `to_frame(...)` | Change the parent/child frame relationship of the quantity. |
| `express_in(...)` | Change only the coordinate basis used to write the components. |

That distinction is essential. Retargeting a point from `camera` to `world` is
not the same as expressing the same vector in a different basis.

Frame-aware path solve operations build on `tal.frames.FrameGraph` and explicit
edge resolver callbacks. TAL fails closed when graph endpoints, edge payloads,
frame tags, or parameter alignment are ambiguous.

Edge metadata belongs to spatial path support, not to the frame graph itself.
Motion and inertial annotations can guide kinematic transforms when path
solving needs more than topology.

## What Usually Goes Wrong

- Core labels do not match the declared spatial representation.
- Compose and apply operations receive incompatible frame tags.
- Pose inputs have incompatible sequence or batch topology.
- Path solving lacks graph endpoints, edge resolvers, or required support
  metadata.
- Parameter interpolation is requested without a declared or explicit
  parameter coordinate.

## Quick Checks

- Capture one `pose_snapshot = pose.as_dataset()`, then inspect it and
  `pose_snapshot.attrs["tal"]`.
- Compare `rot_q.as_dataset()` with `rot_m.as_dataset()` when checking
  representation changes.
- Confirm the declared `param_coord` before using `.param.at(...)` or
  `.param.resample_to(...)`.
- Check `ao.frames.ids()` before frame-aware operations.

## See Also

- {doc}`frames`
- {doc}`time`
- API: {doc}`../api/types/index`
