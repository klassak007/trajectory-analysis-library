(api-type-path-solve)=
# Path Solve

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

Path solving composes frame-edge payloads along a `tal.frames` topology path.
The frame graph finds the path. Bind Pose values or callbacks once with
`bind_pose`, then pass `graph=` to solve a pose or rotation path. An explicit
`edge_pose_fn` or `edge_rotation_fn` overrides bound providers for that call.

## Functional API

```python
from tal.spatial import PathSolveOptions, solve_pose_path_transform, solve_rotation_path_transform
```

```python
rotation = solve_rotation_path_transform(src, dst, *, edge_rotation_fn=None, graph=None, opts=None)
pose = solve_pose_path_transform(src, dst, *, edge_pose_fn=None, graph=None, opts=None)
```

Path solving does not mutate the graph. It validates endpoint resolution, edge
payload compatibility, frame tags, and composition order.

## Endpoint Policy

Endpoints may be `Frame` objects or frame ID strings. Graph resolution uses:

1. `graph` or `opts.graph` when provided,
2. remembered graphs on participating spatial objects,
3. the endpoint's bound graph when endpoint objects are supplied,
4. the active graph when no explicit or remembered source exists.

Cross-graph, disconnected, or unregistered endpoints fail closed.

`graph=` may accompany `opts` when `opts.graph is None`. Two non-`None`
graph sources are rejected, even if they refer to the same graph, before any
identity shortcut. Options instances and subclass policies remain unchanged.

## Options Type

Omitting `opts` or passing `None` selects default `PathSolveOptions`. Otherwise,
pass a `PathSolveOptions` instance; subclasses retain their supplied policy.
Mappings and other objects are not converted, and falsey values such as `{}`,
`0`, and `False` are rejected rather than treated as defaults.

This shared type boundary raises an operation-prefixed `TypeError` before
graph lookup or resolver inspection/invocation. It also applies to ergonomic
`to_frame` and `express_in` calls, including same-frame requests and
`validate=False`. Object-centered same-frame shortcuts validate outer option
and graph consistency, then skip unused strict policy, resolver, provider, and
payload work. Direct functional/class solves remain graph-required even for an
identity path.

## Minimal Example

```python
import numpy as np
import xarray as xr
from tal import AnalysisObject
from tal.frames import FrameGraph
from tal.spatial import Pose, Position, Rotation, bind_pose, solve_pose_path_transform

rotation = Rotation(AnalysisObject.from_data(
    xr.DataArray([0., 0., 0., 1.], dims="q",
                 coords={"q": ["x", "y", "z", "w"]}, name="r"),
    core_dims=("q",)))
position = Position(AnalysisObject.from_data(
    xr.DataArray([1., 0., 0.], dims="axis",
                 coords={"axis": ["x", "y", "z"]}, name="p"),
    core_dims=("axis",)))
edge = Pose.from_components(rotation, position)
graph = FrameGraph()
bind_pose(graph, "world", "body", edge)
bind_pose(graph, "body", "sensor", edge)
result = solve_pose_path_transform("sensor", "world", graph=graph)
translation, orientation = result.decompose()
np.testing.assert_allclose(translation.to_dataarray(), [2., 0., 0.])
assert result.frames.ids() == ("world", "sensor")
```

Static values use existing Pose construction semantics. Binding isolates nested
Dataset, variable, and coordinate attrs and encodings while sharing owned
numerical buffers and retaining lazy payloads. Later path results do not alias
the stored provider metadata. The graph owns the edge relation. Framed inputs
must match the edge, and an explicit `expressed_in` must equal the edge parent;
re-express a third-frame value in the parent before binding or returning it
from a resolver. Stored static values therefore survive graph renames.
Callbacks receive the current `(child, parent)` frame objects and run only
during resolution. Returned frame tags are checked against those current IDs.
Inspectable incompatible signatures fail at binding. For an uninspectable
callable, a boundary-only ``TypeError`` is conservatively treated as invocation
misuse; failures observed after entering a Python callback are callback failures.
Public errors and their direct causes expose only public exception types, and
callback failures name the failing edge.

An existing edge without a provider can be bound. Replacing a provider requires
`on_conflict="replace"` and preserves unrelated edge metadata. It cannot reparent
the child or mutate a frozen graph. Detaching, reparenting, or removing an edge
clears its provider; subtree removal clears every removed descendant. Supported
validation and conflict failures leave the graph unchanged.

Lazy static values and callback results are non-owning aliases. Keep the AO
that owns their backend resource open until dependent path results have been
computed. Replacing or removing a provider does not close that resource.

Binding a static value does not imply static physical motion, inertial status,
or derivative support. Kinematics `to_frame` still requires its documented
motion support; `express_in` keeps its established basis-only semantics.

## Class-Friendly Entry Points

Spatial types expose ergonomic wrappers where appropriate:

- `Position.to_frame(...)` and `Position.express_in(...)`
- `Rotation.solve_path_transform(...)` and `Rotation.express_in(...)`
- `Pose.solve_path_transform(...)` and `Pose.express_in(...)`
- `LinearVelocity`, `AngularVelocity`, and `Velocity` `to_frame(...)` and
  `express_in(...)` methods
- `LinearAcceleration`, `AngularAcceleration`, and `Acceleration`
  `to_frame(...)` and `express_in(...)` methods

These wrappers preserve the same topology policy as the functional path solvers.
When no explicit graph is supplied, object methods prefer the source object's
remembered association. Successful path results remember the graph actually
selected.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/path_solve
   :nosignatures:

   tal.spatial.bind_pose
   tal.spatial.PathSolveOptions
   tal.spatial.solve_rotation_path_transform
   tal.spatial.solve_pose_path_transform
```

## See Also

- {doc}`rotation`
- {doc}`pose`
- {doc}`../frames`
