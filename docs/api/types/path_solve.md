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
rotation = solve_rotation_path_transform(src, dst, *, edge_rotation_fn=None, graph=None, query=None, opts=None)
pose = solve_pose_path_transform(src, dst, *, edge_pose_fn=None, graph=None, query=None, opts=None)
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

## Native-rate providers and direct queries

Providers are classified from declared TAL roles: static providers have no
sequence role, dynamic providers have a sequence role and parameter coordinate,
and exact providers have a sequence role without a parameter coordinate. A
dynamic direct solve requires `query=`. Object operations derive the grid from
the caller, so a registered native-rate provider can be used directly:

```python
native_edge = Pose.from_components(rotation, position, parent="world", child="body", graph=graph)
native_edge.register()
body_samples = Position(samples, parent="body", child="probe", graph=graph)
world_samples = body_samples.to_frame("world")
```

For a direct solve, the configured query dimension names the result sequence
axis. For a scalar or one-dimensional query it is also the parameter
coordinate:

```python
from tal.spatial.temporal import PoseTemporalOptions

result = solve_pose_path_transform(
    "body",
    "world",
    graph=graph,
    query=np.linspace(0.0, 1.0, 101),
    opts=PathSolveOptions(temporal=PoseTemporalOptions()),
)
```

A labeled multidimensional `xarray.DataArray` query owns the output topology:
every leading dimension is a batch dimension and the final dimension becomes
the configured sequence axis. Query-only batch dimensions broadcast providers.
Shared dimensions must have exactly equal public xarray index topology.
Provider-only batch dimensions are removed only when they have size one; a
non-singleton provider-only dimension is rejected rather than creating an
implicit Cartesian expansion. TAL stores batch-varying query values in a
collision-safe auxiliary parameter coordinate (for example `query_value`).
Scalar and one-dimensional queries remain unbatched.

```python
batched_query = xr.DataArray(
    [[0.0, 0.5], [0.5, 1.0]],
    dims=("trial", "when"),
    coords={"trial": ["a", "b"], "when": [0, 1]},
)
batched = solve_pose_path_transform(
    "native_body",
    "world",
    graph=graph,
    query=batched_query,
)
assert batched.as_dataset(copy="none").attrs["tal"]["core"]["roles"]["batch_dims"] == ["trial"]
```

All dynamic providers must cover every structurally valid query in their closed
coordinate domains. Interior gaps are interpolated; extrapolation is rejected,
and TAL does not impose a maximum-gap policy. `PoseTemporalOptions.on` selects
each provider's source coordinate without renaming the caller/direct grid.
Numeric and datetime domains cannot be mixed; clock synchronization and unit
conversion remain caller responsibilities. An exact provider cannot be used
with `query=` or mixed with a dynamic provider.

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
