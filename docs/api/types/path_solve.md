(api-type-path-solve)=
# Path Solve

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

Path solving composes frame-edge payloads along a `tal.frames` topology path.
The frame graph finds the path; caller-provided edge resolver callbacks provide
the rotation or pose payloads for each edge.

## Functional API

```python
from tal.spatial import PathSolveOptions, solve_pose_path_transform, solve_rotation_path_transform
```

```python
rotation = solve_rotation_path_transform(src, dst, *, edge_rotation_fn, opts=None)
pose = solve_pose_path_transform(src, dst, *, edge_pose_fn, opts=None)
```

Path solving does not mutate the graph. It validates endpoint resolution, edge
payload compatibility, frame tags, and composition order.

## Endpoint Policy

Endpoints may be `Frame` objects or frame ID strings. Graph resolution uses:

1. `opts.graph` when provided,
2. the endpoint's bound graph when endpoint objects are supplied,
3. the active graph for all-string endpoints.

Cross-graph, disconnected, or unregistered endpoints fail closed.

## Options Type

Omitting `opts` or passing `None` selects default `PathSolveOptions`. Otherwise,
pass a `PathSolveOptions` instance; subclasses retain their supplied policy.
Mappings and other objects are not converted, and falsey values such as `{}`,
`0`, and `False` are rejected rather than treated as defaults.

This shared type boundary raises an operation-prefixed `TypeError` before
graph lookup or resolver inspection/invocation. It also applies to ergonomic
`to_frame` and `express_in` calls, including same-frame requests and
`validate=False`. Valid same-frame shortcuts retain their existing behavior;
individual option fields are checked only by the operations that consume them.

## Minimal Example

```python
from tal.frames import FrameGraph
from tal.spatial import PathSolveOptions, solve_rotation_path_transform

graph = FrameGraph()
with graph:
    world = graph.get_or_create_frame("world")
    body = graph.get_or_create_frame("body", parent=world)
    sensor = graph.get_or_create_frame("sensor", parent=body)

    def edge_rotation(child, parent):
        return edge_map[(child.id, parent.id)]

    r_sensor_to_world = solve_rotation_path_transform(
        "sensor",
        "world",
        edge_rotation_fn=edge_rotation,
        opts=PathSolveOptions(graph=graph),
    )
```

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

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/path_solve
   :nosignatures:

   tal.spatial.PathSolveOptions
   tal.spatial.solve_rotation_path_transform
   tal.spatial.solve_pose_path_transform
```

## See Also

- {doc}`rotation`
- {doc}`pose`
- {doc}`../frames`
