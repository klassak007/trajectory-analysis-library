(api-pose)=
# `Pose`

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

`tal.spatial.Pose` is an `AnalysisObject` subtype for rigid transforms. A pose
can be represented as rotation/position components or as a matrix, and exposes
composition, inverse, application, representation conversion, and typed
parameter evaluation.

```{contents}
:local:
:depth: 2
```

## Construction

```python
Pose.from_components(rotation, position, ...)
Pose.from_matrix(matrix, ...)
pose.decompose()
```

Component layout stores rotation and position payloads. Matrix layout stores a
matrix-style pose payload. Decomposition returns typed `Position` and
`Rotation` outputs.

Constructors accept `parent=`, `child=`, `expressed_in=`, and `graph=`.
Association is passive and does not register the pose. Inspect `pose.graph` or
return a distinct associated alias with `pose.with_graph(...)`.

Use `pose.register()` after associating a canonical parent/child Pose with a
graph. It accepts static and parameterized native-rate providers, returns the
same Pose by identity, and never implicitly reparents an existing frame.
`tal.spatial.bind_pose(...)` remains the advanced graph-first API for callable
or exact unparameterized providers.

## Representation

```python
pose.to_rep("components" | "matrix")
pose.as_components()
pose.as_matrix()
```

Representation conversion keeps frame, sequence, batch, and parameter metadata
truthful. Matrix-layout poses do not carry stale component registry metadata.

## Transform Algebra

```python
pose.compose(other)
pose.inverse()
pose.apply(target)
```

Compose and inverse execute through canonical split form: cartesian position
plus quaternion rotation. Framed inverse is graph-free and requires the value
to be expressed in its parent basis; use
`pose.express_in(parent).inverse()` for a third-frame value. A value without a
parent is unframed only when its child and expression basis are also absent;
otherwise complete or clear its framing before inversion. `apply(...)` supports
compatible positions and kinematic payloads.

## Frames and Basis

```python
pose.express_in(dst, *, edge_pose_fn=None, graph=None, opts=None, validate=True)
Pose.solve_path_transform(src, dst, *, edge_pose_fn=None, graph=None, query=None, opts=None, validate=True)
```

`express_in(...)` changes basis only. Path solving changes relation by
composing edge poses along a frame graph path.
Direct class solves accept `query=` for dynamic providers. Object
`express_in(...)` uses the pose's own parameter grid when a required provider
is dynamic.

## Parameter Evaluation

`pose.param.at(...)` and `pose.param.resample_to(...)` use split typed
interpolation: position is numeric, rotation is rotation-aware. `on=...`
selects the parameter coordinate used for interpolation.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/pose
   :nosignatures:

   tal.spatial.Pose
   tal.spatial.Pose.graph
   tal.spatial.Pose.with_graph
   tal.spatial.Pose.register
   tal.spatial.Pose.from_components
   tal.spatial.Pose.from_matrix
   tal.spatial.Pose.decompose
   tal.spatial.Pose.to_rep
   tal.spatial.Pose.as_components
   tal.spatial.Pose.as_matrix
   tal.spatial.Pose.compose
   tal.spatial.Pose.inverse
   tal.spatial.Pose.apply
   tal.spatial.Pose.express_in
   tal.spatial.Pose.solve_path_transform
```

## See Also

- User guide: {doc}`../../user-guide/spatial`
- {doc}`position`
- {doc}`rotation`
