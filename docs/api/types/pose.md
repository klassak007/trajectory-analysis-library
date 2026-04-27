(api-pose)=
# `Pose`

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
plus quaternion rotation. `apply(...)` supports compatible positions and
kinematic payloads.

## Frames and Basis

```python
pose.express_in(dst, *, edge_pose_fn, opts=None, validate=True)
Pose.solve_path_transform(src, dst, *, edge_pose_fn, opts=None, validate=True)
```

`express_in(...)` changes basis only. Path solving changes relation by
composing edge poses along a frame graph path.

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
