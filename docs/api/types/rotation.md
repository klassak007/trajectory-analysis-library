(api-rotation)=
# `Rotation`

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

`tal.spatial.Rotation` is an `AnalysisObject` subtype for rotation
trajectories. It supports quaternion and matrix representations, rotation-aware
composition, inverse, application to spatial payloads, and typed parameter
evaluation.

```{contents}
:local:
:depth: 2
```

## Constructor Contract

- Input may be AO-like.
- Exactly one numeric data variable is required.
- Sequence and batch roles are optional; representation core roles must be
  declared, including through `Rotation.from_data(...)`.
- Quaternion representation uses one declared core dimension labeled
  `("x", "y", "z", "w")`.
- Matrix representation uses two distinct length-3 core dimensions labeled
  `("x", "y", "z")`.
- Missing representation metadata means quaternion input. Raw matrix input must
  first declare matrix representation metadata through the spatial metadata
  owner.
- Malformed frame or spatial-role metadata fails at construction.

## Representation

```python
rotation.to_rep("quat" | "matrix")
rotation.as_quat()
rotation.as_matrix()
```

Conversion preserves sequence, batch, parameter, validity, and frame metadata.
It preserves the payload variable name and chooses component dimensions that
avoid all existing dimension, coordinate, and data-variable names.

## Rotation Algebra

```python
rotation.compose(other)
rotation.inverse()
rotation.apply(target)
rotation.norm()
rotation.magnitude()
```

`compose(...)` and `inverse()` execute through quaternion semantics and preserve
the requested representation policy. `apply(...)` supports compatible
positions, velocities, accelerations, and composite spatial6 payloads.

`norm()` and `magnitude()` return geodesic angle magnitude from identity in
radians.

## Frames and Path Solving

```python
rotation.express_in(dst, *, edge_rotation_fn, opts=None, validate=True)
Rotation.solve_path_transform(src, dst, *, edge_rotation_fn, opts=None, validate=True)
```

`express_in(...)` changes coordinate basis only. Path solving composes edge
rotations through a `FrameGraph`.

## Parameter Evaluation

```python
rotation.param.at(...)
rotation.param.resample_to(...)
rotation.slerp(query, ...)
```

Typed rotation interpolation can use nearest, normalized linear blend, or SLERP
semantics. `on=...` selects the parameter coordinate used for correspondence.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/rotation
   :nosignatures:

   tal.spatial.Rotation
   tal.spatial.Rotation.from_data
   tal.spatial.Rotation.to_rep
   tal.spatial.Rotation.as_quat
   tal.spatial.Rotation.as_matrix
   tal.spatial.Rotation.norm
   tal.spatial.Rotation.magnitude
   tal.spatial.Rotation.compose
   tal.spatial.Rotation.inverse
   tal.spatial.Rotation.apply
   tal.spatial.Rotation.express_in
   tal.spatial.Rotation.slerp
   tal.spatial.Rotation.solve_path_transform
```

## See Also

- User guide: {doc}`../../user-guide/spatial`
- {doc}`position`
- {doc}`pose`
