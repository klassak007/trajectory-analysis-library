(api-position)=
# `Position`

`tal.spatial.Position` is an `AnalysisObject` subtype for cartesian xyz
position or displacement trajectories. It validates vector labels, spatial
metadata, optional frame tags, and temporal operations at the typed boundary.

```{contents}
:local:
:depth: 2
```

## Constructor Contract

- Input may be AO-like: `AnalysisObject`, `xarray.Dataset`, or
  `xarray.DataArray`.
- Exactly one numeric data variable is required.
- A declared `sequence_dim` is required.
- Exactly one core dimension of length `3` is required.
- Core labels must be exactly `("x", "y", "z")`.
- Representation is cartesian.

## Spatial Methods

```python
position.as_delta(...)
position.norm(ord=...)
position.magnitude()
position.to_frame(dst, *, edge_pose_fn, opts=None, validate=True)
position.express_in(dst, *, edge_rotation_fn, opts=None, validate=True)
```

`as_delta(...)` marks a position-like payload as displacement-like for
addition. `norm(...)` and `magnitude()` return scalar-core `tal.linalg.Array`
outputs.

`to_frame(...)` retargets the parent/child frame relation. `express_in(...)`
changes only the coordinate basis used to write components.

## Temporal Methods

```python
position.param.at(...)
position.param.resample_to(...)
position.differentiate(...)
position.smooth(...)
```

`differentiate(...)` returns `LinearVelocity`. `smooth(...)` returns
`Position`. Both preserve sequence and batch topology when invariants hold.

## Addition Rules

`Position + Position` is allowed only when frame and displacement intent are
unambiguous. Unframed addition requires exactly one operand to be marked with
`as_delta()`.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/position
   :nosignatures:

   tal.spatial.Position
   tal.spatial.Position.as_delta
   tal.spatial.Position.norm
   tal.spatial.Position.magnitude
   tal.spatial.Position.to_frame
   tal.spatial.Position.express_in
   tal.spatial.Position.differentiate
   tal.spatial.Position.smooth
```

## See Also

- User guide: {doc}`../../user-guide/spatial`
- {doc}`rotation`
- {doc}`pose`
- Frame metadata bridge: {doc}`../frames`
