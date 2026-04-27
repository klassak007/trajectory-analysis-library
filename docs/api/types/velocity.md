(api-velocity)=
# Velocity Types

TAL models velocity as either separate linear/angular typed vectors or as a
combined spatial6 payload.

| Type | Payload |
| --- | --- |
| `LinearVelocity` | Cartesian xyz linear velocity. |
| `AngularVelocity` | Cartesian xyz angular velocity. |
| `Velocity` | Combined linear/angular velocity in components or vector6 representation. |

```{contents}
:local:
:depth: 2
```

## Constructor Contract

`LinearVelocity` and `AngularVelocity` require:

- one numeric data variable
- declared sequence roles
- one length-3 core dimension
- labels exactly `("x", "y", "z")`
- cartesian representation metadata

`Velocity` requires either component registry metadata for `linear` and
`angular` parts or a valid vector6 representation.

## Composition and Decomposition

```python
Velocity.from_linear_angular(linear, angular, *, validate=True)
Velocity.from_vector6(data, *, validate=True)
velocity.linear()
velocity.angular()
velocity.to_rep("components" | "vector6")
velocity.as_components()
velocity.as_vector6()
```

Composition uses exact coordinate alignment. If one component is framed and the
other is unframed, the result inherits the framed tags. If both are framed,
their tags must match.

## Magnitude and Transform Operations

```python
linear_velocity.norm(ord=...)
linear_velocity.magnitude()
angular_velocity.norm(ord=...)
angular_velocity.magnitude()
velocity.to_frame(...)
velocity.express_in(...)
```

`Rotation.apply(...)` and `Pose.apply(...)` support velocity targets. Component
registries are preserved so linear and angular parts remain truthful after
rotation.

## Temporal Operations

```python
linear_velocity.differentiate(...) -> LinearAcceleration
linear_velocity.integrate(...) -> Position
linear_velocity.smooth(...) -> LinearVelocity
angular_velocity.differentiate(...) -> AngularAcceleration
angular_velocity.smooth(...) -> AngularVelocity
velocity.differentiate(...) -> Acceleration
velocity.smooth(...) -> Velocity
```

Temporal operations use the selected parameter coordinate when `on=...` is
provided and preserve representation/frame metadata when invariants hold.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/velocity
   :nosignatures:

   tal.spatial.LinearVelocity
   tal.spatial.LinearVelocity.norm
   tal.spatial.LinearVelocity.magnitude
   tal.spatial.LinearVelocity.differentiate
   tal.spatial.LinearVelocity.integrate
   tal.spatial.LinearVelocity.smooth
   tal.spatial.LinearVelocity.to_frame
   tal.spatial.LinearVelocity.express_in
   tal.spatial.AngularVelocity
   tal.spatial.AngularVelocity.norm
   tal.spatial.AngularVelocity.magnitude
   tal.spatial.AngularVelocity.differentiate
   tal.spatial.AngularVelocity.smooth
   tal.spatial.AngularVelocity.to_frame
   tal.spatial.AngularVelocity.express_in
   tal.spatial.Velocity
   tal.spatial.Velocity.from_linear_angular
   tal.spatial.Velocity.from_vector6
   tal.spatial.Velocity.linear
   tal.spatial.Velocity.angular
   tal.spatial.Velocity.to_rep
   tal.spatial.Velocity.as_components
   tal.spatial.Velocity.as_vector6
   tal.spatial.Velocity.differentiate
   tal.spatial.Velocity.smooth
   tal.spatial.Velocity.to_frame
   tal.spatial.Velocity.express_in
```

## See Also

- {doc}`position`
- {doc}`acceleration`
- User guide: {doc}`../../user-guide/spatial`
