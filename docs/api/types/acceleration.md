(api-acceleration)=
# Acceleration Types

TAL models acceleration as either separate linear/angular typed vectors or as a
combined spatial6 payload.

| Type | Payload |
| --- | --- |
| `LinearAcceleration` | Cartesian xyz linear acceleration. |
| `AngularAcceleration` | Cartesian xyz angular acceleration. |
| `Acceleration` | Combined linear/angular acceleration in components or vector6 representation. |

```{contents}
:local:
:depth: 2
```

## Constructor Contract

`LinearAcceleration` and `AngularAcceleration` require:

- one numeric data variable
- declared sequence roles
- one length-3 core dimension
- labels exactly `("x", "y", "z")`
- cartesian representation metadata

`Acceleration` requires valid linear/angular component metadata or a valid
vector6 representation.

## Composition and Decomposition

```python
Acceleration.from_linear_angular(linear, angular, *, validate=True)
Acceleration.from_vector6(data, *, validate=True)
acceleration.linear()
acceleration.angular()
acceleration.to_rep("components" | "vector6")
acceleration.as_components()
acceleration.as_vector6()
```

Composition uses exact coordinate alignment. Frame tags must either match or be
unambiguous enough to inherit from a single framed component.

## Magnitude and Transform Operations

```python
linear_acceleration.norm(ord=...)
linear_acceleration.magnitude()
angular_acceleration.norm(ord=...)
angular_acceleration.magnitude()
acceleration.to_frame(...)
acceleration.express_in(...)
```

`Rotation.apply(...)` and `Pose.apply(...)` support acceleration targets.
Component registries are preserved across component-wise rotation.

## Temporal Operations

```python
linear_acceleration.integrate(...) -> LinearVelocity
linear_acceleration.smooth(...) -> LinearAcceleration
angular_acceleration.integrate(...) -> AngularVelocity
angular_acceleration.smooth(...) -> AngularAcceleration
acceleration.integrate(...) -> Velocity
acceleration.smooth(...) -> Acceleration
```

Temporal operations use the selected parameter coordinate when `on=...` is
provided and preserve representation/frame metadata when invariants hold.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/acceleration
   :nosignatures:

   tal.spatial.LinearAcceleration
   tal.spatial.LinearAcceleration.norm
   tal.spatial.LinearAcceleration.magnitude
   tal.spatial.LinearAcceleration.integrate
   tal.spatial.LinearAcceleration.smooth
   tal.spatial.LinearAcceleration.to_frame
   tal.spatial.LinearAcceleration.express_in
   tal.spatial.AngularAcceleration
   tal.spatial.AngularAcceleration.norm
   tal.spatial.AngularAcceleration.magnitude
   tal.spatial.AngularAcceleration.integrate
   tal.spatial.AngularAcceleration.smooth
   tal.spatial.AngularAcceleration.to_frame
   tal.spatial.AngularAcceleration.express_in
   tal.spatial.Acceleration
   tal.spatial.Acceleration.from_linear_angular
   tal.spatial.Acceleration.from_vector6
   tal.spatial.Acceleration.linear
   tal.spatial.Acceleration.angular
   tal.spatial.Acceleration.to_rep
   tal.spatial.Acceleration.as_components
   tal.spatial.Acceleration.as_vector6
   tal.spatial.Acceleration.integrate
   tal.spatial.Acceleration.smooth
   tal.spatial.Acceleration.to_frame
   tal.spatial.Acceleration.express_in
```

## See Also

- {doc}`velocity`
- {doc}`position`
- User guide: {doc}`../../user-guide/spatial`
