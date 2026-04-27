(api-vector)=
# `Vector`

`tal.linalg.Vector` is an `Array` with exactly one declared core dimension.
Use it when that payload axis should participate in vector operations.

## Contract

- Exactly one declared core dimension is required.
- `Vector.dot(other)` requires a compatible vector core dimension on both
  operands.
- `Vector.norm(ord=...)` validates the norm order and returns a scalar-core
  `Array`.
- Dot and norm preserve sequence and batch topology.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/vector
   :nosignatures:

   tal.linalg.Vector
   tal.linalg.Vector.dot
   tal.linalg.Vector.norm
```

## See Also

- {doc}`array`
- {doc}`matrix`
- {doc}`vector3`
- User guide: {doc}`../../user-guide/linalg`
