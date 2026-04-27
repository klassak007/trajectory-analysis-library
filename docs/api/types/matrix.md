(api-matrix)=
# `Matrix`

`tal.linalg.Matrix` is an `Array` with exactly two distinct declared core
dimensions. It provides typed transpose and linear-system helpers while
preserving sequence and batch topology.

## Contract

- Exactly two distinct declared core dimensions are required.
- `Matrix.T` swaps matrix core axes and repairs role metadata.
- `Matrix.solve(rhs, opts=None)` solves strict role-driven systems.
- `Matrix.inv()` requires square matrix topology.
- `Matrix.pinv(opts=None)` supports square and non-square matrix topology.

Solve, inverse, and pseudoinverse follow NumPy dtype behavior. Backend
`LinAlgError` paths are normalized to deterministic TAL `ValueError` messages.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/matrix
   :nosignatures:

   tal.linalg.Matrix
   tal.linalg.Matrix.T
   tal.linalg.Matrix.solve
   tal.linalg.Matrix.inv
   tal.linalg.Matrix.pinv
```

## See Also

- {doc}`array`
- {doc}`vector`
- User guide: {doc}`../../user-guide/linalg`
