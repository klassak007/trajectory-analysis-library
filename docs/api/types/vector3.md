(api-vector3)=
# `Vector3`

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

`tal.linalg.Vector3` is a `Vector` whose single core axis is exactly
`("x", "y", "z")`. It is useful for payloads that must carry explicit xyz
semantics.

## Contract

- Exactly one declared core dimension.
- Core-axis length is exactly `3`.
- Core-axis labels are exactly `("x", "y", "z")`.
- No implicit relabeling or repair is performed.

## Construction From Components

```python
Vector3.from_xyz(x, y, z, *, axis="axis", output_var=None, validate=True)
```

Components may be AO-like values or scalars. At least one component must be
AO-like when scalars are used, so TAL has topology to broadcast against.

## Component Accessors

`x`, `y`, and `z` return scalar-core `Array` outputs, not `Vector3` objects.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/vector3
   :nosignatures:

   tal.linalg.Vector3
   tal.linalg.Vector3.from_xyz
   tal.linalg.Vector3.x
   tal.linalg.Vector3.y
   tal.linalg.Vector3.z
```

## See Also

- {doc}`vector`
- {doc}`array`
- User guide: {doc}`../../user-guide/linalg`
