(linalg-guide)=
# Linear Algebra

`tal.linalg` gives vector and matrix operations a TAL-aware boundary. Kernels
still operate on NumPy/xarray-compatible arrays, but the wrapper knows which
dimensions are payload axes and which dimensions are sequence or batch topology.

## Minimal Example

<!-- example-id: UG-LINALG-BASIC -->
```python
import numpy as np
import xarray as xr
from tal.core import AnalysisObject
from tal.linalg import Matrix, Vector, Vector3, add, dot, inv, matmul, norm, pinv, solve, sub

A = Matrix(AnalysisObject.from_data(
    xr.Dataset(
        {"A": (("sample", "row", "col"), [[[2.0, 0.0], [0.0, 3.0]], [[1.0, 1.0], [0.0, 2.0]]])},
        coords={"sample": [0, 1], "row": ["r0", "r1"], "col": ["c0", "c1"]},
    ),
    sequence_dim="sample",
    core_dims=("row", "col"),
    validate=True,
))

v = Vector(AnalysisObject.from_data(
    xr.Dataset(
        {"v": (("sample", "col"), [[4.0, 5.0], [2.0, 8.0]])},
        coords={"sample": [0, 1], "col": ["c0", "c1"]},
    ),
    sequence_dim="sample",
    core_dims=("col",),
    validate=True,
))

x_component = AnalysisObject.from_data(
    xr.Dataset({"x": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
    sequence_dim="sample",
    core_dims=(),
    validate=True,
)

Av = A @ v
Av2 = matmul(A, v)
sum_v = add(v, v)
diff_v = sub(v, v)
energy = dot(v, v)
energy2 = v.dot(v)
mag = norm(v)
mag2 = v.norm()
x = solve(A, v)
x2 = A.solve(v)
Ainv = inv(A)
Ainv2 = A.inv()
Apinv = pinv(A)
Apinv2 = A.pinv()
vec3 = Vector3.from_xyz(x_component, 0.0, 1.0, axis="axis", output_var="vec3")
```

The important declaration is `core_dims`. `Matrix` consumes two core dimensions;
`Vector` consumes one; scalar outputs have `core_dims=()`.

## Choosing A Typed Wrapper

| Type | Use When |
| --- | --- |
| `Array` | You need a general TAL-aware numeric payload. |
| `Vector` | Exactly one core dimension is a vector axis. |
| `Vector3` | The vector axis must be labeled `x`, `y`, `z`. |
| `Matrix` | Exactly two core dimensions are matrix axes. |

Use `set_core_dims(...)`, `as_core(...)`, `.axis(...)`, or `.rc(...)` when an
existing AO already has truthful sequence and batch roles and only needs
payload axes retagged.

## Core Operations

The top-level functions and methods are interchangeable where both are
available:

- `tal.linalg.add`, `sub`, `matmul`
- `tal.linalg.dot`, `norm`
- `tal.linalg.solve`, `inv`, `pinv`
- `Vector.dot(...)`, `Vector.norm(...)`
- `Matrix.solve(...)`, `Matrix.inv(...)`, `Matrix.pinv(...)`

For structural payload work, use `tal.linalg.assemble_core`,
`tal.linalg.stack_core`, `tal.linalg.block_core`, `tal.linalg.concat_core`,
`tal.linalg.decompose_core(...)`, and `tal.linalg.overlay_core(...)`.

## What Usually Goes Wrong

- `Vector` requires exactly one declared core dimension.
- `Matrix` requires exactly two distinct declared core dimensions.
- Matrix multiplication and solve operations require compatible labels and
  sizes on the participating core axes.
- Ambiguous numeric variable selection fails before kernel execution.
- Structural core operations require explicit options so output layout stays
  deterministic.

## Quick Checks

- Inspect `Av.as_dataset()`, `energy.as_dataset()`, `mag.as_dataset()`, and
  `x.as_dataset()`.
- Check `vec3.as_dataset().coords["axis"]` to confirm the canonical
  `("x", "y", "z")` labels.

## See Also

- {doc}`numpy`
- {doc}`core_concepts`
- API: {doc}`../api/types/array`
