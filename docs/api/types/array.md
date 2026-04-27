(api-array)=
# `Array`

`tal.linalg.Array` is the base typed wrapper for dimension-aware linear algebra.
It is an `AnalysisObject` subtype that requires declared core dimensions before
running strict linalg operations.

```{contents}
:local:
:depth: 2
```

## Construction

```python
from tal.linalg import Array

arr = Array(ao_or_xarray)
arr = Array(ao_or_xarray, core_dims=("row", "col"))
```

Accepted inputs are `Array`, `AnalysisObject`, `xarray.Dataset`, and
`xarray.DataArray`. The `core_dims=...` shorthand is for inputs that already
carry sequence and batch role metadata; raw xarray inputs should usually be
wrapped with `AnalysisObject.from_data(...)` first.

## Role Helpers

```python
arr.set_core_dims("axis")
arr.set_vector_axis("axis")
arr.set_matrix_axes("row", "col")
arr.as_core("axis")
arr.axis("axis")
arr.rc("row", "col")
```

These helpers retag payload axes while preserving the existing AO topology.

## Numeric Operations

Top-level functions:

```python
from tal.linalg import add, dot, inv, matmul, norm, pinv, solve, sub
```

Operator helpers:

- `Array.__add__`, `__radd__`
- `Array.__sub__`, `__rsub__`
- `Array.__matmul__`

Contracts:

- add/sub require exact core-dimension equality.
- dot/norm require vector payloads and return scalar-core outputs.
- matmul/solve/inv/pinv use strict core-label compatibility.
- linear algebra backend errors are normalized to TAL owner-prefixed
  `ValueError` messages.
- Dask-backed dot/norm preserve laziness.

## Core Assembly

```python
Array.assemble_core(...)
Array.stack_core(...)
Array.block_core(...)
Array.concat_core(...)
arr.decompose_core(...)
arr.overlay_core(...)
```

Assembly operations create or modify payload axes through shared core combine
owners. They require explicit options for layouts that could otherwise be
ambiguous.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/array
   :nosignatures:

   tal.linalg.Array
   tal.linalg.Array.set_core_dims
   tal.linalg.Array.set_vector_axis
   tal.linalg.Array.set_matrix_axes
   tal.linalg.Array.as_core
   tal.linalg.Array.axis
   tal.linalg.Array.rc
   tal.linalg.add
   tal.linalg.sub
   tal.linalg.matmul
   tal.linalg.solve
   tal.linalg.inv
   tal.linalg.pinv
   tal.linalg.dot
   tal.linalg.norm
   tal.linalg.assemble_core
   tal.linalg.stack_core
   tal.linalg.block_core
   tal.linalg.concat_core
   tal.linalg.Array.assemble_core
   tal.linalg.Array.stack_core
   tal.linalg.Array.block_core
   tal.linalg.Array.concat_core
   tal.linalg.Array.decompose_core
   tal.linalg.decompose_core
   tal.linalg.Array.overlay_core
   tal.linalg.overlay_core
```

## See Also

- {doc}`vector`
- {doc}`matrix`
- User guide: {doc}`../../user-guide/linalg`
