(api-ufuncs)=
# Universal Ufuncs

`tal.ufuncs` provides AO-aware wrappers for xarray-compatible unary, binary,
comparison, and logical ufunc behavior. The wrappers keep xarray label
alignment while finalizing TAL schema metadata on AO outputs.

```python
from tal import ufuncs

y = ufuncs.sin(ao)
z = ufuncs.add(ao_left, ao_right)
condition = ufuncs.greater(ao, 0.0)
```

## Return Policy

- Numeric unary and binary wrappers return finalized AOs when an AO operand is
  present.
- Ordering comparisons return `Condition` for AO-aware operands.
- Logical wrappers operate on `Condition` expressions and return `Condition`.
- Non-AO/non-condition inputs keep native xarray or scalar behavior.

## Alignment Intent

AO operands can carry `.a(...)` and `.b()` hints into supported ufunc families.
Those hints are consumed by TAL's operation planners and do not mutate source
data.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/ufuncs
   :nosignatures:

   tal.ufuncs
```

## See Also

- User guide: {doc}`../user-guide/numpy`
- {doc}`analysis-object`
