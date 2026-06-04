(api-numba)=
# Numba Utilities

> **Audience:** User API. This experimental page is for users writing their
> own Numba-accelerated kernels over already-aligned NumPy blocks.

`tal.utils.numba` exposes policy-free mechanics for row preparation, explicit
scan-axis preparation, optional Numba loading, standard TAL compile policy, and
simple window-bound construction. TAL operation semantics, schema handling,
validity policy, and output finalization remain caller-owned.

The utilities are optional-dependency safe. Importing `tal.utils.numba` does
not import Numba. Compile user-owned kernels only after `require_numba(...)`
succeeds:

```python
from tal.utils import numba as tal_numba

def my_kernel(value):
    return value + 1

try:
    numba = tal_numba.require_numba("my.kernel")
except ImportError:
    compiled = None
else:
    compiled = tal_numba.njit_kernel(numba, my_kernel)
```

## Row and Window Preparation

Use `prepare_block_rows(...)` to broadcast already-aligned arrays into
contiguous row arrays. Use the stencil helpers to build stop-exclusive integer
window bounds. These helpers do not apply TAL domain policy; they only prepare
mechanical shapes for caller-owned kernels.

```python
import numpy as np
from tal.utils import numba as tal_numba

values = np.arange(24.0, dtype=np.float64).reshape(2, 4, 3)
rows = tal_numba.prepare_block_rows(
    (values,),
    (tal_numba.BlockInputSpec("values", 2, np.float64),),
    output_core_shape=(4, 3),
    owner="docs.numba",
)
bounds = tal_numba.centered_window_bounds(4, radius=1, owner="docs.numba")
```

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/numba
   :nosignatures:

   tal.utils.numba.BlockInputSpec
   tal.utils.numba.BlockRows
   tal.utils.numba.prepare_block_rows
   tal.utils.numba.row_count
   tal.utils.numba.ScanAxisSpec
   tal.utils.numba.ScanInputSpec
   tal.utils.numba.ScanRows
   tal.utils.numba.prepare_scan_rows
   tal.utils.numba.WindowBounds
   tal.utils.numba.clipped_window_bounds
   tal.utils.numba.centered_window_bounds
   tal.utils.numba.forward_window_bounds
   tal.utils.numba.backward_window_bounds
   tal.utils.numba.require_numba
   tal.utils.numba.njit_kernel
```
