(api-numba)=
# Numba Utilities

> **Audience:** User API. This experimental page is for users writing their
> own Numba-accelerated kernels over already-aligned NumPy blocks.

`tal.utils.numba` exposes policy-free mechanics for row preparation, explicit
scan-axis preparation, optional Numba loading, standard TAL compile policy,
window-bound construction, window iteration metadata, topology-row metadata,
and simple benchmark timing. TAL operation semantics, schema handling, validity
policy, and output finalization remain caller-owned.

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
window_rows = tal_numba.prepare_window_rows(bounds, owner="docs.numba")
assert window_rows.length == 4
```

## Scan and Topology Metadata

Use `prepare_scan_rows(...)` when a kernel has one or more explicit ordered
axes. Use `prepare_topology_rows(...)` as a single-axis convenience for ordered
topology rows. Nested shapes remain explicit through `prepare_scan_rows(...)`.

```python
import numpy as np
from tal.utils import numba as tal_numba

links = np.zeros((2, 4, 3), dtype=np.float64)
topology_rows = tal_numba.prepare_topology_rows(
    (links,),
    (tal_numba.ScanInputSpec("links", 1, 1, np.float64),),
    topology_axis="chain",
    output_core_shapes=((3,),),
    owner="docs.numba",
)
assert topology_rows.outer_shape == (2,)
assert topology_rows.ordered_shape == (4,)

nested_rows = tal_numba.prepare_scan_rows(
    (np.zeros((2, 5, 4, 3), dtype=np.float64),),
    (tal_numba.ScanInputSpec("links", 2, 1, np.float64),),
    ordered_axes=(
        tal_numba.ScanAxisSpec("time", "scan"),
        tal_numba.ScanAxisSpec("chain", "topology"),
    ),
    output_core_shapes=((3,),),
    owner="docs.numba",
)
assert nested_rows.ordered_shape == (5, 4)
```

## Timing Helpers

Use `time_once(...)` and `warm_median(...)` for lightweight timing in examples.
Use `cold_subprocess(...)` when a benchmark script prints one float and needs a
fresh `NUMBA_CACHE_DIR`.

```python
from tal.utils import numba as tal_numba

elapsed = tal_numba.time_once(lambda value: value + 1, 1)
median = tal_numba.warm_median(lambda value: value + 1, 1, repeats=1)
calls = tal_numba.break_even_calls(baseline=10.0, cold=25.0, warm=5.0)
assert elapsed >= 0.0
assert median >= 0.0
assert calls == 4.0
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
   tal.utils.numba.prepare_topology_rows
   tal.utils.numba.WindowBounds
   tal.utils.numba.WindowRows
   tal.utils.numba.clipped_window_bounds
   tal.utils.numba.centered_window_bounds
   tal.utils.numba.forward_window_bounds
   tal.utils.numba.backward_window_bounds
   tal.utils.numba.prepare_window_rows
   tal.utils.numba.require_numba
   tal.utils.numba.njit_kernel
   tal.utils.numba.time_once
   tal.utils.numba.warm_median
   tal.utils.numba.cold_subprocess
   tal.utils.numba.break_even_calls
```
