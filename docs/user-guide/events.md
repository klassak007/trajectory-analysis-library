(events-conditions)=
# Events and Conditions

Events are how TAL turns trajectory values into questions about when something
happened. A condition can produce a mask, boundary table, interval table,
condition-selected layout, or event-centered windows without rewriting the
predicate each time.

## Minimal Example

<!-- example-id: UG-EVENTS-WINDOWS -->
```python
import numpy as np
import xarray as xr
from tal.core import AnalysisObject
from tal.core.event_ops import (
    AroundOptions,
    AtBoundariesOptions,
    Condition,
    ConditionEvalOptions,
    EventExtractOptions,
    IntervalExtractOptions,
    WhenOptions,
)

sample = np.arange(8)
time_s = np.linspace(0.0, 0.7, sample.size)
speed = np.array([[0.0, 1.0, 2.0, 5.5, 6.2, 4.0, 1.0, 0.0]])

ao = AnalysisObject.from_data(
    xr.Dataset(
        {"speed_mps": (("trial", "sample"), speed)},
        coords={
            "trial": ["flight_0"],
            "sample": sample,
            "time_s": (("trial", "sample"), time_s[None, :]),
            "group_size": ("trial", np.array([sample.size], dtype=np.int64)),
        },
    ),
    sequence_dim="sample",
    batch_dims=("trial",),
    core_dims=(),
    param_coord="time_s",
    sequence_size_coord="group_size",
    validate=True,
)

fast = Condition.compare(Condition.var("speed_mps"), "gt", 5.0)
eval_opts = ConditionEvalOptions(coord_name="time_s")
mask = ao.events.mask(fast, opts=eval_opts)
events = ao.events.events(fast, opts=EventExtractOptions(eval=eval_opts))
intervals = ao.events.intervals(fast, opts=IntervalExtractOptions(eval=eval_opts))
boundaries = ao.events.at_boundaries(fast, opts=AtBoundariesOptions(eval=eval_opts, edges="enter"))
masked = ao.events.when(fast, opts=WhenOptions(layout="mask", eval=eval_opts))
stream = ao.events.when(fast, opts=WhenOptions(layout="stream", eval=eval_opts))
around = ao.events.around(fast, opts=AroundOptions(eval=eval_opts, edge="enter", pre=0.1, post=0.2, dt=0.1))
around_stacked = ao.events.around(
    fast,
    opts=AroundOptions(layout="stacked", eval=eval_opts, edge="enter", pre=0.1, post=0.2, dt=0.1),
)
```

The same `fast` condition drives every product. That makes event analysis easy
to audit: the predicate is separate from the output layout.

## Common Event Products

| API | Output |
| --- | --- |
| `ao.events.mask(...)` | Boolean mask over the original topology. |
| `ao.events.events(...)` | Boundary rows such as enter/exit events. |
| `ao.events.intervals(...)` | True/false interval rows. |
| `ao.events.at_boundaries(...)` | AO values sampled at event boundaries. |
| `ao.events.when(...)` | Samples selected by a condition. |
| `ao.events.around(...)` | Windows around boundaries or explicit anchors. |

`Condition.var(...)`, `Condition.coord(...)`, and `Condition.compare(...)` are
the building blocks. Conditions can be combined with `&`, `|`, `^`, and `~`.

## Layout Choices

| Layout | Shape |
| --- | --- |
| `when(..., layout="mask")` | Original sequence topology with outside-condition values masked. |
| `when(..., layout="stream")` | Selected samples packed into one sequence. |
| `when(..., layout="segments")` | Separate condition intervals with a segment dimension. |
| `around(..., layout="segments")` | One segment per matched event. |
| `around(..., layout="stacked")` | Stacked event-window layout. |

Event evaluation uses a context clock. In practice that clock comes from the
AO's declared `param_coord` or from `ConditionEvalOptions`. Validity metadata is
also respected, so padded tails do not create false crossings or windows.

## What Usually Goes Wrong

- Event APIs require sequence semantics.
- Window extraction needs a valid evaluation coordinate, usually the AO's
  `param_coord`.
- Exact boundary extraction can fail if the requested truth-evaluation mode is
  not supported for the condition.
- Invalid window geometry, such as negative `pre` or `post`, fails during
  option validation.

## Quick Checks

- Inspect `mask.dims` and `mask.sizes`.
- Inspect `events` and `intervals` directly.
- Compare `masked.as_dataset().sizes`, `stream.as_dataset().sizes`, and
  `around_stacked.as_dataset().sizes` to confirm you chose the right layout.

## See Also

- {doc}`time`
- {doc}`indexing`
- API: {doc}`../api/events`
