(api-events)=
# Events

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

`ao.events` evaluates reusable conditions against trajectory data and returns
masks, boundary tables, interval tables, condition-selected layouts, or
event-centered windows.

## Methods

```python
ao.events.mask(condition, *, opts=None) -> xr.DataArray
ao.events.events(condition, *, opts=None) -> xr.Dataset
ao.events.intervals(condition, *, opts=None) -> xr.Dataset
ao.events.at_boundaries(condition, *, opts=None) -> AnalysisObject
ao.events.when(condition, *, opts=None) -> AnalysisObject
ao.events.around(events_or_condition, *, opts=None, edge=UNSET, pre=UNSET, post=UNSET, dt=UNSET, layout=UNSET) -> AnalysisObject
```

## Conditions

<!-- example-id: CORE-EVENT-SURFACE -->
```python
import numpy as np
import xarray as xr
from tal import AnalysisObject, ufuncs
from tal.core.event_ops import Condition

ao = AnalysisObject.from_data(
    xr.Dataset(
        {"value": ("sample", np.asarray([0.0, 2.0, 3.0, 1.0]))},
        coords={"sample": [0, 1, 2, 3], "time": ("sample", [0.0, 1.0, 2.0, 3.0])},
    ),
    sequence_dim="sample",
    core_dims=(),
    param_coord="time",
)

high = ao > 1.5
low = ao < 1.0
combined = (high | low) & ~(ao < 0.0)
same_as_two = ufuncs.equal(ao, 2.0)
assert ao.events.mask(combined).values.tolist() == [True, True, True, False]
assert ao.events.mask(same_as_two).values.tolist() == [False, True, False, False]

# Advanced named expression without a concrete AO operand.
after_two_seconds = Condition.compare(Condition.coord("time"), ">=", 2.0)
assert ao.events.mask(after_two_seconds).values.tolist() == [False, False, True, True]
```

Conditions are separate from output layout. The same predicate can drive a mask,
event boundary extraction, interval extraction, or event-window selection.
AO `==` and `!=` test object identity; `tal.ufuncs.equal(...)` and
`tal.ufuncs.not_equal(...)` build elementwise equality conditions.

## Layouts

`events.when(...)` supports:

- `layout="mask"`: preserve original topology and mask outside-condition values
- `layout="stream"`: pack selected samples into one sequence
- `layout="segments"`: return separate condition intervals

`events.around(...)` supports event-locked windows from condition boundaries or
explicit anchor times. Stacked layouts are useful when each matched event should
become one comparable row.

Pass common `edge`, `pre`, `post`, `dt`, and `layout` choices directly. Use
`AroundOptions` for advanced `eval` or `grid` configuration; combining `opts`
with direct overrides is rejected.

## Invariants

- Event APIs require sequence semantics.
- Window extraction needs a valid parameter/evaluation coordinate.
- Validity metadata limits the active sample domain.
- Unsupported truth or window policies fail during option validation.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/events
   :nosignatures:

   tal.core.event_ops.Condition
   tal.core.event_ops.EventsAccessor.mask
   tal.core.event_ops.EventsAccessor.events
   tal.core.event_ops.EventsAccessor.intervals
   tal.core.event_ops.EventsAccessor.at_boundaries
   tal.core.event_ops.EventsAccessor.when
   tal.core.event_ops.EventsAccessor.around
```

## See Also

- User guide: {doc}`../user-guide/events`
- {doc}`timebase`
