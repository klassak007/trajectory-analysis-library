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
ao.events.around(events_or_condition, *, opts=None) -> AnalysisObject
```

## Conditions

```python
from tal.core.event_ops import Condition

fast = Condition.compare(Condition.var("speed_mps"), "gt", 5.0)
near_goal = Condition.compare(Condition.coord("time_s"), "ge", 2.0)
combined = fast & near_goal
```

Conditions are separate from output layout. The same predicate can drive a mask,
event boundary extraction, interval extraction, or event-window selection.

## Layouts

`events.when(...)` supports:

- `layout="mask"`: preserve original topology and mask outside-condition values
- `layout="stream"`: pack selected samples into one sequence
- `layout="segments"`: return separate condition intervals

`events.around(...)` supports event-locked windows from condition boundaries or
explicit anchor times. Stacked layouts are useful when each matched event should
become one comparable row.

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
