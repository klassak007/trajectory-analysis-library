(api-viz)=
# Visualization

`tal.viz` and `ao.viz` provide optional plotting helpers for quick inspection
of trajectories, variables, grouped runs, and registered components. The
visualization layer reuses core AO coercion, dataset context resolution,
validity masking, grouping, and component extraction.

```{contents}
:local:
:depth: 2
```

## Optional Dependencies

Visualization requires optional runtime packages such as HoloViews and hvPlot.
Install them with:

```bash
pip install "tal[viz]"
```

## Options

```python
from tal.viz import AOVizOptions

AOVizOptions(
    var=None,
    x=None,
    by=(),
    groupby=(),
    group_key=None,
    group_foundation_opts=None,
    validity="respect",
    max_overlay_items=8,
    kwargs={},
)
```

Common fields:

- `var`: explicit payload variable.
- `x`: explicit x-axis coordinate.
- `by`: overlay channels.
- `groupby`: interactive grouping channels.
- `group_key`: one grouping key resolved by TAL grouping owners.
- `validity`: `"respect"` masks invalid sequence tails; `"ignore"` bypasses
  validity masking.

## Functional API

```python
from tal.viz import AOVizOptions, component, explorer, line, scatter

plot1 = line(ao)
plot2 = scatter(ao, opts=AOVizOptions(x="time", by=("trial",)))
plot3 = explorer(ao, opts=AOVizOptions(group_key="trial"))
plot4 = component(ao, "xyz", kind="line")
```

## Accessor API

```python
ao.viz.line(opts=AOVizOptions(...))
ao.viz.scatter(opts=AOVizOptions(...))
ao.viz.explorer(opts=AOVizOptions(...))
ao.viz.component("xyz", kind="explorer", opts=AOVizOptions(...))
```

## Component Plotting

`ao.viz.component(name, ...)` resolves the component through the component
registry, extracts a component AO, then dispatches to `line`, `scatter`, or
`explorer`.

When both `by` and `groupby` are omitted, TAL overlays component labels when the
label count is small enough and uses grouping otherwise.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/viz
   :nosignatures:

   tal.viz.AOVizOptions
   tal.viz.line
   tal.viz.scatter
   tal.viz.explorer
   tal.viz.component
   tal.viz.AnalysisObjectVizAccessor.line
   tal.viz.AnalysisObjectVizAccessor.scatter
   tal.viz.AnalysisObjectVizAccessor.explorer
   tal.viz.AnalysisObjectVizAccessor.component
```

## See Also

- User guide: {doc}`../user-guide/viewing`
- {doc}`components`
