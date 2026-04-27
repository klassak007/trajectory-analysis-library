(user-guide-index)=
# User Guide

The TAL user guide is the practical path through the library. It starts with
ordinary `xarray` data, shows how to add trajectory semantics with
`AnalysisObject`, then moves into parameter-aware indexing, events, linear
algebra, spatial types, frames, and visualization.

Read the guide in order if you are new to TAL. Each chapter introduces one
piece of the mental model, gives a runnable example, and points to the API
reference when you need exact method contracts.

```{toctree}
:maxdepth: 1
:caption: User Guide

overview
core_concepts
creating_trajectory_objects
indexing
time
events
linalg
numpy
spatial
frames
viewing
```

For method-level contracts, see {doc}`../api/index`.
