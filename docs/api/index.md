(api-index)=
# API Reference

The API reference is organized by audience. **User API** pages are for normal
analysis workflows. **Extension Author API** pages are for package authors
building typed `AnalysisObject` subclasses and domain operations. **Internal
API** pages define what is intentionally outside the supported documentation
surface.

```{contents}
:local:
:depth: 2
```

```{toctree}
:maxdepth: 2
:caption: User API

analysis-object
types/index
components
concat
timebase
events
reducers
schema
ufuncs
numba
catalog
astro
frames
io
viz
```

```{toctree}
:maxdepth: 1
:caption: Extension Author API

domain-extensions
```

```{toctree}
:maxdepth: 1
:caption: Internal API Policy

internal-api
```
