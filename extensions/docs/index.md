# TAL Extensions

Geo and astro are experimental, separately installed domain packages for TAL.
Their extraction preserves existing guarantees and limitations; it does not
establish new domain maturity. TAL Extensions 0.1.0 requires TAL 0.2.0.

```{toctree}
:maxdepth: 2
:caption: User guide

user-guide/geo
user-guide/astro
```

```{toctree}
:maxdepth: 2
:caption: API

api/types/geodetic
api/astro
```

See the [TAL documentation](https://klassak007.github.io/trajectory-analysis-library/)
for base trajectory semantics and the extension author API. Geo requires the
`tal-extensions[geo]` extra; astro requires `tal-extensions[astro]`.

Accessor integration is explicit. Call `tal_extensions.geo.register_position_accessor()`
once before using `Position.geo`. Importing either package does not install it.
Typed geo operations and interpolation work without registration.
