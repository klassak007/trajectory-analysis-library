# TAL Extensions

Experimental geo and astro packages developed alongside TAL in this monorepo.
TAL Extensions 0.1.0 requires exactly TAL 0.2.0. Installing TAL alone does not
include these packages. Shared metadata continues to use `tal.ext.geo` and
`tal.ext.astro`; it does not require the extensions to be installed.

From the repository root, install both local projects for development:

```sh
python -m pip install -e '.[dev]'
python -m pip install -e './extensions[dev]'
```

Select `tal-extensions[geo]` for pyproj, or `tal-extensions[astro]` for pyproj and
Astropy. The root import loads no domains or optional backends. Import explicitly:

```python
from tal_extensions.geo import GeodeticPosition, register_position_accessor

# Optional convenience integration, called explicitly by the application.
register_position_accessor()
```

`Position.geo` is installed only by the registration call. Typed geo constructors
and interpolation work without it. Registration is idempotent for this extension
and rejects an attribute installed by another owner.

Geo retains its WGS84/ellipsoidal-height boundary. Astro currently supports the
Astropy Sun-direction backend, rejects Dask-backed observer/time inputs before
compute, and retains reserved SPICE options that fail closed. IERS configuration
uses shared Astropy state; concurrent backend calls are not hardened. Extraction
does not establish production readiness or add SPICE/GTSAM implementations.

Run project-owned tests and documentation builds separately:

```sh
python -m pytest -c extensions/pyproject.toml extensions/tests
python -m sphinx -n -W -b html docs /tmp/tal-docs
python -m sphinx -n -W -b html extensions/docs /tmp/tal-extension-docs
lint-imports --config pyproject.toml
```

The [extension documentation sources](docs/index.md) describe domain APIs.
The [TAL author API](../docs/api/domain-extensions.md) defines integration owners.
Both documentation builds use the same visual theme; hosting and publishing are
separate decisions.
