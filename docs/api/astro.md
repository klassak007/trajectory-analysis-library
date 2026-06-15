(api-astro)=
# Astro

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

`tal.astro` provides foundation types for topocentric astronomy calculations.
`tal.astro.sun` adds the Astropy-backed Sun direction operation without making
the base `tal.astro` import heavy.

```{contents}
:local:
:depth: 2
```

## Foundation Types

```python
AstroBackend = Literal["astropy", "spice"]
AstroOptions(backend="astropy", time=None, iers=None)
AstroTimeOptions(scale="utc", source=None)
AstroIERSOptions(auto_download=False, degraded_accuracy="error")
TopocentricDirection(data)
```

`AstroBackend` is the public backend literal accepted by
`AstroOptions.backend`. Astro A1 records `"spice"` as a valid option value, but
SPICE behavior is contracted in a later slice.

`TopocentricDirection` stores a `direction` variable with ENU labels
`east`, `north`, and `up`. It also stores `altitude_deg` and `azimuth_deg`
payload variables derived from the ENU vector when they are not supplied.

## Sun Direction

```python
from tal.astro.sun import SpiceSunOptions, SunDirectionOptions, direction_to_sun
```

`direction_to_sun(...)` computes a topocentric ENU direction to the Sun from a
geodetic observer and absolute datetime-like observation time. A2 executes only
the Astropy backend. `SpiceSunOptions` is a public reserved options stub so the
final option shape is stable, but passing SPICE options or selecting
`backend="spice"` fails closed until a later backend phase.

No top-level `tal.astro.direction_to_sun` alias is added in A2; import the
operation from `tal.astro.sun`.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/astro
   :nosignatures:

   tal.astro.AstroBackend
   tal.astro.AstroOptions
   tal.astro.AstroTimeOptions
   tal.astro.AstroIERSOptions
   tal.astro.TopocentricDirection
   tal.astro.sun.SpiceSunOptions
   tal.astro.sun.SunDirectionOptions
   tal.astro.sun.direction_to_sun
```

## See Also

- User guide: {doc}`../user-guide/astro`
- Geodetic observers: {doc}`types/geodetic`
