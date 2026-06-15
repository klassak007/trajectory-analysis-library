from __future__ import annotations

from importlib import import_module
from types import ModuleType

import numpy as np

from tal.astro.options import AstroIERSOptions


def require_astropy(owner: str) -> ModuleType:
    """Import Astropy at an explicit backend boundary."""
    try:
        return import_module("astropy")
    except ImportError as exc:
        raise ImportError(f"{owner}: Astropy is required for this astro backend. Install with 'tal[astro]'.") from exc


def _astropy_modules(owner: str) -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    require_astropy(owner)
    return (
        import_module("astropy.coordinates"),
        import_module("astropy.time"),
        import_module("astropy.units"),
        import_module("astropy.utils.iers"),
    )


def _with_iers_options(iers: ModuleType, opts: AstroIERSOptions):
    conf = iers.conf
    old_auto_download = conf.auto_download
    old_degraded_accuracy = conf.iers_degraded_accuracy
    conf.auto_download = opts.auto_download
    conf.iers_degraded_accuracy = opts.degraded_accuracy
    return conf, old_auto_download, old_degraded_accuracy


def compute_sun_altaz(
    *,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    altitude_m: np.ndarray,
    time: np.ndarray,
    scale: str,
    iers_options: AstroIERSOptions,
    owner: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Sun altitude and azimuth through Astropy."""
    coordinates, astropy_time, units, iers = _astropy_modules(owner)
    conf, old_auto_download, old_degraded_accuracy = _with_iers_options(iers, iers_options)
    try:
        location = coordinates.EarthLocation(
            lat=latitude_deg * units.deg,
            lon=longitude_deg * units.deg,
            height=altitude_m * units.m,
        )
        obstime = astropy_time.Time(time.astype("datetime64[ns]"), scale=scale)
        frame = coordinates.AltAz(obstime=obstime, location=location)
        sun_altaz = coordinates.get_sun(obstime).transform_to(frame)
        return np.asarray(sun_altaz.alt.deg, dtype=float), np.asarray(sun_altaz.az.deg, dtype=float)
    finally:
        conf.auto_download = old_auto_download
        conf.iers_degraded_accuracy = old_degraded_accuracy


__all__ = ["compute_sun_altaz", "require_astropy"]
