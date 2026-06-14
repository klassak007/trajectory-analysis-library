from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AstroBackend = Literal["astropy", "spice"]
AstroTimeScale = Literal["utc", "tai", "tt", "tdb"]
AstroIERSDegradedAccuracy = Literal["error", "warn", "ignore"]

_BACKENDS = frozenset({"astropy", "spice"})
_TIME_SCALES = frozenset({"utc", "tai", "tt", "tdb"})
_IERS_DEGRADED = frozenset({"error", "warn", "ignore"})


@dataclass(frozen=True)
class AstroIERSOptions:
    """Options controlling IERS network and degraded-accuracy policy.

    Parameters
    ----------
    auto_download : bool, optional
        Whether astronomy backends may download IERS data.
    degraded_accuracy : {'error', 'warn', 'ignore'}, optional
        Policy for backend degraded-accuracy conditions.

    Examples
    --------
    >>> from tal.astro import AstroIERSOptions
    >>> AstroIERSOptions().degraded_accuracy
    'error'
    """

    auto_download: bool = False
    degraded_accuracy: AstroIERSDegradedAccuracy = "error"


@dataclass(frozen=True)
class AstroTimeOptions:
    """Options describing observation-time interpretation.

    Parameters
    ----------
    scale : {'utc', 'tai', 'tt', 'tdb'}, optional
        Time scale recorded in astro metadata and passed to future backends.
    source : str | None, optional
        Optional coordinate name used when time is resolved from an observer
        dataset.

    Examples
    --------
    >>> from tal.astro import AstroTimeOptions
    >>> AstroTimeOptions(source="utc_time").source
    'utc_time'
    """

    scale: AstroTimeScale = "utc"
    source: str | None = None


@dataclass(frozen=True)
class AstroOptions:
    """Foundation options shared by astro operations.

    Parameters
    ----------
    backend : {'astropy', 'spice'}, optional
        Requested astronomy backend. A1 records this option but does not
        dispatch Sun calculations.
    time : AstroTimeOptions | None, optional
        Observation-time interpretation options.
    iers : AstroIERSOptions | None, optional
        IERS policy passed to later Astropy-backed operations.

    Examples
    --------
    >>> from tal.astro import AstroOptions, AstroTimeOptions
    >>> opts = AstroOptions(time=AstroTimeOptions(scale="tt"))
    >>> opts.time.scale
    'tt'
    """

    backend: AstroBackend = "astropy"
    time: AstroTimeOptions | None = None
    iers: AstroIERSOptions | None = None


def _require_choice(value: object, *, choices: frozenset[str], field: str, owner: str) -> str:
    if isinstance(value, str) and value in choices:
        return value
    raise ValueError(f"{owner}: {field} must be one of {sorted(choices)!r}; got {value!r}.")


def coerce_iers_options(opts: AstroIERSOptions | None, *, owner: str) -> AstroIERSOptions:
    """Normalize IERS options at an operation boundary."""
    if opts is None:
        return AstroIERSOptions()
    if not isinstance(opts, AstroIERSOptions):
        raise TypeError(f"{owner}: iers options must be AstroIERSOptions or None.")
    if not isinstance(opts.auto_download, bool):
        raise TypeError(f"{owner}: iers.auto_download must be bool.")
    degraded = _require_choice(
        opts.degraded_accuracy,
        choices=_IERS_DEGRADED,
        field="iers.degraded_accuracy",
        owner=owner,
    )
    return AstroIERSOptions(auto_download=opts.auto_download, degraded_accuracy=degraded)  # type: ignore[arg-type]


def coerce_time_options(opts: AstroTimeOptions | None, *, owner: str) -> AstroTimeOptions:
    """Normalize astro time options at an operation boundary."""
    if opts is None:
        return AstroTimeOptions()
    if not isinstance(opts, AstroTimeOptions):
        raise TypeError(f"{owner}: time options must be AstroTimeOptions or None.")
    scale = _require_choice(opts.scale, choices=_TIME_SCALES, field="time.scale", owner=owner)
    if opts.source is not None and (not isinstance(opts.source, str) or not opts.source.strip()):
        raise TypeError(f"{owner}: time.source must be a non-empty string or None.")
    source = opts.source.strip() if isinstance(opts.source, str) else None
    return AstroTimeOptions(scale=scale, source=source)  # type: ignore[arg-type]


def coerce_astro_options(opts: AstroOptions | None, *, owner: str) -> AstroOptions:
    """Normalize foundation astro options at an operation boundary."""
    if opts is None:
        return AstroOptions()
    if not isinstance(opts, AstroOptions):
        raise TypeError(f"{owner}: options must be AstroOptions or None.")
    backend = _require_choice(opts.backend, choices=_BACKENDS, field="backend", owner=owner)
    time = None if opts.time is None else coerce_time_options(opts.time, owner=owner)
    iers = None if opts.iers is None else coerce_iers_options(opts.iers, owner=owner)
    return AstroOptions(backend=backend, time=time, iers=iers)  # type: ignore[arg-type]


__all__ = [
    "AstroBackend",
    "AstroIERSOptions",
    "AstroOptions",
    "AstroTimeOptions",
    "coerce_astro_options",
    "coerce_iers_options",
    "coerce_time_options",
]
