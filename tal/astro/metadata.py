from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from tal.core.schema import merge_schema

_ASTRO_KEYS = {
    "altitude_var",
    "azimuth_var",
    "backend",
    "direction_var",
    "kind",
    "observer_frame",
    "target",
    "time_scale",
}
_BACKENDS = frozenset({"astropy", "spice"})
_TIME_SCALES = frozenset({"utc", "tai", "tt", "tdb"})
_DIRECTION_KIND = "topocentric_direction"
_TARGET = "sun"
_OBSERVER_FRAME = "enu"
_DIRECTION_VAR = "direction"
_ALTITUDE_VAR = "altitude_deg"
_AZIMUTH_VAR = "azimuth_deg"


def _require_dataset(ds: object, *, owner: str) -> xr.Dataset:
    if isinstance(ds, xr.Dataset):
        return ds
    raise TypeError(f"{owner}: expected xr.Dataset; got {type(ds).__name__}.")


def _read_ext(ds: xr.Dataset, *, owner: str) -> Mapping[str, Any] | None:
    tal = ds.attrs.get("tal", {})
    if not isinstance(tal, Mapping):
        raise ValueError(f"{owner}: tal must be a mapping when present.")
    ext = tal.get("ext")
    if ext is None:
        return None
    if isinstance(ext, Mapping):
        return ext
    raise ValueError(f"{owner}: tal.ext must be a mapping when present.")


def _validate_keys(block: Mapping[Any, Any], *, owner: str) -> None:
    for key in sorted(block, key=lambda item: (type(item).__name__, repr(item))):
        if not isinstance(key, str) or key not in _ASTRO_KEYS:
            raise ValueError(
                f"{owner}: unknown key {key!r} at tal.ext.astro; allowed keys are {sorted(_ASTRO_KEYS)!r}."
            )


def _string_field(block: Mapping[str, Any], name: str, default: str, *, owner: str) -> str:
    value = block.get(name, default)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{owner}: tal.ext.astro.{name} must be a non-empty string.")


def _choice_field(
    block: Mapping[str, Any],
    name: str,
    default: str,
    *,
    choices: frozenset[str],
    owner: str,
) -> str:
    value = _string_field(block, name, default, owner=owner)
    if value in choices:
        return value
    raise ValueError(f"{owner}: tal.ext.astro.{name} must be one of {sorted(choices)!r}; got {value!r}.")


def _validate_fixed(value: str, *, expected: str, path: str, owner: str) -> str:
    if value == expected:
        return value
    raise ValueError(f"{owner}: {path} must be {expected!r}; got {value!r}.")


def read_astro_block(ds: xr.Dataset, *, owner: str) -> Mapping[str, Any] | None:
    """Read and validate the TAL astro extension block."""
    ds = _require_dataset(ds, owner=owner)
    ext = _read_ext(ds, owner=owner)
    if ext is None or "astro" not in ext:
        return None
    block = ext["astro"]
    if not isinstance(block, Mapping):
        raise ValueError(f"{owner}: tal.ext.astro must be a mapping when present.")
    _validate_keys(block, owner=owner)
    return block


def topocentric_metadata_payload(
    *,
    backend: str,
    time_scale: str,
    owner: str,
) -> dict[str, str]:
    """Build the canonical A1 topocentric-direction metadata payload."""
    normalized_backend = _choice_field({"backend": backend}, "backend", "astropy", choices=_BACKENDS, owner=owner)
    normalized_scale = _choice_field({"time_scale": time_scale}, "time_scale", "utc", choices=_TIME_SCALES, owner=owner)
    return {
        "kind": _DIRECTION_KIND,
        "target": _TARGET,
        "backend": normalized_backend,
        "observer_frame": _OBSERVER_FRAME,
        "time_scale": normalized_scale,
        "direction_var": _DIRECTION_VAR,
        "altitude_var": _ALTITUDE_VAR,
        "azimuth_var": _AZIMUTH_VAR,
    }


def _payload_from_block(block: Mapping[str, Any] | None, *, owner: str) -> dict[str, str]:
    source = block or {}
    kind = _validate_fixed(
        _string_field(source, "kind", _DIRECTION_KIND, owner=owner),
        expected=_DIRECTION_KIND,
        path="tal.ext.astro.kind",
        owner=owner,
    )
    target = _validate_fixed(
        _string_field(source, "target", _TARGET, owner=owner),
        expected=_TARGET,
        path="tal.ext.astro.target",
        owner=owner,
    )
    frame = _validate_fixed(
        _string_field(source, "observer_frame", _OBSERVER_FRAME, owner=owner),
        expected=_OBSERVER_FRAME,
        path="tal.ext.astro.observer_frame",
        owner=owner,
    )
    backend = _choice_field(source, "backend", "astropy", choices=_BACKENDS, owner=owner)
    time_scale = _choice_field(source, "time_scale", "utc", choices=_TIME_SCALES, owner=owner)
    direction_var = _validate_fixed(
        _string_field(source, "direction_var", _DIRECTION_VAR, owner=owner),
        expected=_DIRECTION_VAR,
        path="tal.ext.astro.direction_var",
        owner=owner,
    )
    altitude_var = _validate_fixed(
        _string_field(source, "altitude_var", _ALTITUDE_VAR, owner=owner),
        expected=_ALTITUDE_VAR,
        path="tal.ext.astro.altitude_var",
        owner=owner,
    )
    azimuth_var = _validate_fixed(
        _string_field(source, "azimuth_var", _AZIMUTH_VAR, owner=owner),
        expected=_AZIMUTH_VAR,
        path="tal.ext.astro.azimuth_var",
        owner=owner,
    )
    return {
        "kind": kind,
        "target": target,
        "backend": backend,
        "observer_frame": frame,
        "time_scale": time_scale,
        "direction_var": direction_var,
        "altitude_var": altitude_var,
        "azimuth_var": azimuth_var,
    }


def normalize_topocentric_metadata(
    ds: xr.Dataset,
    *,
    backend: str | None = None,
    time_scale: str | None = None,
    validate: bool,
    owner: str,
) -> xr.Dataset:
    """Normalize `tal.ext.astro` metadata for a TopocentricDirection."""
    block = read_astro_block(ds, owner=owner)
    payload = _payload_from_block(block, owner=owner)
    if backend is not None:
        payload["backend"] = topocentric_metadata_payload(backend=backend, time_scale=payload["time_scale"], owner=owner)[
            "backend"
        ]
    if time_scale is not None:
        payload["time_scale"] = topocentric_metadata_payload(
            backend=payload["backend"],
            time_scale=time_scale,
            owner=owner,
        )["time_scale"]
    return merge_schema(ds, {"ext": {"astro": payload}}, validate=validate)


__all__ = [
    "normalize_topocentric_metadata",
    "read_astro_block",
    "topocentric_metadata_payload",
]
