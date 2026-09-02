from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from .common import ALLOWED_CORE_KEYS
from .common import (
    SCHEMA_VERSION,
    fail,
    is_active_schema_version,
    is_mapping,
    sorted_mapping_keys,
    unknown_key_actual,
    unknown_key_path,
)

ALLOWED_TAL_KEYS = {"version", "core", "ext"}


def phase_root_shape(ds: xr.Dataset) -> Mapping[str, Any]:
    if "tal" in ds.attrs and not is_mapping(ds.attrs["tal"]):
        fail(
            code="schema.not_mapping",
            path="tal",
            expected="mapping",
            actual=type(ds.attrs["tal"]).__name__,
            hint="set ds.attrs['tal'] to a mapping payload",
        )
    tal = ds.attrs.get("tal", {})
    for key in sorted_mapping_keys(tal):
        if key in ALLOWED_TAL_KEYS:
            continue
        fail(
            code="schema.unknown_key",
            path=unknown_key_path("tal", key),
            expected=sorted(ALLOWED_TAL_KEYS),
            actual=unknown_key_actual(key),
            hint="remove unknown key or move extension payload under tal.ext.<namespace>",
        )
    if "core" in tal and not is_mapping(tal["core"]):
        fail(
            code="schema.core.not_mapping",
            path="tal.core",
            expected="mapping",
            actual=type(tal["core"]).__name__,
            hint="set tal.core to a mapping",
        )
    if "ext" in tal and not is_mapping(tal["ext"]):
        fail(
            code="schema.not_mapping",
            path="tal.ext",
            expected="mapping",
            actual=type(tal["ext"]).__name__,
            hint="set tal.ext to a mapping",
        )
    return tal


def phase_version(tal: Mapping[str, Any]) -> None:
    version = tal.get("version")
    if not is_active_schema_version(version):
        fail(
            code="schema.version.invalid",
            path="tal.version",
            expected=SCHEMA_VERSION,
            actual=version,
            hint=f"set tal.version to {SCHEMA_VERSION}",
        )


def phase_core_envelope(tal: Mapping[str, Any]) -> Mapping[str, Any]:
    core = tal.get("core", {})
    for key in sorted_mapping_keys(core):
        if key in ALLOWED_CORE_KEYS:
            continue
        fail(
            code="schema.core.unknown_key",
            path=unknown_key_path("tal.core", key),
            expected=sorted(ALLOWED_CORE_KEYS),
            actual=unknown_key_actual(key),
            hint="remove unknown key or move it to tal.ext.<namespace>",
        )
    return core


def phase_extension_envelope(tal: Mapping[str, Any]) -> None:
    if "ext" in tal and not is_mapping(tal["ext"]):
        fail(
            code="schema.not_mapping",
            path="tal.ext",
            expected="mapping",
            actual=type(tal["ext"]).__name__,
            hint="set tal.ext to a mapping",
        )
    if "ext" not in tal:
        return
    ext = tal["ext"]
    for key in sorted_mapping_keys(ext):
        if isinstance(key, str) and key:
            continue
        fail(
            code="schema.ext.namespace.invalid",
            path=unknown_key_path("tal.ext", key),
            expected="non-empty string namespace key",
            actual=unknown_key_actual(key),
            hint="use a non-empty string key under tal.ext.<namespace>",
        )
