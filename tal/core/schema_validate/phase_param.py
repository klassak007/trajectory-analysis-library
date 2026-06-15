from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from .common import ALLOWED_PARAM_KEYS
from .common import (
    fail,
    is_mapping,
    is_valid_name,
    sorted_mapping_keys,
    unknown_key_actual,
    unknown_key_path,
)


def param_block(core: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if "param_coord" not in core:
        return None
    param = core["param_coord"]
    if is_mapping(param):
        return param
    fail(
        code="schema.param_coord.not_mapping",
        path="tal.core.param_coord",
        expected="mapping",
        actual=type(param).__name__,
        hint="set tal.core.param_coord = {'name': '<coord>'}",
    )


def validate_param_keys(param: Mapping[str, Any]) -> None:
    for key in sorted_mapping_keys(param):
        if key in ALLOWED_PARAM_KEYS:
            continue
        fail(
            code="schema.param_coord.unknown_key",
            path=unknown_key_path("tal.core.param_coord", key),
            expected=["name"],
            actual=unknown_key_actual(key),
            hint="remove unknown param_coord keys",
        )


def param_name(param: Mapping[str, Any]) -> str:
    if "name" not in param:
        fail(
            code="schema.param_coord.name.missing",
            path="tal.core.param_coord.name",
            expected="non-empty string",
            actual=None,
            hint="set param_coord.name to a dataset coordinate",
        )
    name = param["name"]
    if is_valid_name(name):
        return name
    fail(
        code="schema.param_coord.name.invalid",
        path="tal.core.param_coord.name",
        expected="non-empty string",
        actual=name,
        hint="set param_coord.name to a dataset coordinate",
    )


def allowed_param_dims(sequence_dim: str, batch_dims: list[str]) -> list[tuple[str, ...]]:
    dims = [(sequence_dim,), tuple(batch_dims) + (sequence_dim,)]
    out: list[tuple[str, ...]] = []
    for item in dims:
        if item not in out:
            out.append(item)
    return out


def phase_param_coord(
    ds: xr.Dataset,
    *,
    core: Mapping[str, Any],
    sequence_dim: str | None,
    batch_dims: list[str],
) -> None:
    param = param_block(core)
    if param is None:
        return
    if sequence_dim is None:
        fail(
            code="schema.param_coord.sequence_dim.missing",
            path="tal.core.roles.sequence_dim",
            expected="sequence_dim declared when tal.core.param_coord is present",
            actual=None,
            hint="set tal.core.roles.sequence_dim or remove tal.core.param_coord",
        )
    validate_param_keys(param)
    name = param_name(param)
    if name not in ds.coords:
        fail(
            code="schema.param_coord.not_found",
            path="tal.core.param_coord.name",
            expected="coord in ds.coords",
            actual={"coord": name, "coords": list(ds.coords)},
            hint="add the coordinate or choose an existing coord name",
        )
    dims = tuple(ds.coords[name].dims)
    allowed = allowed_param_dims(sequence_dim, batch_dims)
    if dims not in allowed:
        fail(
            code="schema.param_coord.dims.invalid",
            path="tal.core.param_coord.name",
            expected=[list(item) for item in allowed],
            actual={"coord": name, "dims": list(dims)},
            hint="use dims (sequence_dim,) or (*batch_dims, sequence_dim)",
        )
    dtype = np.dtype(ds.coords[name].dtype)
    if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.datetime64)):
        fail(
            code="schema.param_coord.dtype.invalid",
            path="tal.core.param_coord.name",
            expected="numeric or datetime64 coordinate dtype",
            actual={"coord": name, "dtype": str(ds.coords[name].dtype)},
            hint="set param_coord to a numeric or datetime64 coordinate",
        )
