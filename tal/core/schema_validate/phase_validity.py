from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from .common import ALLOWED_LAYOUTS, ALLOWED_VALIDITY_KEYS
from .common import (
    fail,
    is_mapping,
    is_valid_name,
    sorted_mapping_keys,
    unknown_key_actual,
    unknown_key_path,
)


def validity_block(core: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if "validity" not in core:
        return None
    validity = core["validity"]
    if is_mapping(validity):
        return validity
    fail(
        code="schema.validity.not_mapping",
        path="tal.core.validity",
        expected="mapping",
        actual=type(validity).__name__,
        hint="set tal.core.validity with sequence_size_coord and layout",
    )


def validate_validity_keys(validity: Mapping[str, Any]) -> None:
    for key in sorted_mapping_keys(validity):
        if key in ALLOWED_VALIDITY_KEYS:
            continue
        fail(
            code="schema.validity.unknown_key",
            path=unknown_key_path("tal.core.validity", key),
            expected=sorted(ALLOWED_VALIDITY_KEYS),
            actual=unknown_key_actual(key),
            hint="remove unknown validity keys",
        )


def sequence_size_coord_name(validity: Mapping[str, Any]) -> str:
    if "sequence_size_coord" not in validity:
        fail(
            code="schema.validity.sequence_size_coord.missing",
            path="tal.core.validity.sequence_size_coord",
            expected="non-empty string",
            actual=None,
            hint="set sequence_size_coord to a coordinate name",
        )
    name = validity["sequence_size_coord"]
    if is_valid_name(name):
        return name
    fail(
        code="schema.validity.sequence_size_coord.invalid",
        path="tal.core.validity.sequence_size_coord",
        expected="non-empty string",
        actual=name,
        hint="set sequence_size_coord to a coordinate name",
    )


def check_validity_coord_dims(ds: xr.Dataset, name: str, batch_dims: list[str]) -> None:
    dims = tuple(ds.coords[name].dims)
    if batch_dims and dims == tuple(batch_dims):
        return
    if not batch_dims and dims == ():
        return
    expected = list(batch_dims)
    hint = "set sequence_size_coord dims to exactly batch_dims"
    if not batch_dims:
        hint = "use scalar sequence_size_coord when batch_dims is empty"
    fail(
        code="schema.validity.sequence_size_coord.dims.invalid",
        path="tal.core.validity.sequence_size_coord",
        expected=expected,
        actual={"coord": name, "dims": list(dims)},
        hint=hint,
    )


def check_validity_layout(validity: Mapping[str, Any]) -> None:
    layout = validity.get("layout")
    if layout in ALLOWED_LAYOUTS:
        return
    fail(
        code="schema.validity.layout.invalid",
        path="tal.core.validity.layout",
        expected=sorted(ALLOWED_LAYOUTS),
        actual=layout,
        hint="set layout='left_packed'",
    )


def check_validity_coord_values(
    ds: xr.Dataset,
    *,
    name: str,
    sequence_dim: str,
) -> None:
    coord = ds.coords[name]
    if not np.issubdtype(np.dtype(coord.dtype), np.number):
        fail(
            code="schema.validity.sequence_size_coord.values.invalid",
            path="tal.core.validity.sequence_size_coord",
            expected=f"finite integer values in [0, {int(ds.sizes.get(sequence_dim, 0))}]",
            actual={"coord": name, "dtype": str(coord.dtype)},
            hint="use a numeric sequence_size_coord with finite integer values",
        )
    if getattr(coord.data, "chunks", None) is not None:
        fail(
            code="schema.validity.sequence_size_coord.values.invalid",
            path="tal.core.validity.sequence_size_coord",
            expected=f"finite integer values in [0, {int(ds.sizes.get(sequence_dim, 0))}]",
            actual={"coord": name, "reason": "chunked coordinate not schema-value-validatable"},
            hint="materialize or rechunk sequence_size_coord to an unchunked coord before schema validation",
        )
    values = np.asarray(coord.data, dtype="float64")
    ints = np.rint(values)
    sequence_len = int(ds.sizes.get(sequence_dim, 0))
    if np.any(~np.isfinite(values)) or np.any(ints != values) or np.any((ints < 0) | (ints > sequence_len)):
        fail(
            code="schema.validity.sequence_size_coord.values.invalid",
            path="tal.core.validity.sequence_size_coord",
            expected=f"finite integer values in [0, {sequence_len}]",
            actual={"coord": name, "dtype": str(coord.dtype)},
            hint="set sequence_size_coord values to finite integers within sequence bounds",
        )


def phase_validity(
    ds: xr.Dataset,
    *,
    core: Mapping[str, Any],
    sequence_dim: str | None,
    batch_dims: list[str],
) -> None:
    validity = validity_block(core)
    if validity is None:
        return
    if sequence_dim is None:
        fail(
            code="schema.validity.sequence_dim.missing",
            path="tal.core.roles.sequence_dim",
            expected="sequence_dim declared when tal.core.validity is present",
            actual=None,
            hint="set tal.core.roles.sequence_dim or remove tal.core.validity",
        )
    validate_validity_keys(validity)
    name = sequence_size_coord_name(validity)
    if name not in ds.coords:
        fail(
            code="schema.validity.sequence_size_coord.not_found",
            path="tal.core.validity.sequence_size_coord",
            expected="coord in ds.coords",
            actual={"coord": name, "coords": list(ds.coords)},
            hint="add the coordinate or choose an existing coord name",
        )
    check_validity_coord_dims(ds, name, batch_dims)
    check_validity_layout(validity)
    check_validity_coord_values(ds, name=name, sequence_dim=sequence_dim)
