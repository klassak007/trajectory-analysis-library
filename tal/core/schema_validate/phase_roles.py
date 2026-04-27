from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from .common import ALLOWED_ROLE_KEYS
from .common import (
    fail,
    first_duplicate,
    is_mapping,
    is_valid_name,
    is_valid_name_list,
    sorted_mapping_keys,
    unknown_key_actual,
    unknown_key_path,
)


def roles_block(core: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if "roles" not in core:
        return None
    roles = core.get("roles")
    if not is_mapping(roles):
        fail(
            code="schema.roles.not_mapping",
            path="tal.core.roles",
            expected="mapping",
            actual=type(roles).__name__,
            hint="set tal.core.roles with sequence_dim, batch_dims, core_dims",
        )
    return roles


def validate_role_keys(roles: Mapping[str, Any]) -> None:
    for key in sorted_mapping_keys(roles):
        if key in ALLOWED_ROLE_KEYS:
            continue
        fail(
            code="schema.roles.unknown_key",
            path=unknown_key_path("tal.core.roles", key),
            expected=sorted(ALLOWED_ROLE_KEYS),
            actual=unknown_key_actual(key),
            hint="remove unknown roles key",
        )


def extract_role_fields(roles: Mapping[str, Any]) -> tuple[str | None, list[str], list[str]]:
    sequence_dim = roles.get("sequence_dim")
    if sequence_dim is not None and not is_valid_name(sequence_dim):
        fail(
            code="schema.roles.sequence_dim.invalid",
            path="tal.core.roles.sequence_dim",
            expected="non-empty string",
            actual=sequence_dim,
            hint="set sequence_dim to a dataset dimension name",
        )
    batch_dims = roles.get("batch_dims", [])
    if not is_valid_name_list(batch_dims):
        fail(
            code="schema.roles.batch_dims.invalid",
            path="tal.core.roles.batch_dims",
            expected="list[str]",
            actual=batch_dims,
            hint="set batch_dims to a list of dataset dimension names",
        )
    core_dims = roles.get("core_dims")
    if not is_valid_name_list(core_dims):
        fail(
            code="schema.roles.core_dims.invalid",
            path="tal.core.roles.core_dims",
            expected="list[str]",
            actual=core_dims,
            hint="set core_dims to a list of dataset dimension names",
        )
    return sequence_dim, batch_dims, core_dims


def phase_roles_structure(core: Mapping[str, Any]) -> tuple[str | None, list[str], list[str]]:
    roles = roles_block(core)
    if roles is None:
        return None, [], []
    validate_role_keys(roles)
    return extract_role_fields(roles)


def check_sequence_dim_in_dataset(ds: xr.Dataset, sequence_dim: str | None) -> None:
    if sequence_dim is None:
        return
    if sequence_dim in ds.dims:
        return
    fail(
        code="schema.roles.sequence_dim.not_in_dataset",
        path="tal.core.roles.sequence_dim",
        expected="dim in ds.dims",
        actual={"dim": sequence_dim, "ds_dims": list(ds.dims)},
        hint="set sequence_dim to one of ds.dims",
    )


def check_duplicate_dims(path: str, code: str, dims: list[str], hint: str) -> None:
    duplicate = first_duplicate(dims)
    if duplicate is None:
        return
    fail(
        code=code,
        path=path,
        expected="duplicate-free list",
        actual={"duplicate": duplicate, "dims": dims},
        hint=hint,
    )


def check_dims_exist(path: str, code: str, dims: list[str], ds: xr.Dataset, hint: str) -> None:
    missing = [dim for dim in dims if dim not in ds.dims]
    if not missing:
        return
    fail(
        code=code,
        path=path,
        expected="all dims in ds.dims",
        actual={"missing": missing, "ds_dims": list(ds.dims)},
        hint=hint,
    )


def check_roles_overlap(sequence_dim: str | None, batch_dims: list[str], core_dims: list[str]) -> None:
    overlap: set[str] = set()
    if sequence_dim is not None:
        overlap |= {sequence_dim} & set(batch_dims)
        overlap |= {sequence_dim} & set(core_dims)
    overlap |= set(batch_dims) & set(core_dims)
    if not overlap:
        return
    fail(
        code="schema.roles.overlap",
        path="tal.core.roles",
        expected="sequence_dim, batch_dims, core_dims are disjoint",
        actual={"overlap": sorted(overlap)},
        hint="choose non-overlapping role dimensions",
    )


def phase_roles_vs_dims(
    ds: xr.Dataset,
    *,
    sequence_dim: str | None,
    batch_dims: list[str],
    core_dims: list[str],
) -> None:
    check_sequence_dim_in_dataset(ds, sequence_dim)
    check_duplicate_dims(
        "tal.core.roles.batch_dims",
        "schema.roles.batch_dims.duplicate",
        batch_dims,
        "remove duplicate entries from batch_dims",
    )
    check_dims_exist(
        "tal.core.roles.batch_dims",
        "schema.roles.batch_dims.not_in_dataset",
        batch_dims,
        ds,
        "use only existing dataset dims in batch_dims",
    )
    check_duplicate_dims(
        "tal.core.roles.core_dims",
        "schema.roles.core_dims.duplicate",
        core_dims,
        "remove duplicate entries from core_dims",
    )
    check_dims_exist(
        "tal.core.roles.core_dims",
        "schema.roles.core_dims.not_in_dataset",
        core_dims,
        ds,
        "use only existing dataset dims in core_dims",
    )
    check_roles_overlap(sequence_dim, batch_dims, core_dims)
