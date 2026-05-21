from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from .schema import _is_bootstrap_schema
from .schema_errors import schema_error
from .schema_validate import validate_schema as _validate_schema
from .schema_validate.common import sorted_mapping_keys, unknown_key_actual, unknown_key_path

_ALLOWED_ROLE_KEYS = {"sequence_dim", "batch_dims", "core_dims"}
_ALLOWED_PARAM_KEYS = {"name"}
_ALLOWED_VALIDITY_KEYS = {"sequence_size_coord", "layout"}


def _schema_fail(*, code: str, path: str, expected: Any, actual: Any, hint: str) -> None:
    raise schema_error(code=code, path=path, expected=expected, actual=actual, hint=hint)


def _tal_payload(ds: xr.Dataset) -> tuple[bool, Any]:
    if "tal" not in ds.attrs:
        return False, None
    return True, ds.attrs["tal"]


def validate_schema_if_needed(ds: xr.Dataset) -> xr.Dataset:
    had_tal, payload = _tal_payload(ds)
    if not had_tal:
        return ds
    if isinstance(payload, Mapping) and _is_bootstrap_schema(payload):
        return ds
    return _validate_schema(ds)


def _core_mapping(ds: xr.Dataset) -> Mapping[str, Any] | None:
    had_tal, payload = _tal_payload(ds)
    if not had_tal or not isinstance(payload, Mapping):
        return None
    core = payload.get("core")
    if core is None:
        return None
    if isinstance(core, Mapping):
        return core
    _schema_fail(
        code="schema.core.not_mapping",
        path="tal.core",
        expected="mapping",
        actual=type(core).__name__,
        hint="set tal.core to a mapping",
    )


def _is_name_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and bool(item) for item in value)


def _check_unknown_keys(
    block: Mapping[str, Any],
    *,
    path: str,
    code: str,
    allowed: set[str],
    hint: str,
) -> None:
    for key in sorted_mapping_keys(block):
        if key in allowed:
            continue
        _schema_fail(
            code=code,
            path=unknown_key_path(path, key),
            expected=sorted(allowed),
            actual=unknown_key_actual(key),
            hint=hint,
        )


def _extract_roles_fields(roles: Mapping[str, Any]) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    sequence_dim = roles.get("sequence_dim")
    if sequence_dim is not None and (not isinstance(sequence_dim, str) or not sequence_dim):
        _schema_fail(
            code="schema.roles.sequence_dim.invalid",
            path="tal.core.roles.sequence_dim",
            expected="non-empty string",
            actual=sequence_dim,
            hint="set sequence_dim to a dataset dimension name",
        )
    batch_dims = roles.get("batch_dims", [])
    if not _is_name_list(batch_dims):
        _schema_fail(
            code="schema.roles.batch_dims.invalid",
            path="tal.core.roles.batch_dims",
            expected="list[str]",
            actual=batch_dims,
            hint="set batch_dims to a list of dataset dimension names",
        )
    core_dims = roles.get("core_dims")
    if not _is_name_list(core_dims):
        _schema_fail(
            code="schema.roles.core_dims.invalid",
            path="tal.core.roles.core_dims",
            expected="list[str]",
            actual=core_dims,
            hint="set core_dims to a list of dataset dimension names",
        )
    return str(sequence_dim) if sequence_dim is not None else None, tuple(batch_dims), tuple(core_dims)


def read_roles(
    ds: xr.Dataset,
) -> tuple[bool, str | None, tuple[str, ...], tuple[str, ...]]:
    """Read declared TAL role metadata from a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose ``ds.attrs["tal"]`` payload should be inspected.

    Returns
    -------
    tuple[bool, str | None, tuple[str, ...], tuple[str, ...]]
        ``(declared, sequence_dim, batch_dims, core_dims)``. When roles are not
        declared, returns ``(False, None, (), ())``.

    Raises
    ------
    ValueError
        If a present roles block is malformed or contains unknown keys.

    Notes
    -----
    This reader does not mutate the dataset. It validates only the roles block
    shape needed for safe metadata inspection.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.schema_read import read_roles
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> read_roles(ao.unsafe_data)
    (True, 'sample', (), ())
    """
    core = _core_mapping(ds)
    if core is None:
        return False, None, (), ()
    if "roles" not in core:
        return False, None, (), ()
    roles = core["roles"]
    if not isinstance(roles, Mapping):
        _schema_fail(
            code="schema.roles.not_mapping",
            path="tal.core.roles",
            expected="mapping",
            actual=type(roles).__name__,
            hint="set tal.core.roles with sequence_dim, batch_dims, core_dims",
        )
    _check_unknown_keys(
        roles,
        path="tal.core.roles",
        code="schema.roles.unknown_key",
        allowed=_ALLOWED_ROLE_KEYS,
        hint="remove unknown roles key",
    )
    sequence_dim, batch_dims, core_dims = _extract_roles_fields(roles)
    return True, sequence_dim, batch_dims, core_dims


def read_param_coord_name(ds: xr.Dataset) -> str | None:
    core = _core_mapping(ds)
    if core is None:
        return None
    if "param_coord" not in core:
        return None
    block = core["param_coord"]
    if not isinstance(block, Mapping):
        _schema_fail(
            code="schema.param_coord.not_mapping",
            path="tal.core.param_coord",
            expected="mapping",
            actual=type(block).__name__,
            hint="set tal.core.param_coord = {'name': '<coord>'}",
        )
    _check_unknown_keys(
        block,
        path="tal.core.param_coord",
        code="schema.param_coord.unknown_key",
        allowed=_ALLOWED_PARAM_KEYS,
        hint="remove unknown param_coord keys",
    )
    if "name" not in block:
        _schema_fail(
            code="schema.param_coord.name.missing",
            path="tal.core.param_coord.name",
            expected="non-empty string",
            actual=None,
            hint="set param_coord.name to a dataset coordinate",
        )
    name = block["name"]
    if not isinstance(name, str) or not name:
        _schema_fail(
            code="schema.param_coord.name.invalid",
            path="tal.core.param_coord.name",
            expected="non-empty string",
            actual=name,
            hint="set param_coord.name to a dataset coordinate",
        )
    return name


def read_sequence_size_coord_name(ds: xr.Dataset) -> str | None:
    core = _core_mapping(ds)
    if core is None:
        return None
    if "validity" not in core:
        return None
    block = core["validity"]
    if not isinstance(block, Mapping):
        _schema_fail(
            code="schema.validity.not_mapping",
            path="tal.core.validity",
            expected="mapping",
            actual=type(block).__name__,
            hint="set tal.core.validity with sequence_size_coord and layout",
        )
    _check_unknown_keys(
        block,
        path="tal.core.validity",
        code="schema.validity.unknown_key",
        allowed=_ALLOWED_VALIDITY_KEYS,
        hint="remove unknown validity keys",
    )
    if "sequence_size_coord" not in block:
        _schema_fail(
            code="schema.validity.sequence_size_coord.missing",
            path="tal.core.validity.sequence_size_coord",
            expected="non-empty string",
            actual=None,
            hint="set sequence_size_coord to a coordinate name",
        )
    name = block["sequence_size_coord"]
    if not isinstance(name, str) or not name:
        _schema_fail(
            code="schema.validity.sequence_size_coord.invalid",
            path="tal.core.validity.sequence_size_coord",
            expected="non-empty string",
            actual=name,
            hint="set sequence_size_coord to a coordinate name",
        )
    layout = block.get("layout")
    if layout != "left_packed":
        _schema_fail(
            code="schema.validity.layout.invalid",
            path="tal.core.validity.layout",
            expected=["left_packed"],
            actual=layout,
            hint="set layout='left_packed'",
        )
    return name


def read_param_coord(ds: xr.Dataset) -> tuple[bool, str | None]:
    name = read_param_coord_name(ds)
    if name is None:
        return False, None
    return True, name


def read_validity(ds: xr.Dataset) -> tuple[bool, str | None, str | None]:
    name = read_sequence_size_coord_name(ds)
    if name is None:
        return False, None, None
    return True, name, "left_packed"


__all__ = [
    "read_param_coord",
    "read_param_coord_name",
    "read_roles",
    "read_validity",
    "read_sequence_size_coord_name",
    "validate_schema_if_needed",
]
