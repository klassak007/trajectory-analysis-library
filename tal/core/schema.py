from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal

import xarray as xr

from .schema_errors import schema_error
from .schema_validate.common import ALLOWED_LAYOUTS
from .schema_validate import SCHEMA_VERSION
from .schema_validate import validate_schema as _validate_schema


class UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET = UnsetType()


def _fail_patch(code: str, actual: Any, hint: str) -> None:
    raise schema_error(
        code=code,
        path="tal",
        expected="patch mapping rooted at tal payload",
        actual=actual,
        hint=hint,
    )


def _require_dataset(ds: Any, *, owner: str) -> xr.Dataset:
    if isinstance(ds, xr.Dataset):
        return ds
    raise TypeError(f"{owner} expects xr.Dataset, got {type(ds).__name__}.")


def _copy_tal(ds: xr.Dataset) -> dict[str, Any]:
    if "tal" not in ds.attrs:
        return {}
    tal = ds.attrs["tal"]
    if not isinstance(tal, Mapping):
        raise schema_error(
            code="schema.not_mapping",
            path="tal",
            expected="mapping",
            actual=type(tal).__name__,
            hint="set ds.attrs['tal'] to a mapping payload",
        )
    return deepcopy(dict(tal))


def _with_schema(ds: xr.Dataset, tal_schema: Mapping[str, Any]) -> xr.Dataset:
    out = ds.copy(deep=False)
    attrs = dict(out.attrs)
    attrs["tal"] = deepcopy(dict(tal_schema))
    out.attrs = attrs
    return out


def _apply_writer(
    ds: xr.Dataset,
    tal_schema: Mapping[str, Any],
    *,
    validate: bool,
) -> xr.Dataset:
    candidate = _with_schema(ds, tal_schema)
    if not validate:
        return candidate
    return _validate_schema(candidate)


def copy_dataset_attrs(
    source: xr.Dataset,
    target: xr.Dataset,
    *,
    validate: bool = True,
) -> xr.Dataset:
    """Copy dataset attrs while routing the TAL schema through its writer.

    Parameters
    ----------
    source : xr.Dataset
        Dataset whose complete attrs mapping is copied.
    target : xr.Dataset
        Dataset that receives the copied attrs without payload duplication.
    validate : bool, optional
        Whether to validate the resulting TAL schema.

    Returns
    -------
    xr.Dataset
        A shallow target copy carrying the source's ordinary attrs and an
        isolated copy of its TAL schema.

    Raises
    ------
    TypeError
        If either input is not an ``xarray.Dataset``.
    ValueError
        If the source TAL payload or resulting schema is invalid. TAL raises
        its ``tal.core.SchemaError`` subtype for these failures.

    Notes
    -----
    This owner is intended for kernels that construct a fresh dataset while
    preserving source metadata. It replaces the target attrs rather than
    merging ordinary attrs. The input datasets are not mutated.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import copy_dataset_attrs, set_roles
    >>> source = xr.Dataset(
    ...     {"value": ("sample", [1.0])},
    ...     attrs={"note": "source"},
    ... )
    >>> source = set_roles(source, sequence_dim="sample", core_dims=())
    >>> target = xr.Dataset({"result": ("sample", [2.0])})
    >>> out = copy_dataset_attrs(source, target)
    >>> (out.attrs["note"], out.attrs["tal"]["core"]["roles"]["sequence_dim"])
    ('source', 'sample')
    """
    source_ds = _require_dataset(source, owner="copy_dataset_attrs")
    target_ds = _require_dataset(target, owner="copy_dataset_attrs")
    has_schema = "tal" in source_ds.attrs
    tal_schema = _copy_tal(source_ds) if has_schema else None
    out = target_ds.copy(deep=False)
    out.attrs = {
        name: value for name, value in source_ds.attrs.items() if name != "tal"
    }
    if tal_schema is not None:
        return _apply_writer(out, tal_schema, validate=validate)
    if validate:
        return _validate_schema(out)
    return out


def _is_bootstrap_schema(tal_schema: Mapping[str, Any]) -> bool:
    core = tal_schema.get("core")
    if not isinstance(core, Mapping) or core:
        return False
    allowed = {"version", "core", "ext"}
    if any(key not in allowed for key in tal_schema):
        return False
    if tal_schema.get("version") != SCHEMA_VERSION:
        return False
    if "ext" not in tal_schema:
        return True
    ext = tal_schema["ext"]
    if not isinstance(ext, Mapping):
        return False
    return all(isinstance(key, str) and bool(key) for key in ext)


def _apply_structural_writer(
    ds: xr.Dataset,
    tal_schema: Mapping[str, Any],
    *,
    validate: bool,
    allow_bootstrap_without_validate: bool,
) -> xr.Dataset:
    candidate = _with_schema(ds, tal_schema)
    if not validate:
        return candidate
    if allow_bootstrap_without_validate and _is_bootstrap_schema(tal_schema):
        return candidate
    return _validate_schema(candidate)


def _merge_dict(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(base))
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
            continue
        current = out.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            out[key] = _merge_dict(current, value)
            continue
        if isinstance(value, Mapping):
            out[key] = _merge_dict({}, value)
            continue
        out[key] = deepcopy(value)
    return out


def _existing_roles(tal: Mapping[str, Any]) -> dict[str, Any]:
    core = tal.get("core")
    if not isinstance(core, Mapping):
        return {"batch_dims": [], "core_dims": []}
    roles = core.get("roles")
    if not isinstance(roles, Mapping):
        return {"batch_dims": [], "core_dims": []}
    out = deepcopy(dict(roles))
    out.setdefault("batch_dims", [])
    out.setdefault("core_dims", [])
    return out


def _canon_dim_list(value: Sequence[str] | UnsetType) -> Any:
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return value


def set_roles(
    ds: xr.Dataset,
    *,
    sequence_dim: str | UnsetType = UNSET,
    batch_dims: Sequence[str] | UnsetType = UNSET,
    core_dims: Sequence[str] | UnsetType = UNSET,
    validate: bool = True,
) -> xr.Dataset:
    """Set TAL role metadata on a dataset payload.

    Parameters
    ----------
    ds
        Input dataset.
    sequence_dim
        Sequence dimension or ``UNSET`` to preserve current value.
    batch_dims
        Batch dimensions or ``UNSET`` to preserve current values.
    core_dims
        Core dimensions or ``UNSET`` to preserve current values.
    validate
        Whether to validate schema before returning.

    Returns
    -------
    xarray.Dataset
        Dataset with updated TAL role metadata.

    Notes
    -----
    ``batch_dims`` and ``core_dims`` may be written without ``sequence_dim``.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core.schema import set_roles
    >>> ds = xr.Dataset(
    ...     {"value": (("trial", "axis"), [[1.0, 2.0, 3.0]])},
    ...     coords={"trial": ["t0"], "axis": ["x", "y", "z"]},
    ... )
    >>> updated = set_roles(ds, batch_dims=("trial",), core_dims=("axis",), validate=False)
    >>> updated.attrs["tal"]["core"]["roles"].get("sequence_dim") is None
    True
    """
    ds = _require_dataset(ds, owner="set_roles")
    tal = _copy_tal(ds)
    roles = _existing_roles(tal)
    if sequence_dim is not UNSET:
        roles["sequence_dim"] = sequence_dim
    if batch_dims is not UNSET:
        roles["batch_dims"] = _canon_dim_list(batch_dims)
    if core_dims is not UNSET:
        roles["core_dims"] = _canon_dim_list(core_dims)
    patch = {"version": SCHEMA_VERSION, "core": {"roles": roles}}
    tal = _merge_dict(tal, patch)
    return _apply_writer(ds, tal, validate=validate)


def set_param_coord(
    ds: xr.Dataset,
    *,
    name: str | None,
    validate: bool = True,
) -> xr.Dataset:
    """Set or clear ``param_coord`` metadata on a dataset.

    Parameters
    ----------
    ds
        Dataset to update.
    name
        Coordinate name to record as the parameter coordinate. Pass ``None`` to
        clear the block.
    validate
        When ``True`` (default), validate the resulting TAL schema before
        returning.

    Returns
    -------
    xr.Dataset
        Dataset with updated ``tal.core.param_coord`` metadata.

    Notes
    -----
    ``param_coord`` is sequence-scoped metadata. Validation fails closed if a
    parameter coordinate is configured while ``tal.core.roles.sequence_dim`` is
    absent.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core.schema import set_param_coord
    >>> ds = xr.Dataset(coords={"sample": [0, 1], "time": ("sample", [0.0, 1.0])})
    >>> updated = set_param_coord(ds, name="time", validate=False)
    >>> updated.attrs["tal"]["core"]["param_coord"]["name"]
    'time'
    """
    ds = _require_dataset(ds, owner="set_param_coord")
    block: dict[str, Any] | None
    block = None if name is None else {"name": str(name)}
    patch = {"version": SCHEMA_VERSION, "core": {"param_coord": block}}
    tal = _merge_dict(_copy_tal(ds), patch)
    return _apply_writer(ds, tal, validate=validate)


def set_validity(
    ds: xr.Dataset,
    *,
    sequence_size_coord: str | None,
    layout: Literal["left_packed"] = "left_packed",
    validate: bool = True,
) -> xr.Dataset:
    """Set or clear sequence-validity metadata on a dataset.

    Parameters
    ----------
    ds
        Dataset to update.
    sequence_size_coord
        Coordinate name that stores per-batch valid lengths. Pass ``None`` to
        clear validity metadata.
    layout
        Validity layout policy. The canonical runtime layout is
        ``"left_packed"``.
    validate
        When ``True`` (default), validate the resulting TAL schema before
        returning.

    Returns
    -------
    xr.Dataset
        Dataset with updated ``tal.core.validity`` metadata.

    Notes
    -----
    Validity metadata is sequence-scoped. Validation fails closed when a
    validity block is declared without ``tal.core.roles.sequence_dim``.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core.schema import set_validity
    >>> ds = xr.Dataset(coords={"sample": [0, 1], "group_size": ("trial", [2])})
    >>> updated = set_validity(ds, sequence_size_coord="group_size", validate=False)
    >>> updated.attrs["tal"]["core"]["validity"]["sequence_size_coord"]
    'group_size'
    """
    ds = _require_dataset(ds, owner="set_validity")
    block: dict[str, Any] | None
    if sequence_size_coord is None:
        block = None
    else:
        block = {"sequence_size_coord": str(sequence_size_coord), "layout": str(layout)}
    patch = {"version": SCHEMA_VERSION, "core": {"validity": block}}
    tal = _merge_dict(_copy_tal(ds), patch)
    return _apply_writer(ds, tal, validate=validate)


def merge_schema(
    ds: xr.Dataset,
    patch: Mapping[str, Any],
    *,
    validate: bool = True,
) -> xr.Dataset:
    """Merge a TAL schema patch into ``ds.attrs['tal']``.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    patch : Mapping[str, Any]
        Operand/component input consumed by this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    xr.Dataset
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core.schema import merge_schema
    >>> ds = xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]})
    >>> out = merge_schema(ds, {"version": 1, "core": {"roles": {"sequence_dim": "sample", "batch_dims": [], "core_dims": []}}})
    >>> out.attrs["tal"]["core"]["roles"]["sequence_dim"]
    'sample'
    """
    ds = _require_dataset(ds, owner="merge_schema")
    if not isinstance(patch, Mapping):
        _fail_patch("schema.patch.type.invalid", type(patch).__name__, "pass a mapping")
    if "tal" in patch:
        _fail_patch(
            "schema.patch.root.invalid",
            {"keys": list(patch)},
            "remove top-level 'tal' wrapper from patch",
        )
    tal = _merge_dict(_copy_tal(ds), patch)
    return _apply_writer(ds, tal, validate=validate)


def _normalize_rename_map(rename_map: Mapping[str, str] | None) -> dict[str, str]:
    if not isinstance(rename_map, Mapping):
        return {}
    out: dict[str, str] = {}
    for old, new in rename_map.items():
        if isinstance(old, str) and isinstance(new, str):
            out[old] = new
    return out


def _remap_name(name: Any, rename_map: Mapping[str, str]) -> str | None:
    if not isinstance(name, str):
        return None
    return rename_map.get(name, name)


def _remap_name_list(names: Any, rename_map: Mapping[str, str]) -> list[str]:
    if not isinstance(names, list):
        return []
    out: list[str] = []
    for item in names:
        remapped = _remap_name(item, rename_map)
        if remapped is not None:
            out.append(remapped)
    return out


def _roles_after_structure(
    roles: Mapping[str, Any],
    *,
    ds: xr.Dataset,
    rename_map: Mapping[str, str],
) -> dict[str, Any]:
    def _stable_unique(values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    sequence_dim = _remap_name(roles.get("sequence_dim"), rename_map)
    if sequence_dim is not None and sequence_dim not in ds.dims:
        sequence_dim = None
    batch_dims = [dim for dim in _remap_name_list(roles.get("batch_dims"), rename_map) if dim in ds.dims]
    core_dims = [dim for dim in _remap_name_list(roles.get("core_dims"), rename_map) if dim in ds.dims]
    if sequence_dim is not None:
        batch_dims = [dim for dim in batch_dims if dim != sequence_dim]
        core_dims = [dim for dim in core_dims if dim != sequence_dim]
    batch_dims = _stable_unique(batch_dims)
    core_dims = [dim for dim in _stable_unique(core_dims) if dim not in set(batch_dims)]
    return {
        # Keep explicit None so merge_schema removes stale sequence_dim keys.
        "sequence_dim": sequence_dim,
        "batch_dims": batch_dims,
        "core_dims": core_dims,
    }


def _param_after_structure(
    param: Mapping[str, Any],
    *,
    ds: xr.Dataset,
    sequence_dim: str | None,
    batch_dims: list[str],
    rename_map: Mapping[str, str],
) -> dict[str, Any] | None:
    if sequence_dim is None:
        return None
    name = _remap_name(param.get("name"), rename_map)
    if name is None or name not in ds.coords:
        return None
    dims = tuple(ds.coords[name].dims)
    allowed = [(sequence_dim,), tuple(batch_dims) + (sequence_dim,)]
    if dims not in allowed:
        return None
    return {"name": name}


def _validity_after_structure(
    validity: Mapping[str, Any],
    *,
    ds: xr.Dataset,
    sequence_dim: str | None,
    batch_dims: list[str],
    rename_map: Mapping[str, str],
) -> dict[str, Any] | None:
    if sequence_dim is None:
        return None
    coord_name = _remap_name(validity.get("sequence_size_coord"), rename_map)
    layout = validity.get("layout")
    if coord_name is None or coord_name not in ds.coords or layout not in ALLOWED_LAYOUTS:
        return None
    dims = tuple(ds.coords[coord_name].dims)
    if batch_dims and dims != tuple(batch_dims):
        return None
    if not batch_dims and dims != ():
        return None
    return {"sequence_size_coord": coord_name, "layout": str(layout)}


def _write_structural(
    ds: xr.Dataset,
    tal_schema: Mapping[str, Any],
    *,
    validate: bool,
    allow_bootstrap: bool = False,
) -> xr.Dataset:
    return _apply_structural_writer(
        ds,
        tal_schema,
        validate=validate,
        allow_bootstrap_without_validate=allow_bootstrap,
    )


def _optional_blocks_patch(
    core: Mapping[str, Any],
    *,
    ds: xr.Dataset,
    repaired_roles: Mapping[str, Any],
    rename_map: Mapping[str, str],
) -> dict[str, Any]:
    patch: dict[str, Any] = {"roles": dict(repaired_roles)}
    if "param_coord" in core:
        param = core["param_coord"]
        patch["param_coord"] = _param_after_structure(
            param if isinstance(param, Mapping) else {},
            ds=ds,
            sequence_dim=repaired_roles.get("sequence_dim"),
            batch_dims=repaired_roles["batch_dims"],
            rename_map=rename_map,
        )
    if "validity" in core:
        validity = core["validity"]
        patch["validity"] = _validity_after_structure(
            validity if isinstance(validity, Mapping) else {},
            ds=ds,
            sequence_dim=repaired_roles.get("sequence_dim"),
            batch_dims=repaired_roles["batch_dims"],
            rename_map=rename_map,
        )
    return patch


def repair_schema_after_structure(
    ds: xr.Dataset,
    *,
    validate: bool = True,
    rename_map: Mapping[str, str] | None = None,
) -> xr.Dataset:
    tal = _merge_dict(_copy_tal(ds), {"version": SCHEMA_VERSION, "core": {}})
    core = tal.get("core")
    if not isinstance(core, Mapping):
        return _write_structural(ds, tal, validate=validate)
    roles = core.get("roles")
    if not isinstance(roles, Mapping):
        return _write_structural(ds, tal, validate=validate)
    name_map = _normalize_rename_map(rename_map)
    repaired_roles = _roles_after_structure(roles, ds=ds, rename_map=name_map)
    if repaired_roles is None:
        tal = _merge_dict(tal, {"core": {"roles": None, "param_coord": None, "validity": None}})
        return _write_structural(ds, tal, validate=validate, allow_bootstrap=True)
    patch = _optional_blocks_patch(
        core,
        ds=ds,
        repaired_roles=repaired_roles,
        rename_map=name_map,
    )
    tal = _merge_dict(tal, {"core": patch})
    return _write_structural(ds, tal, validate=validate)


def validate_schema(ds: xr.Dataset) -> xr.Dataset:
    """Validate TAL schema payload and return validated dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.

    Returns
    -------
    xr.Dataset
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core.schema import set_roles, validate_schema
    >>> ds = xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]})
    >>> tagged = set_roles(ds, sequence_dim="sample", core_dims=(), validate=True)
    >>> validate_schema(tagged).attrs["tal"]["version"]
    1
    """
    return _validate_schema(ds)
