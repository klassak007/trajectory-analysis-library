from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import xarray as xr


def _copy_schema_value(value: Any, memo: dict[int, Any]) -> Any:
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, Mapping):
        out: dict[Any, Any] = {}
        memo[id(value)] = out
        for key, item in value.items():
            out[deepcopy(key, memo)] = _copy_schema_value(item, memo)
        return out
    if type(value) is list:
        items: list[Any] = []
        memo[id(value)] = items
        items.extend(_copy_schema_value(item, memo) for item in value)
        return items
    if type(value) is tuple:
        items = tuple(_copy_schema_value(item, memo) for item in value)
        memo[id(value)] = items
        return items
    return deepcopy(value, memo)


def _copy_tal_graph(tal: Mapping[Any, Any]) -> dict[Any, Any]:
    """Own canonical and extension regions without cross-region aliases."""
    out: dict[Any, Any] = {}
    root_memo: dict[int, Any] = {id(tal): out}
    for key, value in tal.items():
        copied_key = deepcopy(key, root_memo)
        region = str.__str__(key) if isinstance(key, str) else None
        memo = {} if region in {"core", "ext"} else root_memo
        out[copied_key] = _copy_schema_value(value, memo)
    return out


def _attrs_without_tal(attrs: Mapping[Any, Any]) -> dict[Any, Any]:
    return {name: value for name, value in attrs.items() if name != "tal"}


def _replace_dataset_attrs_with_tal(
    ds: xr.Dataset,
    *,
    ordinary_attrs: Mapping[Any, Any],
    tal: Mapping[str, Any] | None,
    canonicalize: bool = False,
    isolate_tal: bool = True,
    variable_without_tal: str | None = None,
) -> xr.Dataset:
    """Replace dataset attrs through the single TAL-attribute write owner."""
    attrs = _attrs_without_tal(ordinary_attrs)
    if tal is not None:
        tal_value = tal
        if isolate_tal:
            tal_value = canonicalize_tal(tal) if canonicalize else _copy_tal_graph(tal)
        attrs["tal"] = tal_value
    out = ds.copy(deep=False)
    out.attrs = attrs
    if variable_without_tal is not None:
        out[variable_without_tal].attrs = _attrs_without_tal(
            out[variable_without_tal].attrs
        )
    return out


def _relocate_promoted_dataarray_tal(
    ds: xr.Dataset,
    *,
    variable_name: str,
    tal: Mapping[str, Any],
) -> xr.Dataset:
    """Move a promoted DataArray TAL payload before ingress validation."""
    return _replace_dataset_attrs_with_tal(
        ds,
        ordinary_attrs=ds.attrs,
        tal=tal,
        isolate_tal=False,
        variable_without_tal=variable_name,
    )


def canonicalize_tal(tal: Mapping[str, Any]) -> dict[str, Any]:
    out = _copy_tal_graph(tal)
    out["version"] = int(out["version"])
    core = out.get("core")
    if not isinstance(core, Mapping):
        return out
    if "roles" in core and isinstance(core["roles"], Mapping):
        roles = core["roles"]
        sequence_dim = roles.get("sequence_dim")
        if sequence_dim is not None:
            roles["sequence_dim"] = str(sequence_dim)
        roles["batch_dims"] = [str(item) for item in roles.get("batch_dims", [])]
        roles["core_dims"] = [str(item) for item in roles.get("core_dims", [])]
    if "param_coord" in core and isinstance(core["param_coord"], Mapping):
        core["param_coord"]["name"] = str(core["param_coord"]["name"])
    if "validity" in core and isinstance(core["validity"], Mapping):
        validity = core["validity"]
        validity["sequence_size_coord"] = str(validity["sequence_size_coord"])
        validity["layout"] = str(validity["layout"])
    return out


def finalize_validated_schema(ds: xr.Dataset, tal: Mapping[str, Any]) -> xr.Dataset:
    return _replace_dataset_attrs_with_tal(
        ds,
        ordinary_attrs=ds.attrs,
        tal=tal,
        canonicalize=True,
    )
