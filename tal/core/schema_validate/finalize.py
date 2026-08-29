from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import xarray as xr


def _replace_dataset_attrs_with_tal(
    ds: xr.Dataset,
    *,
    ordinary_attrs: Mapping[Any, Any],
    tal: Mapping[str, Any] | None,
    canonicalize: bool = False,
) -> xr.Dataset:
    """Replace dataset attrs while isolating the optional TAL payload."""
    attrs = {name: value for name, value in ordinary_attrs.items() if name != "tal"}
    if tal is not None:
        attrs["tal"] = canonicalize_tal(tal) if canonicalize else deepcopy(dict(tal))
    out = ds.copy(deep=False)
    out.attrs = attrs
    return out


def canonicalize_tal(tal: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(tal))
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
