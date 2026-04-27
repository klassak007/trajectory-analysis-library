from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import xarray as xr


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
    out = ds.copy(deep=False)
    attrs = dict(out.attrs)
    attrs["tal"] = canonicalize_tal(tal)
    out.attrs = attrs
    return out
