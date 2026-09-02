from __future__ import annotations

from collections.abc import Mapping

import xarray as xr

from ..analysis_object import AnalysisObject
from ..dataset_ownership import analysis_object_dataset
from ..orchestration.finalize import finalize_like
from ..schema import merge_schema as _merge_schema
from ..schema import set_roles
from ..schema_read import read_roles
from ..var_naming import default_datavar_name, is_valid_user_var_name, preserve_or_datavar


def _coerce_result_dataset(result: object, *, source: AnalysisObject, owner: str) -> xr.Dataset:
    if isinstance(result, xr.Dataset):
        return result
    if isinstance(result, xr.DataArray):
        if is_valid_user_var_name(result.name):
            return result.to_dataset(name=result.name)
        source_names = tuple(str(name) for name in analysis_object_dataset(source).data_vars)
        if len(source_names) == 1:
            return result.to_dataset(name=preserve_or_datavar(source_names[0]))
        return result.to_dataset(name=default_datavar_name())
    raise TypeError(
        f"{owner}: AO ufunc expected xarray Dataset/DataArray kernel output; got {type(result).__name__}."
    )


def _attach_source_schema(result: xr.Dataset, *, source: AnalysisObject) -> xr.Dataset:
    tal_payload = analysis_object_dataset(source).attrs.get("tal")
    if not isinstance(tal_payload, Mapping):
        return result
    return _merge_schema(result, patch=dict(tal_payload), validate=False)


def _attach_output_core_dims(
    result: xr.Dataset,
    *,
    source: AnalysisObject,
    output_core_dims: tuple[str, ...] | None,
) -> xr.Dataset:
    if output_core_dims is None:
        return result
    declared, sequence_dim, batch_dims, _ = read_roles(analysis_object_dataset(source))
    roles_kwargs: dict[str, object]
    if declared:
        roles_kwargs = {
            "batch_dims": batch_dims,
            "core_dims": output_core_dims,
        }
        if sequence_dim is not None:
            roles_kwargs["sequence_dim"] = sequence_dim
    else:
        # Undeclared-role inputs run with core-only runtime semantics.
        roles_kwargs = {"batch_dims": (), "core_dims": output_core_dims}
    return set_roles(
        result,
        validate=False,
        **roles_kwargs,
    )


def _finalize_ao_output(
    source: AnalysisObject,
    result: object,
    *,
    owner: str,
    validate: bool,
    output_core_dims: tuple[str, ...] | None = None,
) -> AnalysisObject:
    ds = _coerce_result_dataset(result, source=source, owner=owner)
    ds = _attach_source_schema(ds, source=source)
    ds = _attach_output_core_dims(ds, source=source, output_core_dims=output_core_dims)
    return finalize_like(source, ds, validate=validate, owner=owner)


def finalize_unary_ao_result(
    source: AnalysisObject,
    result: object,
    *,
    owner: str,
    validate: bool = True,
) -> AnalysisObject:
    return _finalize_ao_output(source, result, owner=owner, validate=validate)


def finalize_binary_ao_result(
    source: AnalysisObject,
    result: object,
    *,
    owner: str,
    validate: bool = True,
    output_core_dims: tuple[str, ...] | None = None,
) -> AnalysisObject:
    return _finalize_ao_output(
        source,
        result,
        owner=owner,
        validate=validate,
        output_core_dims=output_core_dims,
    )


__all__ = [
    "finalize_binary_ao_result",
    "finalize_unary_ao_result",
]
