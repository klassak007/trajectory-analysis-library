from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from ..orchestration.context import DatasetContext, DatasetContextOptions, resolve_dataset_contexts
from ..orchestration.inputs import coerce_analysis_object_input
from ..orchestration.lazy import is_chunked_dataarray, require_unchunked_dataarray
from ..validity_mask import resolve_validated_structural_mask_base
from .key_resolve import normalize_grouping_key_input, resolve_grouping_key
from .options import coerce_grouping_foundation_options
from .types import GroupingFoundationContext, GroupingFoundationOptions, GroupingKeyInput, ResolvedGroupingKey

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject

_DEFAULT_NA_GROUP_LABEL = "__tal_na_group__"
_NA_POLICY_CHUNKED_GUIDANCE = (
    "chunked grouping NA-policy scalar checks are not supported; provide unchunked grouping keys."
)


def _resolve_grouping_dataset_context(ao: AnalysisObject, *, owner: str) -> DatasetContext:
    contexts = resolve_dataset_contexts(
        (ao,),
        owner=owner,
        options=DatasetContextOptions(require_roles=True, require_sequence_dim=True),
    )
    return contexts[0]


def _na_mask(data: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(pd.isna, data, dask="parallelized", output_dtypes=[bool])


def _combined_na_mask(keys: tuple[ResolvedGroupingKey, ...]) -> xr.DataArray:
    return reduce(lambda left, right: left | right, (_na_mask(key.data) for key in keys))


def _scope_to_structural_validity(
    mask: xr.DataArray,
    *,
    structural_valid_mask: xr.DataArray | None,
) -> xr.DataArray:
    return mask if structural_valid_mask is None else mask & structural_valid_mask


def _structural_valid_mask(context: DatasetContext) -> xr.DataArray | None:
    return resolve_validated_structural_mask_base(
        context.ds,
        sequence_dim=context.sequence_dim,
        sequence_size_coord=context.sequence_size_coord,
    )


def _fill_na_group_label(
    data: xr.DataArray,
    *,
    label: object,
    structural_valid_mask: xr.DataArray | None,
) -> xr.DataArray:
    mask = _scope_to_structural_validity(
        _na_mask(data),
        structural_valid_mask=structural_valid_mask,
    )
    return xr.where(mask, label, data.astype(object))


def _mask_has_true(mask: xr.DataArray, *, owner: str, field: str) -> bool:
    reduced = mask.any()
    require_unchunked_dataarray(
        reduced,
        owner=owner,
        field=field,
        guidance=_NA_POLICY_CHUNKED_GUIDANCE,
    )
    return bool(np.asarray(reduced.data).item())


def _active_na_policy_masks(
    combined_mask: xr.DataArray,
    *,
    context: DatasetContext,
    owner: str,
) -> tuple[xr.DataArray, xr.DataArray | None] | None:
    if not _mask_has_true(combined_mask, owner=owner, field="grouping key NA check"):
        return None
    structural = _structural_valid_mask(context)
    active = _scope_to_structural_validity(
        combined_mask,
        structural_valid_mask=structural,
    )
    if structural is not None and not _mask_has_true(
        active,
        owner=owner,
        field="active grouping key NA check",
    ):
        return None
    return active, structural


def _group_label_collision(
    key: ResolvedGroupingKey,
    *,
    label: object,
    structural_valid_mask: xr.DataArray | None,
    owner: str,
) -> None:
    mask = _na_mask(key.data)
    equal = xr.apply_ufunc(np.equal, key.data.astype(object), label, dask="parallelized", output_dtypes=[bool])
    collision = _scope_to_structural_validity(
        equal & (~mask),
        structural_valid_mask=structural_valid_mask,
    )
    if _mask_has_true(collision, owner=owner, field=f"grouping key[{key.index}] collision check"):
        raise ValueError(f"{owner}: na_group_label={label!r} collides with non-NA values in key {key.name!r}.")


def _apply_na_policy(
    keys: tuple[ResolvedGroupingKey, ...],
    *,
    context: DatasetContext,
    opts: GroupingFoundationOptions,
    owner: str,
) -> tuple[tuple[ResolvedGroupingKey, ...], xr.DataArray | None, object | None]:
    combined_mask = _combined_na_mask(keys)
    if opts.na_key_policy == "error":
        if is_chunked_dataarray(combined_mask):
            return keys, None, None
        if _active_na_policy_masks(combined_mask, context=context, owner=owner) is not None:
            raise ValueError(f"{owner}: grouping key domain contains NA values under na_key_policy='error'.")
        return keys, None, None
    if opts.na_key_policy == "drop":
        return keys, ~combined_mask, None
    label = opts.na_group_label if opts.na_group_label is not None else _DEFAULT_NA_GROUP_LABEL
    active_masks = _active_na_policy_masks(combined_mask, context=context, owner=owner)
    if active_masks is None:
        return keys, None, label
    _, structural_valid_mask = active_masks
    for key in keys:
        _group_label_collision(
            key,
            label=label,
            structural_valid_mask=structural_valid_mask,
            owner=owner,
        )
    grouped = tuple(
        ResolvedGroupingKey(
            index=key.index,
            kind=key.kind,
            name=key.name,
            data=_fill_na_group_label(
                key.data,
                label=label,
                structural_valid_mask=structural_valid_mask,
            ),
            domain_order=key.domain_order,
        )
        for key in keys
    )
    return grouped, None, label


def resolve_grouping_foundation_context(
    value: object,
    key: GroupingKeyInput,
    *,
    opts: GroupingFoundationOptions | None = None,
    owner: str = "grouping.foundation",
) -> GroupingFoundationContext:
    source = coerce_analysis_object_input(value, owner=owner)
    options = coerce_grouping_foundation_options(opts, owner=owner)
    context = _resolve_grouping_dataset_context(source, owner=owner)
    assert context.sequence_dim is not None
    normalized = normalize_grouping_key_input(key, owner=owner)
    resolved = tuple(
        resolve_grouping_key(context, item, index=index, owner=owner)
        for index, item in enumerate(normalized)
    )
    keys, na_exclusion_mask, na_group_label = _apply_na_policy(
        resolved,
        context=context,
        opts=options,
        owner=owner,
    )
    return GroupingFoundationContext(
        ao=context.ao,
        ds=context.ds,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        core_dims=context.core_dims,
        param_coord=context.param_coord,
        sequence_size_coord=context.sequence_size_coord,
        keys=keys,
        na_key_policy=options.na_key_policy,
        na_exclusion_mask=na_exclusion_mask,
        na_group_label=na_group_label,
    )


__all__ = ["resolve_grouping_foundation_context"]
