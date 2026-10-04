from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ...utils.topology_operation_families import (
    operation_intent_support_for_operation_family,
)
from ..orchestration.alignment import align_exact_for_plan
from ..orchestration.alignment_intent import select_topology_policy_with_intents
from ..orchestration.context import DatasetContext
from ..orchestration.indexing import require_exact_lane_indexes
from ..orchestration.topology import (
    STRICT_EXACT_POLICY,
    SemanticTopology,
    TopologyOperand,
    resolve_binary_topology,
)
from .label_plan import object_label_vector
from .options import validate_grouping_bin_spec
from .row_dim_compat import require_row_dim_compatibility
from .types import (
    GroupingBinSpec,
    GroupingKeyInput,
    GroupingSingleKey,
    ResolvedGroupingKey,
    ResolvedGroupingKeyKind,
)

_GROUPING_OPERATION_FAMILY = "core.grouping.foundation"


def _require_single_key(key: object, *, owner: str) -> GroupingSingleKey:
    if isinstance(key, (str, xr.DataArray, GroupingBinSpec)):
        return key
    raise TypeError(f"{owner}: key must be str, xr.DataArray, GroupingBinSpec, or tuple/list of those.")


def _require_flat_key_item(
    item: object,
    *,
    index: int,
    owner: str,
) -> GroupingSingleKey:
    if isinstance(item, (tuple, list)):
        raise TypeError(
            f"{owner}: nested key containers are not supported; key[{index}] is nested."
        )
    return _require_single_key(item, owner=owner)


def _normalize_key_items(
    key: tuple[object, ...] | list[object],
    *,
    owner: str,
) -> tuple[GroupingSingleKey, ...]:
    return tuple(
        _require_flat_key_item(item, index=index, owner=owner)
        for index, item in enumerate(key)
    )


def normalize_grouping_key_input(
    key: GroupingKeyInput,
    *,
    owner: str,
) -> tuple[GroupingSingleKey, ...]:
    if isinstance(key, (tuple, list)):
        if not key:
            raise ValueError(f"{owner}: key tuple/list must be non-empty.")
        return _normalize_key_items(key, owner=owner)
    return (_require_single_key(key, owner=owner),)


def _semantic_dims(ctx: DatasetContext) -> tuple[str, ...]:
    assert ctx.sequence_dim is not None
    return ctx.batch_dims + (ctx.sequence_dim,)


def _required_group_key_dims(ctx: DatasetContext) -> tuple[str, ...]:
    if not ctx.batch_dims:
        return ()
    # Keep grouping fail-closed on the primary batch lane while allowing
    # broadcast across supplemental row dims (for example event/tau windows).
    return (ctx.batch_dims[0],)


def _reference_alignment_array(
    ctx: DatasetContext,
    *,
    probe: xr.DataArray,
    owner: str,
    what: str,
) -> xr.DataArray:
    dims = _semantic_dims(ctx)
    require_row_dim_compatibility(
        probe,
        row_dims=dims,
        allow_missing_row_dims=True,
        required_dims=_required_group_key_dims(ctx),
        owner=owner,
        what=what,
    )
    shape = tuple(int(ctx.ds.sizes[dim]) for dim in dims)
    coords = {dim: ctx.ds.coords[dim] for dim in dims if dim in ctx.ds.coords}
    data = np.broadcast_to(np.asarray(0, dtype=np.int8), shape)
    return xr.DataArray(data, dims=dims, coords=coords, name="__group_ref__")


def _expand_key_to_row_dims(
    data: xr.DataArray,
    *,
    reference: xr.DataArray,
    row_dims: tuple[str, ...],
) -> xr.DataArray:
    if tuple(data.dims) == row_dims:
        return data
    expanded, _ = xr.broadcast(data, reference)
    return expanded.transpose(*row_dims)


def _build_semantic_operand(data: xr.DataArray, *, index: int, ctx: DatasetContext) -> TopologyOperand:
    assert ctx.sequence_dim is not None
    return TopologyOperand(
        index=index,
        data=data,
        semantic=SemanticTopology(
            sequence_dim=ctx.sequence_dim,
            batch_dims=ctx.batch_dims,
            core_dims=(),
        ),
        param_coord=None,
    )


def _align_key_to_reference(
    key_data: xr.DataArray,
    *,
    ctx: DatasetContext,
    owner: str,
    what: str,
) -> xr.DataArray:
    reference = _reference_alignment_array(
        ctx,
        probe=key_data,
        owner=owner,
        what=what,
    )
    expanded_key = _expand_key_to_row_dims(
        key_data,
        reference=reference,
        row_dims=_semantic_dims(ctx),
    )
    support = operation_intent_support_for_operation_family(
        _GROUPING_OPERATION_FAMILY,
        owner=owner,
    )
    topology_intent = select_topology_policy_with_intents(
        (ctx.ao,),
        owner=owner,
        operation_family=_GROUPING_OPERATION_FAMILY,
        support=support,
        strict_policy=STRICT_EXACT_POLICY,
        semantic_policy=STRICT_EXACT_POLICY,
    )
    plan = resolve_binary_topology(
        _build_semantic_operand(
            reference,
            index=0,
            ctx=ctx,
        ),
        _build_semantic_operand(expanded_key, index=1, ctx=ctx),
        owner=owner,
        what=what,
        policy=topology_intent.policy,
    )
    _, aligned = align_exact_for_plan(plan, owner=owner, what=what)
    return aligned


def _resolve_name_key_data(
    ctx: DatasetContext,
    key_name: str,
    *,
    owner: str,
) -> tuple[ResolvedGroupingKeyKind, xr.DataArray]:
    in_coord = key_name in ctx.ds.coords
    in_var = key_name in ctx.ds.data_vars
    if in_coord and not in_var:
        return "coord", ctx.ds.coords[key_name]
    if in_var and not in_coord:
        return "data_var", ctx.ds[key_name]
    if in_coord and in_var:
        raise ValueError(f"{owner}: grouping key name {key_name!r} is ambiguous between coord and data_var.")
    if "." in key_name:
        raise ValueError(f"{owner}: dotted grouping paths are not supported in TAL v3 grouping foundation.")
    raise ValueError(f"{owner}: grouping key name {key_name!r} was not found.")


def _align_batch_key(
    data: xr.DataArray,
    *,
    ctx: DatasetContext,
    primary_dim: str,
    owner: str,
    what: str,
) -> xr.DataArray:
    if tuple(data.dims) != (primary_dim,):
        raise ValueError(
            f"{owner}: {what} must vary over exactly the primary batch dimension "
            f"{primary_dim!r}; got dims={tuple(data.dims)!r}."
        )
    indexed = require_exact_lane_indexes(
        ctx.ds,
        data,
        lane_dim=primary_dim,
        owner=owner,
        what=what,
    )
    if not indexed and int(data.sizes[primary_dim]) != int(ctx.ds.sizes[primary_dim]):
        raise ValueError(
            f"{owner}: {what} size must match primary batch dimension {primary_dim!r}."
        )
    return data


def _coerce_numeric_source(data: xr.DataArray, *, owner: str) -> xr.DataArray:
    try:
        return data.astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: bin source values must be numeric.") from exc


def _cut_to_codes(
    source: xr.DataArray,
    *,
    edges: np.ndarray,
    right: bool,
    include_lowest: bool,
) -> xr.DataArray:
    def _cut(values: np.ndarray, *, bins: np.ndarray, right: bool, include_lowest: bool) -> np.ndarray:
        flat = values.reshape(-1)
        cut = pd.cut(flat, bins=bins, labels=False, right=right, include_lowest=include_lowest)
        return np.asarray(cut, dtype="float64").reshape(values.shape)

    return xr.apply_ufunc(
        _cut,
        source,
        kwargs={"bins": edges, "right": right, "include_lowest": include_lowest},
        dask="parallelized",
        output_dtypes=[np.float64],
    )


def _codes_to_labels(codes: xr.DataArray, labels: Sequence[object]) -> xr.DataArray:
    label_values = object_label_vector(labels)

    def _map(values: np.ndarray, *, label_values: np.ndarray) -> np.ndarray:
        out = np.empty(values.shape, dtype=object)
        out.fill(np.nan)
        finite = np.isfinite(values)
        out[finite] = label_values[values[finite].astype(np.int64)]
        return out

    return xr.apply_ufunc(
        _map,
        codes,
        kwargs={"label_values": label_values},
        dask="parallelized",
        output_dtypes=[object],
    )


def _resolve_bin_source(
    ctx: DatasetContext,
    key: GroupingBinSpec,
    *,
    index: int,
    owner: str,
) -> tuple[np.ndarray, xr.DataArray, str]:
    bin_owner = f"{owner} key[{index}]"
    edges = validate_grouping_bin_spec(key, owner=bin_owner)
    if isinstance(key.source, xr.DataArray):
        return edges, key.source, key.source.name or f"external_key_{index}"
    _, data = _resolve_name_key_data(ctx, key.source, owner=bin_owner)
    return edges, data, key.source


def _build_bin_key(
    aligned: xr.DataArray,
    key: GroupingBinSpec,
    *,
    edges: np.ndarray,
    source_name: str,
    index: int,
) -> ResolvedGroupingKey:
    codes = _cut_to_codes(aligned, edges=edges, right=key.right, include_lowest=key.include_lowest)
    data = codes if key.labels is None else _codes_to_labels(codes, key.labels)
    domain = tuple(float(i) for i in range(int(edges.size - 1))) if key.labels is None else tuple(key.labels)
    return ResolvedGroupingKey(
        index=index,
        kind="bin",
        name=f"{source_name}__bin",
        data=data,
        domain_order=domain,
    )


def _resolve_bin_key(
    ctx: DatasetContext,
    key: GroupingBinSpec,
    *,
    index: int,
    owner: str,
) -> ResolvedGroupingKey:
    bin_owner = f"{owner} key[{index}]"
    edges, source_data, source_name = _resolve_bin_source(
        ctx,
        key,
        index=index,
        owner=owner,
    )
    aligned_source = _align_key_to_reference(
        _coerce_numeric_source(source_data, owner=bin_owner),
        ctx=ctx,
        owner=owner,
        what=f"grouping key[{index}]",
    )
    return _build_bin_key(
        aligned_source,
        key,
        edges=edges,
        source_name=source_name,
        index=index,
    )


def _resolve_batch_bin_key(
    ctx: DatasetContext,
    key: GroupingBinSpec,
    *,
    primary_dim: str,
    index: int,
    owner: str,
) -> ResolvedGroupingKey:
    bin_owner = f"{owner} key[{index}]"
    edges, source_data, source_name = _resolve_bin_source(
        ctx,
        key,
        index=index,
        owner=owner,
    )
    aligned = _align_batch_key(
        _coerce_numeric_source(source_data, owner=bin_owner),
        ctx=ctx,
        primary_dim=primary_dim,
        owner=owner,
        what=f"grouping key[{index}]",
    )
    return _build_bin_key(
        aligned,
        key,
        edges=edges,
        source_name=source_name,
        index=index,
    )


def resolve_grouping_key(
    ctx: DatasetContext,
    key: GroupingSingleKey,
    *,
    index: int,
    owner: str,
) -> ResolvedGroupingKey:
    key_owner = f"{owner} key[{index}]"
    if isinstance(key, GroupingBinSpec):
        return _resolve_bin_key(ctx, key, index=index, owner=owner)
    if isinstance(key, str):
        kind, data = _resolve_name_key_data(ctx, key, owner=key_owner)
        aligned = _align_key_to_reference(data, ctx=ctx, owner=owner, what=f"grouping key[{index}]")
        return ResolvedGroupingKey(index=index, kind=kind, name=key, data=aligned)
    aligned = _align_key_to_reference(key, ctx=ctx, owner=owner, what=f"grouping key[{index}]")
    name = key.name or f"external_key_{index}"
    return ResolvedGroupingKey(index=index, kind="external", name=name, data=aligned)


def resolve_batch_grouping_key(
    ctx: DatasetContext,
    key: GroupingSingleKey,
    *,
    primary_dim: str,
    index: int,
    owner: str,
) -> ResolvedGroupingKey:
    """Resolve one batch-only grouping key without realizing its payload."""
    key_owner = f"{owner} key[{index}]"
    if isinstance(key, GroupingBinSpec):
        return _resolve_batch_bin_key(
            ctx,
            key,
            primary_dim=primary_dim,
            index=index,
            owner=owner,
        )
    if isinstance(key, str):
        kind, data = _resolve_name_key_data(ctx, key, owner=key_owner)
        name = key
    else:
        kind, data = "external", key
        name = key.name or f"external_key_{index}"
    aligned = _align_batch_key(
        data,
        ctx=ctx,
        primary_dim=primary_dim,
        owner=owner,
        what=f"grouping key[{index}]",
    )
    return ResolvedGroupingKey(index=index, kind=kind, name=name, data=aligned)


__all__ = [
    "normalize_grouping_key_input",
    "resolve_batch_grouping_key",
    "resolve_grouping_key",
]
