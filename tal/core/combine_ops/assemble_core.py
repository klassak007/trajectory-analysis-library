from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..dataset_ownership import analysis_object_dataset
from .align import align_contexts
from .finalize import finalize_combine_output
from .metadata import shared_optional_name
from .normalize import normalize_inputs, resolve_contexts
from .types import AlignOptions, CombineContext, CombineResolveOptions


@dataclass(frozen=True)
class _LayoutSpec:
    shape: tuple[int, ...]
    leaves: tuple[object, ...]


@dataclass(frozen=True)
class _LeafPayload:
    context: CombineContext
    var_name: str
    data: xr.DataArray


def _is_nested_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _parse_dense_layout(
    values: object,
    *,
    depth: int,
    owner: str,
) -> _LayoutSpec:
    if depth == 0:
        if _is_nested_sequence(values):
            raise ValueError(f"{owner}: expected leaf value at nesting depth; got nested sequence.")
        return _LayoutSpec(shape=(), leaves=(values,))
    if not _is_nested_sequence(values):
        raise ValueError(f"{owner}: expected nested sequence depth {depth}; got leaf value.")
    items = list(values)
    if not items:
        raise ValueError(f"{owner}: expected non-empty nested sequence.")
    child_specs = [_parse_dense_layout(item, depth=depth - 1, owner=owner) for item in items]
    base_shape = child_specs[0].shape
    for spec in child_specs[1:]:
        if spec.shape != base_shape:
            raise ValueError(f"{owner}: ragged nested layout is not supported.")
    leaves: list[object] = []
    for spec in child_specs:
        leaves.extend(spec.leaves)
    return _LayoutSpec(shape=(len(items),) + base_shape, leaves=tuple(leaves))


def _normalize_core_dims(core_dims: Sequence[str], *, owner: str) -> tuple[str, ...]:
    if not core_dims:
        raise ValueError(f"{owner}: core_dims must be non-empty.")
    out = tuple(core_dims)
    if any(not isinstance(dim, str) or not dim for dim in out):
        raise ValueError(f"{owner}: core_dims must contain non-empty strings.")
    if len(set(out)) != len(out):
        raise ValueError(f"{owner}: core_dims must be unique; got {out!r}.")
    return out


def _normalize_axis_labels(
    raw: object | None,
    *,
    axis_dim: str,
    axis_size: int,
    owner: str,
) -> tuple[object, ...]:
    if raw is None:
        return tuple(range(axis_size))
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise ValueError(f"{owner}: labels for axis {axis_dim!r} must be a sequence.")
    labels = tuple(raw)
    if len(labels) != axis_size:
        raise ValueError(f"{owner}: labels for axis {axis_dim!r} length {len(labels)} must equal axis size {axis_size}.")
    seen: set[object] = set()
    for idx, label in enumerate(labels):
        try:
            hash(label)
        except TypeError as exc:
            raise ValueError(f"{owner}: labels for axis {axis_dim!r} must be hashable; index {idx} is invalid.") from exc
        if label in seen:
            raise ValueError(f"{owner}: labels for axis {axis_dim!r} must be unique.")
        seen.add(label)
    return labels


def _normalize_core_labels(
    core_labels: Sequence[Sequence[object]] | None,
    *,
    core_dims: tuple[str, ...],
    shape: tuple[int, ...],
    owner: str,
) -> tuple[tuple[object, ...], ...]:
    if core_labels is None:
        return tuple(_normalize_axis_labels(None, axis_dim=dim, axis_size=size, owner=owner) for dim, size in zip(core_dims, shape))
    if len(core_labels) != len(core_dims):
        raise ValueError(f"{owner}: core_labels length {len(core_labels)} must equal core_dims length {len(core_dims)}.")
    return tuple(
        _normalize_axis_labels(raw, axis_dim=dim, axis_size=size, owner=owner)
        for raw, dim, size in zip(core_labels, core_dims, shape)
    )


def _resolve_aligned_contexts(leaves: tuple[object, ...], *, owner: str) -> list[CombineContext]:
    aos = normalize_inputs(leaves, owner=owner)
    contexts = resolve_contexts(
        aos,
        resolve=CombineResolveOptions(require_sequence=True, owner=owner),
    )
    return align_contexts(
        contexts,
        opts=AlignOptions(
            batch_join="exact",
            sequence_join="exact",
            fill_value=np.nan,
            pad_invalid_outer=True,
        ),
    )


def _require_declared_context(contexts: Sequence[CombineContext], *, owner: str) -> tuple[str, tuple[str, ...]]:
    if not contexts:
        raise ValueError(f"{owner}: expected at least one leaf input.")
    sequence_dim = contexts[0].sequence_dim
    batch_dims = contexts[0].batch_dims
    if not contexts[0].roles_declared or sequence_dim is None:
        raise ValueError(f"{owner}: declared roles with sequence_dim are required on all leaves.")
    for idx, ctx in enumerate(contexts[1:], start=1):
        if not ctx.roles_declared or ctx.sequence_dim is None:
            raise ValueError(f"{owner}: declared roles with sequence_dim are required on leaf index {idx}.")
        if ctx.sequence_dim != sequence_dim:
            raise ValueError(f"{owner}: leaves have different sequence_dim values {sequence_dim!r} vs {ctx.sequence_dim!r}.")
        if ctx.batch_dims != batch_dims:
            raise ValueError(f"{owner}: leaves have different batch_dims {batch_dims!r} vs {ctx.batch_dims!r}.")
    return sequence_dim, batch_dims


def _select_single_numeric_leaf_var(context: CombineContext, *, owner: str) -> tuple[str, xr.DataArray]:
    if len(context.ds.data_vars) != 1:
        raise ValueError(f"{owner}: each leaf must contain exactly one data variable.")
    var_name = str(next(iter(context.ds.data_vars)))
    data = context.ds[var_name]
    if data.dtype.kind not in {"i", "u", "f", "c"}:
        raise ValueError(f"{owner}: data variable {var_name!r} must be numeric.")
    required = [dim for dim in (context.sequence_dim, *context.batch_dims, *context.core_dims) if dim is not None]
    missing = [dim for dim in required if dim not in data.dims]
    if missing:
        raise ValueError(f"{owner}: data variable {var_name!r} is missing declared semantic dims {missing!r}.")
    return var_name, data


def _resolve_leaf_payloads(
    contexts: Sequence[CombineContext],
    *,
    owner: str,
) -> tuple[list[_LeafPayload], tuple[str, ...], str]:
    payloads: list[_LeafPayload] = []
    leaf_core_dims: tuple[str, ...] | None = None
    default_var: str | None = None
    for idx, ctx in enumerate(contexts):
        var_name, data = _select_single_numeric_leaf_var(ctx, owner=owner)
        if default_var is None:
            default_var = var_name
        if leaf_core_dims is None:
            leaf_core_dims = ctx.core_dims
        elif ctx.core_dims != leaf_core_dims:
            raise ValueError(f"{owner}: leaf core_dims conflict {leaf_core_dims!r} vs {ctx.core_dims!r} (index {idx}).")
        payloads.append(_LeafPayload(context=ctx, var_name=var_name, data=data))
    assert leaf_core_dims is not None
    assert default_var is not None
    return payloads, leaf_core_dims, default_var


def _validate_core_dim_collisions(
    *,
    new_core_dims: tuple[str, ...],
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    leaf_core_dims: tuple[str, ...],
    owner: str,
) -> None:
    reserved = {sequence_dim, *batch_dims, *leaf_core_dims}
    overlap = [dim for dim in new_core_dims if dim in reserved]
    if overlap:
        raise ValueError(f"{owner}: new core dims collide with existing semantic dims {overlap!r}.")


def _concat_along_axis(arrays: Sequence[xr.DataArray], *, dim: str, labels: tuple[object, ...]) -> xr.DataArray:
    coord = xr.DataArray(np.asarray(labels), dims=(dim,), coords={dim: list(labels)})
    return xr.concat(list(arrays), dim=coord)


def _assemble_nd_dataarray(
    arrays: Sequence[xr.DataArray],
    *,
    shape: tuple[int, ...],
    core_dims: tuple[str, ...],
    core_labels: tuple[tuple[object, ...], ...],
) -> xr.DataArray:
    iterator = iter(arrays)

    def _build(level: int) -> xr.DataArray:
        if level == len(shape):
            return next(iterator)
        children = [_build(level + 1) for _ in range(shape[level])]
        return _concat_along_axis(children, dim=core_dims[level], labels=core_labels[level])

    out = _build(0)
    try:
        next(iterator)
    except StopIteration:
        return out
    raise ValueError("assemble_core: internal layout mismatch while assembling output.")


def _normalize_output_var(output_var: str | None, *, fallback: str, owner: str) -> str:
    if output_var is None:
        return fallback
    if not isinstance(output_var, str) or not output_var:
        raise ValueError(f"{owner}: output_var must be a non-empty string when provided.")
    return output_var


def _finalize_assembled_output(
    contexts: Sequence[CombineContext],
    assembled: xr.DataArray,
    *,
    output_var: str,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    output_core_dims: tuple[str, ...],
    validate: bool,
    source_ao: "AnalysisObject",
) -> "AnalysisObject":
    ds = assembled.rename(output_var).to_dataset()
    param_name = shared_optional_name([ctx.param_coord for ctx in contexts])
    size_name = shared_optional_name([ctx.sequence_size_coord for ctx in contexts])
    if param_name is not None and param_name not in ds.coords:
        param_name = None
    if size_name is not None and size_name not in ds.coords:
        size_name = None
    return finalize_combine_output(
        contexts[0],
        ds,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=output_core_dims,
        param_coord=param_name,
        sequence_size_coord=size_name,
        validate=validate,
        source_ao=source_ao,
    )


def _base_finalize_source(
    contexts: Sequence[CombineContext],
    *,
    owner: str,
) -> "AnalysisObject":
    if not contexts:
        raise ValueError(f"{owner}: expected at least one leaf input.")
    from ..analysis_object import AnalysisObject

    source = contexts[0].ao
    if source.__class__ is AnalysisObject:
        return source
    return AnalysisObject._from_validated(analysis_object_dataset(source))


def assemble_core(
    values: object,
    *,
    core_dims: Sequence[str],
    core_labels: Sequence[Sequence[object]] | None = None,
    output_var: str | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    owner = "assemble_core"
    new_core_dims = _normalize_core_dims(tuple(core_dims), owner=owner)
    layout = _parse_dense_layout(values, depth=len(new_core_dims), owner=owner)
    labels = _normalize_core_labels(core_labels, core_dims=new_core_dims, shape=layout.shape, owner=owner)
    aligned = _resolve_aligned_contexts(layout.leaves, owner=owner)
    sequence_dim, batch_dims = _require_declared_context(aligned, owner=owner)
    payloads, leaf_core_dims, default_var = _resolve_leaf_payloads(aligned, owner=owner)
    _validate_core_dim_collisions(
        new_core_dims=new_core_dims,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        leaf_core_dims=leaf_core_dims,
        owner=owner,
    )
    assembled = _assemble_nd_dataarray(
        [payload.data for payload in payloads],
        shape=layout.shape,
        core_dims=new_core_dims,
        core_labels=labels,
    )
    output_name = _normalize_output_var(output_var, fallback=default_var, owner=owner)
    source_ao = _base_finalize_source(aligned, owner=owner)
    return _finalize_assembled_output(
        aligned,
        assembled,
        output_var=output_name,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        output_core_dims=new_core_dims + leaf_core_dims,
        validate=validate,
        source_ao=source_ao,
    )


def stack_core(
    values: object,
    *,
    core_dim: str,
    core_labels: Sequence[object] | None = None,
    output_var: str | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    labels = None if core_labels is None else (core_labels,)
    return assemble_core(
        values,
        core_dims=(core_dim,),
        core_labels=labels,
        output_var=output_var,
        validate=validate,
    )


def block_core(
    values: object,
    *,
    row_dim: str,
    col_dim: str,
    row_labels: Sequence[object] | None = None,
    col_labels: Sequence[object] | None = None,
    output_var: str | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    labels = None
    if row_labels is not None or col_labels is not None:
        labels = (row_labels, col_labels)
    return assemble_core(
        values,
        core_dims=(row_dim, col_dim),
        core_labels=labels,
        output_var=output_var,
        validate=validate,
    )


__all__ = [
    "assemble_core",
    "block_core",
    "stack_core",
]
