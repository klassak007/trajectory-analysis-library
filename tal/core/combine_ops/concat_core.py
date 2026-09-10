from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from .align import align_contexts
from .finalize import (
    CombineFinalizationPlan,
    finalize_combine_output,
    prepare_combine_finalization,
)
from .metadata import canonicalize_optional_names, shared_optional_name
from .normalize import normalize_inputs, resolve_contexts
from .options import coerce_core_concat_options
from .types import AlignOptions, CombineContext, CombineResolveOptions, CoreConcatOptions


@dataclass(frozen=True)
class _CoreConcatPayload:
    context: CombineContext
    var_name: str
    data: xr.DataArray


def _require_shared_declared_roles(
    contexts: Sequence[CombineContext],
    *,
    owner: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if not contexts:
        raise ValueError(f"{owner}: expected at least one input.")
    first = contexts[0]
    if not first.roles_declared or first.sequence_dim is None:
        raise ValueError(f"{owner}: declared roles with sequence_dim are required on all inputs.")
    sequence_dim = first.sequence_dim
    batch_dims = first.batch_dims
    core_dims = first.core_dims
    for idx, ctx in enumerate(contexts[1:], start=1):
        if not ctx.roles_declared or ctx.sequence_dim is None:
            raise ValueError(f"{owner}: declared roles with sequence_dim are required on input index {idx}.")
        if ctx.sequence_dim != sequence_dim:
            raise ValueError(f"{owner}: sequence_dim mismatch {sequence_dim!r} vs {ctx.sequence_dim!r}.")
        if ctx.batch_dims != batch_dims:
            raise ValueError(f"{owner}: batch_dims mismatch {batch_dims!r} vs {ctx.batch_dims!r}.")
        if ctx.core_dims != core_dims:
            raise ValueError(f"{owner}: core_dims mismatch {core_dims!r} vs {ctx.core_dims!r}.")
    return sequence_dim, batch_dims, core_dims


def _require_target_core_dim(
    core_dims: tuple[str, ...],
    *,
    core_dim: str,
    owner: str,
) -> int:
    try:
        return core_dims.index(core_dim)
    except ValueError as exc:
        raise ValueError(f"{owner}: opts.core_dim {core_dim!r} is not in declared core_dims {core_dims!r}.") from exc


def _select_single_numeric_var(context: CombineContext, *, owner: str) -> tuple[str, xr.DataArray]:
    if len(context.ds.data_vars) != 1:
        raise ValueError(f"{owner}: each input must contain exactly one data variable.")
    var_name = str(next(iter(context.ds.data_vars)))
    data = context.ds[var_name]
    if data.dtype.kind not in {"i", "u", "f", "c"}:
        raise ValueError(f"{owner}: data variable {var_name!r} must be numeric.")
    required = [dim for dim in (context.sequence_dim, *context.batch_dims, *context.core_dims) if dim is not None]
    missing = [dim for dim in required if dim not in data.dims]
    if missing:
        raise ValueError(f"{owner}: data variable {var_name!r} is missing declared semantic dims {missing!r}.")
    return var_name, data


def _resolve_payloads(
    contexts: Sequence[CombineContext],
    *,
    owner: str,
) -> list[_CoreConcatPayload]:
    out: list[_CoreConcatPayload] = []
    for ctx in contexts:
        var_name, data = _select_single_numeric_var(ctx, owner=owner)
        out.append(_CoreConcatPayload(context=ctx, var_name=var_name, data=data))
    return out


def _require_non_target_core_match(
    payloads: Sequence[_CoreConcatPayload],
    *,
    core_dims: tuple[str, ...],
    target_core_dim: str,
    owner: str,
) -> None:
    if len(payloads) <= 1:
        return
    base = payloads[0].data
    for dim in core_dims:
        if dim == target_core_dim:
            continue
        expected = base.get_index(dim)
        for idx, payload in enumerate(payloads[1:], start=1):
            actual = payload.data.get_index(dim)
            if actual.equals(expected):
                continue
            raise ValueError(f"{owner}: non-target core dim {dim!r} labels differ at input index {idx}.")


def _validate_labels(
    labels: tuple[object, ...],
    *,
    expected_size: int,
    owner: str,
) -> tuple[object, ...]:
    if len(labels) != expected_size:
        raise ValueError(f"{owner}: core_labels length {len(labels)} must equal concatenated core size {expected_size}.")
    seen: set[object] = set()
    for idx, label in enumerate(labels):
        try:
            hash(label)
        except TypeError as exc:
            raise ValueError(f"{owner}: core_labels entries must be hashable; index {idx} is invalid.") from exc
        if label in seen:
            raise ValueError(f"{owner}: core_labels must be unique; duplicate {label!r}.")
        seen.add(label)
    return labels


def _resolve_concat_axis_labels(
    payloads: Sequence[_CoreConcatPayload],
    *,
    core_dim: str,
    explicit_labels: tuple[object, ...] | None,
    owner: str,
) -> tuple[object, ...]:
    size = sum(int(payload.data.sizes[core_dim]) for payload in payloads)
    if explicit_labels is not None:
        return _validate_labels(explicit_labels, expected_size=size, owner=owner)
    if not all(core_dim in payload.data.coords for payload in payloads):
        return tuple(range(size))
    labels: list[object] = []
    for payload in payloads:
        labels.extend(payload.data.get_index(core_dim).tolist())
    return _validate_labels(tuple(labels), expected_size=size, owner=owner)


def _concat_core_kernel(
    arrays: Sequence[xr.DataArray],
    *,
    core_dim: str,
    labels: tuple[object, ...],
) -> xr.DataArray:
    coord = xr.DataArray(np.asarray(labels), dims=(core_dim,), coords={core_dim: list(labels)})
    return xr.concat(
        list(arrays),
        dim=coord,
        join="exact",
        coords="different",
        compat="no_conflicts",
    )


def concat_core_contexts(
    contexts: list[CombineContext],
    *,
    opts: CoreConcatOptions,
    validate: bool,
    finalization: CombineFinalizationPlan,
) -> "AnalysisObject":
    sequence_dim, batch_dims, core_dims = _require_shared_declared_roles(contexts, owner="concat_core")
    _require_target_core_dim(core_dims, core_dim=opts.core_dim, owner="concat_core")
    payloads = _resolve_payloads(contexts, owner="concat_core")
    _require_non_target_core_match(
        payloads,
        core_dims=core_dims,
        target_core_dim=opts.core_dim,
        owner="concat_core",
    )
    labels = _resolve_concat_axis_labels(
        payloads,
        core_dim=opts.core_dim,
        explicit_labels=opts.core_labels,
        owner="concat_core",
    )
    out_var = opts.output_var if opts.output_var is not None else payloads[0].var_name
    out_data = _concat_core_kernel([payload.data for payload in payloads], core_dim=opts.core_dim, labels=labels)
    ds_out = out_data.to_dataset(name=out_var)
    ds_out, param_name, size_name = canonicalize_optional_names(
        ds_out,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        param_name=shared_optional_name([ctx.param_coord for ctx in contexts]),
        size_name=shared_optional_name([ctx.sequence_size_coord for ctx in contexts]),
    )
    return finalize_combine_output(
        finalization,
        ds_out,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=core_dims,
        param_coord=param_name,
        sequence_size_coord=size_name,
        validate=validate,
    )


def concat_core(
    aos: Sequence[object],
    *,
    opts: CoreConcatOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    options = coerce_core_concat_options(opts, owner="concat_core")
    objects = normalize_inputs(aos, owner="concat_core")
    contexts = resolve_contexts(objects, resolve=CombineResolveOptions(require_sequence=True, owner="concat_core"))
    finalization = prepare_combine_finalization(
        contexts,
        owner="concat_core",
    )
    aligned = align_contexts(
        contexts,
        opts=AlignOptions(batch_join="exact", sequence_join="exact", fill_value=np.nan, pad_invalid_outer=True),
    )
    return concat_core_contexts(
        aligned,
        opts=options,
        validate=validate,
        finalization=finalization,
    )


__all__ = [
    "concat_core",
    "concat_core_contexts",
]
