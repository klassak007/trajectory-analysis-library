from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr

from .. import validity_values
from ..param_ops import synchronize_param
from .align import align_contexts
from .finalize import finalize_combine_output
from .metadata import (
    canonicalize_optional_names,
    resolve_core_dims,
    shared_optional_name,
)
from .normalize import effective_batch_dims, effective_sequence_dim, resolve_contexts
from .types import AlignOptions, CombineContext, CombineResolveOptions, MergeOptions


def _outer_dims(
    *,
    opts: MergeOptions,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
) -> tuple[str, ...]:
    dims: list[str] = []
    if opts.batch_join == "outer":
        dims.extend(batch_dims)
    if opts.sequence_join == "outer" and sequence_dim is not None:
        dims.append(sequence_dim)
    return tuple(dims)


def _find_var_source(var_name: str, *, sources: list[CombineContext]) -> xr.Dataset | None:
    for ctx in sources:
        if var_name in ctx.ds.data_vars:
            return ctx.ds
    return None


def _mask_for_outer_holes(
    source: xr.Dataset,
    target: xr.Dataset,
    *,
    var_dims: tuple[str, ...],
    outer_dims: tuple[str, ...],
) -> xr.DataArray | None:
    mask: xr.DataArray | None = None
    for dim in var_dims:
        if dim not in outer_dims or dim not in target.dims:
            continue
        if dim in source.dims:
            missing = ~target.get_index(dim).isin(source.get_index(dim))
        else:
            continue
        axis = xr.DataArray(np.asarray(missing, dtype=bool), dims=(dim,), coords={dim: target.coords[dim]})
        mask = axis if mask is None else (mask | axis)
    return mask


def _fill_value_for_var(
    fill: float | int | None | Mapping[str, float | int | None],
    *,
    var_name: str,
) -> float | int | None:
    if isinstance(fill, Mapping):
        return fill.get(var_name)
    return fill


def _is_noop_fill(value: float | int | None) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _apply_outer_fill_scoped(
    ds: xr.Dataset,
    *,
    sources: list[CombineContext],
    opts: MergeOptions,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
) -> xr.Dataset:
    outer_dims = _outer_dims(opts=opts, sequence_dim=sequence_dim, batch_dims=batch_dims)
    if not outer_dims:
        return ds
    out = ds.copy(deep=False)
    for name, var in list(out.data_vars.items()):
        fill_value = _fill_value_for_var(opts.outer_fill_value, var_name=str(name))
        if _is_noop_fill(fill_value):
            continue
        source = _find_var_source(str(name), sources=sources)
        if source is None:
            continue
        mask = _mask_for_outer_holes(
            source,
            out,
            var_dims=tuple(var.dims),
            outer_dims=outer_dims,
        )
        if mask is None:
            continue
        out[str(name)] = var.where(~mask, other=fill_value)
    return out


def _merge_override_left_biased_fill_holes(
    datasets: list[xr.Dataset],
    *,
    combine_attrs: str,
) -> xr.Dataset:
    out = xr.merge(
        datasets,
        compat="override",
        combine_attrs=combine_attrs,
    )
    for ds in datasets[1:]:
        out = _merge_override_source_vars(out, source=ds)
    return out


def _merge_override_source_vars(out: xr.Dataset, *, source: xr.Dataset) -> xr.Dataset:
    for name in source.variables:
        out = _merge_override_var(out, source=source, name=str(name))
    return out


def _merge_override_var(out: xr.Dataset, *, source: xr.Dataset, name: str) -> xr.Dataset:
    if name not in out.variables:
        return _assign_missing_override_var(out, source=source, name=name)
    return _fill_override_var_holes(out, source=source, name=name)


def _assign_missing_override_var(out: xr.Dataset, *, source: xr.Dataset, name: str) -> xr.Dataset:
    if name in source.coords:
        return out.assign_coords({name: source[name]})
    out[name] = source[name]
    return out


def _fill_override_var_holes(out: xr.Dataset, *, source: xr.Dataset, name: str) -> xr.Dataset:
    filled = out[name].combine_first(source[name])
    if name in out.coords:
        return out.assign_coords({name: filled})
    out[name] = filled
    return out


def _prealign_contexts(contexts: list[CombineContext], *, opts: MergeOptions) -> list[CombineContext]:
    if opts.param_prealign is None:
        return contexts
    synced = synchronize_param(
        [ctx.ao for ctx in contexts],
        grid=opts.param_prealign.grid,
        opts=opts.param_prealign.sync_opts,
    )
    return resolve_contexts(
        synced,
        resolve=CombineResolveOptions(require_sequence=False, owner="merge"),
    )


def _validate_sequence_size_coord_if_present(
    ds: xr.Dataset,
    *,
    size_name: str | None,
    sequence_dim: str | None,
) -> tuple[xr.Dataset, str | None]:
    if size_name is None or sequence_dim is None or size_name not in ds.coords:
        return ds, size_name
    try:
        validated = validity_values.require_valid_sequence_size_values(
            ds.coords[size_name],
            sequence_size_coord=size_name,
            sequence_len=int(ds.sizes.get(sequence_dim, 0)),
            owner="merge",
        )
    except ValueError:
        return ds.drop_vars(size_name, errors="ignore"), None
    return ds.assign_coords({size_name: validated}), size_name


def _merge_metadata(
    aligned: list[CombineContext],
    *,
    ds: xr.Dataset,
) -> tuple[xr.Dataset, str | None, tuple[str, ...], str | None, str | None, tuple[str, ...]]:
    sequence_dim = effective_sequence_dim(aligned, owner="merge", require=False)
    batch_dims = effective_batch_dims(aligned)
    out, param_name, size_name = canonicalize_optional_names(
        ds,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        param_name=shared_optional_name([ctx.param_coord for ctx in aligned]),
        size_name=shared_optional_name([ctx.sequence_size_coord for ctx in aligned]),
    )
    out, size_name = _validate_sequence_size_coord_if_present(
        out,
        size_name=size_name,
        sequence_dim=sequence_dim,
    )
    core_dims = resolve_core_dims(
        aligned,
        ds=out,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        owner="merge",
    )
    return out, sequence_dim, batch_dims, param_name, size_name, core_dims


def _assert_compat_no_chunked_overlap(
    contexts: list[CombineContext],
    *,
    compat: str,
) -> None:
    if not _is_chunked_overlap_guarded_compat(compat):
        return
    counts, chunked = _collect_overlap_counts(contexts)
    bad = _overlapping_chunked_names(counts=counts, chunked=chunked)
    if not bad:
        return
    preview = ", ".join(repr(name) for name in bad[:3])
    suffix = " ..." if len(bad) > 3 else ""
    raise ValueError(
        "merge: compat="
        f"{compat!r} does not support overlapping chunked variables/coords ({preview}{suffix}); "
        "precompute overlapping inputs or use compat='override'."
    )


def _is_chunked_overlap_guarded_compat(compat: str) -> bool:
    return compat in {"identical", "equals", "broadcast_equals", "no_conflicts"}


def _collect_overlap_counts(contexts: list[CombineContext]) -> tuple[dict[str, int], set[str]]:
    counts: dict[str, int] = {}
    chunked: set[str] = set()
    for ctx in contexts:
        for name, var in ctx.ds.variables.items():
            key = str(name)
            counts[key] = counts.get(key, 0) + 1
            if getattr(var.data, "chunks", None) is not None:
                chunked.add(key)
    return counts, chunked


def _overlapping_chunked_names(*, counts: dict[str, int], chunked: set[str]) -> list[str]:
    return sorted(name for name, count in counts.items() if count > 1 and name in chunked)


def merge_contexts(
    contexts: list[CombineContext],
    *,
    opts: MergeOptions,
    validate: bool,
) -> "AnalysisObject":
    prealigned = _prealign_contexts(contexts, opts=opts)
    aligned = align_contexts(
        prealigned,
        opts=AlignOptions(
            batch_join=opts.batch_join,
            sequence_join=opts.sequence_join,
            fill_value=np.nan,
            pad_invalid_outer=True,
        ),
    )
    merged = _merge_aligned_datasets(aligned, opts=opts)
    return _finalize_merged_context(
        aligned,
        prealigned=prealigned,
        merged=merged,
        opts=opts,
        validate=validate,
    )


def _merge_aligned_datasets(aligned: list[CombineContext], *, opts: MergeOptions) -> xr.Dataset:
    if opts.compat == "override":
        return _merge_override_left_biased_fill_holes(
            [ctx.ds for ctx in aligned],
            combine_attrs=opts.combine_attrs,
        )
    _assert_compat_no_chunked_overlap(aligned, compat=opts.compat)
    return xr.merge(
        [ctx.ds for ctx in aligned],
        compat=opts.compat,
        combine_attrs=opts.combine_attrs,
    )


def _finalize_merged_context(
    aligned: list[CombineContext],
    *,
    prealigned: list[CombineContext],
    merged: xr.Dataset,
    opts: MergeOptions,
    validate: bool,
) -> "AnalysisObject":
    ds, sequence_dim, batch_dims, param_name, size_name, core_dims = _merge_metadata(aligned, ds=merged)
    ds = _apply_outer_fill_scoped(
        ds,
        sources=prealigned,
        opts=opts,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
    )
    ds, size_name = _validate_sequence_size_coord_if_present(
        ds,
        size_name=size_name,
        sequence_dim=sequence_dim,
    )
    return finalize_combine_output(
        aligned[0],
        ds,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims if sequence_dim is not None else (),
        core_dims=core_dims,
        param_coord=param_name,
        sequence_size_coord=size_name,
        validate=validate,
    )
