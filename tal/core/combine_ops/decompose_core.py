from __future__ import annotations

from itertools import product

import xarray as xr

from ..dataset_ownership import analysis_object_dataset
from .finalize import finalize_combine_output
from .normalize import normalize_inputs, resolve_contexts
from .options import coerce_core_decompose_options
from .types import CombineContext, CombineResolveOptions, CoreDecomposeOptions


def _require_declared_roles_and_prefix_core_dims(
    context: CombineContext,
    *,
    opts: CoreDecomposeOptions,
    owner: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if not context.roles_declared or context.sequence_dim is None:
        raise ValueError(f"{owner}: declared roles with sequence_dim are required.")
    declared_core_dims = context.core_dims
    prefix_len = len(opts.core_dims)
    if prefix_len > len(declared_core_dims) or declared_core_dims[:prefix_len] != opts.core_dims:
        raise ValueError(
            f"{owner}: opts.core_dims {opts.core_dims!r} must be a prefix of declared core_dims {declared_core_dims!r}."
        )
    remaining = declared_core_dims[prefix_len:]
    return context.sequence_dim, context.batch_dims, remaining


def _select_single_numeric_var(
    context: CombineContext,
    *,
    owner: str,
) -> tuple[str, xr.DataArray]:
    if len(context.ds.data_vars) != 1:
        raise ValueError(f"{owner}: input must contain exactly one data variable.")
    var_name = str(next(iter(context.ds.data_vars)))
    data = context.ds[var_name]
    if data.dtype.kind not in {"i", "u", "f", "c"}:
        raise ValueError(f"{owner}: data variable {var_name!r} must be numeric.")
    required = [dim for dim in (context.sequence_dim, *context.batch_dims, *context.core_dims) if dim is not None]
    missing = [dim for dim in required if dim not in data.dims]
    if missing:
        raise ValueError(f"{owner}: data variable {var_name!r} is missing declared semantic dims {missing!r}.")
    return var_name, data


def _iter_row_major_keys(lengths: tuple[int, ...]) -> tuple[tuple[object, ...], ...]:
    axes = [range(length) for length in lengths]
    return tuple(tuple(idx) for idx in product(*axes))


def _resolve_label_keys_or_fail(
    data: xr.DataArray,
    *,
    core_dims: tuple[str, ...],
    owner: str,
) -> tuple[tuple[object, ...], ...]:
    axes: list[tuple[object, ...]] = []
    for dim in core_dims:
        if dim not in data.coords:
            raise ValueError(f"{owner}: key_mode='label' requires explicit coord for dim {dim!r}.")
        index = data.get_index(dim)
        if index.has_duplicates:
            raise ValueError(f"{owner}: key_mode='label' requires unique labels for dim {dim!r}.")
        labels = tuple(index.tolist())
        axes.append(labels)
    return tuple(tuple(key) for key in product(*axes))


def _resolve_keys(
    data: xr.DataArray,
    *,
    core_dims: tuple[str, ...],
    key_mode: str,
    owner: str,
) -> tuple[tuple[object, ...], ...]:
    if key_mode == "index":
        lengths = tuple(int(data.sizes[dim]) for dim in core_dims)
        return _iter_row_major_keys(lengths)
    if key_mode == "label":
        return _resolve_label_keys_or_fail(data, core_dims=core_dims, owner=owner)
    raise ValueError(f"{owner}: unsupported key_mode {key_mode!r}.")


def _extract_leaf_dataset(
    data: xr.DataArray,
    *,
    source_var: str,
    output_var: str | None,
    core_dims: tuple[str, ...],
    key: tuple[object, ...],
    key_mode: str,
) -> xr.Dataset:
    selector = {dim: value for dim, value in zip(core_dims, key)}
    if key_mode == "index":
        leaf = data.isel(selector, drop=True)
    else:
        leaf = data.sel(selector, drop=True)
    out_var = output_var if output_var is not None else source_var
    return leaf.rename(out_var).to_dataset()


def _optional_coord_name(
    ds: xr.Dataset,
    *,
    name: str | None,
) -> str | None:
    if name is None or name not in ds.coords:
        return None
    return name


def _base_finalize_source(context: CombineContext) -> "AnalysisObject":
    from ..analysis_object import AnalysisObject

    source = context.ao
    if source.__class__ is AnalysisObject:
        return source
    return AnalysisObject._from_validated(analysis_object_dataset(source))


def _finalize_leaf(
    context: CombineContext,
    *,
    ds: xr.Dataset,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    core_dims: tuple[str, ...],
    validate: bool,
    source_ao: "AnalysisObject",
) -> "AnalysisObject":
    return finalize_combine_output(
        context,
        ds,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=core_dims,
        param_coord=_optional_coord_name(ds, name=context.param_coord),
        sequence_size_coord=_optional_coord_name(ds, name=context.sequence_size_coord),
        validate=validate,
        source_ao=source_ao,
    )


def decompose_core(
    ao: object,
    *,
    opts: CoreDecomposeOptions,
    validate: bool = True,
) -> dict[tuple[object, ...], "AnalysisObject"]:
    owner = "decompose_core"
    options = coerce_core_decompose_options(opts, owner=owner)
    objects = normalize_inputs([ao], owner=owner)
    contexts = resolve_contexts(objects, resolve=CombineResolveOptions(require_sequence=False, owner=owner))
    context = contexts[0]
    sequence_dim, batch_dims, remaining_core_dims = _require_declared_roles_and_prefix_core_dims(
        context,
        opts=options,
        owner=owner,
    )
    var_name, data = _select_single_numeric_var(context, owner=owner)
    keys = _resolve_keys(data, core_dims=options.core_dims, key_mode=options.key_mode, owner=owner)
    source_ao = _base_finalize_source(context)
    out: dict[tuple[object, ...], "AnalysisObject"] = {}
    for key in keys:
        leaf_ds = _extract_leaf_dataset(
            data,
            source_var=var_name,
            output_var=options.output_var,
            core_dims=options.core_dims,
            key=key,
            key_mode=options.key_mode,
        )
        out[key] = _finalize_leaf(
            context,
            ds=leaf_ds,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            core_dims=remaining_core_dims,
            validate=validate,
            source_ao=source_ao,
        )
    return out


__all__ = [
    "decompose_core",
]
