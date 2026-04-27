from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from .align import align_contexts
from .finalize import finalize_combine_output
from .normalize import normalize_inputs, resolve_contexts
from .options import coerce_core_overlay_options
from .types import AlignOptions, CombineContext, CombineResolveOptions, CoreOverlayOptions

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject

_MISSING_SCALAR_SENTINEL = ("missing", "scalar")


@dataclass(frozen=True)
class _OverlayPayload:
    context: CombineContext
    var_name: str
    data: xr.DataArray


@dataclass(frozen=True)
class _OverlayRuntime:
    sequence_dim: str
    batch_dims: tuple[str, ...]
    core_dims: tuple[str, ...]
    base_payload: _OverlayPayload
    patch_payloads: tuple[_OverlayPayload, ...]


def _normalize_patches(patches: Sequence[object] | object, *, owner: str) -> list[object]:
    if isinstance(patches, Sequence) and not isinstance(patches, (str, bytes, bytearray)):
        out = list(patches)
    else:
        out = [patches]
    if not out:
        raise ValueError(f"{owner}: at least one patch input is required.")
    return out


def _require_shared_declared_roles(
    contexts: Sequence[CombineContext],
    *,
    owner: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    first = contexts[0]
    if not first.roles_declared or first.sequence_dim is None:
        raise ValueError(f"{owner}: declared roles with sequence_dim are required on all inputs.")
    sequence_dim = first.sequence_dim
    batch_dims = first.batch_dims
    core_dims = first.core_dims
    for idx, ctx in enumerate(contexts[1:], start=1):
        if not ctx.roles_declared or ctx.sequence_dim is None:
            raise ValueError(f"{owner}: declared roles with sequence_dim are required on input index {idx}.")
        if ctx.sequence_dim != sequence_dim or ctx.batch_dims != batch_dims or ctx.core_dims != core_dims:
            raise ValueError(f"{owner}: all inputs must share sequence_dim, batch_dims, and core_dims.")
    return sequence_dim, batch_dims, core_dims


def _require_target_core_dim(
    core_dims: tuple[str, ...],
    *,
    core_dim: str,
    owner: str,
) -> None:
    if core_dim not in core_dims:
        raise ValueError(f"{owner}: opts.core_dim {core_dim!r} is not in declared core_dims {core_dims!r}.")


def _select_single_numeric_var(
    context: CombineContext,
    *,
    owner: str,
) -> tuple[str, xr.DataArray]:
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
) -> list[_OverlayPayload]:
    out: list[_OverlayPayload] = []
    for ctx in contexts:
        var_name, data = _select_single_numeric_var(ctx, owner=owner)
        out.append(_OverlayPayload(context=ctx, var_name=var_name, data=data))
    return out


def _require_hashable_unique_labels(
    index: pd.Index,
    *,
    dim: str,
    owner: str,
    operand: str,
) -> None:
    _index_key_map(index, owner=owner, operand=operand, dim=dim)


def _require_explicit_unique_target_labels(
    data: xr.DataArray,
    *,
    dim: str,
    owner: str,
    operand: str,
) -> pd.Index:
    if dim not in data.coords:
        raise ValueError(f"{owner}: {operand} requires explicit coordinate labels for core dim {dim!r}.")
    index = data.get_index(dim)
    _require_hashable_unique_labels(index, dim=dim, owner=owner, operand=operand)
    return index


def _require_non_target_core_match(
    *,
    base: xr.DataArray,
    patch: xr.DataArray,
    core_dims: tuple[str, ...],
    target_core_dim: str,
    owner: str,
    patch_index: int,
) -> None:
    for dim in core_dims:
        if dim == target_core_dim:
            continue
        if patch.get_index(dim).equals(base.get_index(dim)):
            continue
        raise ValueError(f"{owner}: non-target core dim {dim!r} labels differ at patch index {patch_index}.")


def _is_missing_scalar(value: object) -> bool:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    is_na = pd.isna(value)
    return isinstance(is_na, bool) and is_na


def _canonical_label_key(label: object) -> object:
    if isinstance(label, tuple):
        return ("tuple", tuple(_canonical_label_key(item) for item in label))
    if isinstance(label, frozenset):
        parts = [_canonical_label_key(item) for item in label]
        ordered = tuple(sorted(parts, key=repr))
        return ("frozenset", ordered)
    if _is_missing_scalar(label):
        return _MISSING_SCALAR_SENTINEL
    try:
        hash(label)
    except TypeError as exc:
        raise TypeError("overlay_core: labels must be hashable.") from exc
    return ("value", label)


def _index_key_map(
    index: pd.Index,
    *,
    owner: str,
    operand: str,
    dim: str,
) -> dict[object, int]:
    out: dict[object, int] = {}
    duplicates: list[object] = []
    for idx, label in enumerate(index.tolist()):
        try:
            key = _canonical_label_key(label)
        except TypeError as exc:
            raise ValueError(
                f"{owner}: {operand} labels for dim {dim!r} must be hashable; index {idx} is invalid."
            ) from exc
        if key in out and label not in duplicates:
            duplicates.append(label)
        out.setdefault(key, idx)
    if duplicates:
        raise ValueError(f"{owner}: {operand} labels for dim {dim!r} must be unique.")
    return out


def _require_patch_labels_subset(
    *,
    patch_labels: pd.Index,
    base_labels: pd.Index,
    owner: str,
    patch_index: int,
) -> None:
    base_map = _index_key_map(base_labels, owner=owner, operand="base", dim=base_labels.name or "<unnamed>")
    missing: list[object] = []
    for label in patch_labels.tolist():
        key = _canonical_label_key(label)
        if key in base_map:
            continue
        if label not in missing:
            missing.append(label)
    if not missing:
        return
    raise ValueError(
        f"{owner}: patch index {patch_index} contains labels not present in base target axis: {missing!r}."
    )


def _require_no_patch_overlap(
    patch_labels: Sequence[pd.Index],
    *,
    owner: str,
) -> None:
    seen: set[object] = set()
    for patch_index, labels in enumerate(patch_labels):
        overlap: list[object] = []
        for label in labels.tolist():
            key = _canonical_label_key(label)
            if key in seen and label not in overlap:
                overlap.append(label)
            seen.add(key)
        if overlap:
            raise ValueError(
                f"{owner}: overlapping patch labels are not allowed with on_overlap='error'; "
                f"patch index {patch_index} overlaps labels {overlap!r}."
            )


def _overlay_single_patch(
    base_data: xr.DataArray,
    patch_data: xr.DataArray,
    *,
    core_dim: str,
) -> xr.DataArray:
    base_map = _index_key_map(base_data.get_index(core_dim), owner="overlay_core", operand="base", dim=core_dim)
    patch_map = _index_key_map(patch_data.get_index(core_dim), owner="overlay_core", operand="patch", dim=core_dim)
    positions: list[tuple[int, int]] = []
    for key, patch_pos in patch_map.items():
        base_pos = base_map.get(key)
        if base_pos is None:
            continue
        positions.append((base_pos, patch_pos))
    if not positions:
        return base_data
    patched = base_data.copy(deep=False)
    for base_pos, patch_pos in sorted(positions):
        patched[{core_dim: base_pos}] = patch_data.isel({core_dim: patch_pos})
    return patched.transpose(*base_data.dims)


def _overlay_kernel(
    *,
    base_data: xr.DataArray,
    patches: Sequence[xr.DataArray],
    core_dim: str,
) -> xr.DataArray:
    out = base_data
    for patch in patches:
        out = _overlay_single_patch(out, patch, core_dim=core_dim)
    return out.reindex({core_dim: base_data.get_index(core_dim)})


def _optional_coord_name(
    ds: xr.Dataset,
    *,
    name: str | None,
) -> str | None:
    if name is None or name not in ds.coords:
        return None
    return name


def _finalize_overlay_output(
    context: CombineContext,
    *,
    ds: xr.Dataset,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    core_dims: tuple[str, ...],
    output_var: str,
    validate: bool,
) -> "AnalysisObject":
    renamed = ds.rename({next(iter(ds.data_vars)): output_var})
    return finalize_combine_output(
        context,
        renamed,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=core_dims,
        param_coord=_optional_coord_name(renamed, name=context.param_coord),
        sequence_size_coord=_optional_coord_name(renamed, name=context.sequence_size_coord),
        validate=validate,
    )


def _prepare_overlay_runtime(
    aligned: Sequence[CombineContext],
    *,
    options: CoreOverlayOptions,
    owner: str,
) -> _OverlayRuntime:
    sequence_dim, batch_dims, core_dims = _require_shared_declared_roles(aligned, owner=owner)
    _require_target_core_dim(core_dims, core_dim=options.core_dim, owner=owner)
    payloads = _resolve_payloads(aligned, owner=owner)
    base_payload = payloads[0]
    patch_payloads = tuple(payloads[1:])
    base_labels = _require_explicit_unique_target_labels(
        base_payload.data,
        dim=options.core_dim,
        owner=owner,
        operand="base",
    )
    patch_label_indexes: list[pd.Index] = []
    for idx, payload in enumerate(patch_payloads):
        _require_non_target_core_match(
            base=base_payload.data,
            patch=payload.data,
            core_dims=core_dims,
            target_core_dim=options.core_dim,
            owner=owner,
            patch_index=idx,
        )
        labels = _require_explicit_unique_target_labels(
            payload.data,
            dim=options.core_dim,
            owner=owner,
            operand=f"patch index {idx}",
        )
        _require_patch_labels_subset(patch_labels=labels, base_labels=base_labels, owner=owner, patch_index=idx)
        patch_label_indexes.append(labels)
    if options.on_overlap == "error":
        _require_no_patch_overlap(patch_label_indexes, owner=owner)
    return _OverlayRuntime(
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=core_dims,
        base_payload=base_payload,
        patch_payloads=patch_payloads,
    )


def overlay_core(
    base: object,
    patches: Sequence[object] | object,
    *,
    opts: CoreOverlayOptions,
    validate: bool = True,
) -> "AnalysisObject":
    owner = "overlay_core"
    options = coerce_core_overlay_options(opts, owner=owner)
    patch_values = _normalize_patches(patches, owner=owner)
    objects = normalize_inputs([base, *patch_values], owner=owner)
    contexts = resolve_contexts(objects, resolve=CombineResolveOptions(require_sequence=True, owner=owner))
    aligned = align_contexts(
        contexts,
        opts=AlignOptions(batch_join="exact", sequence_join="exact", fill_value=float("nan"), pad_invalid_outer=True),
    )
    runtime = _prepare_overlay_runtime(aligned, options=options, owner=owner)
    out_var = options.output_var if options.output_var is not None else runtime.base_payload.var_name
    out_data = _overlay_kernel(
        base_data=runtime.base_payload.data,
        patches=[payload.data for payload in runtime.patch_payloads],
        core_dim=options.core_dim,
    )
    return _finalize_overlay_output(
        runtime.base_payload.context,
        ds=out_data.to_dataset(name=runtime.base_payload.var_name),
        sequence_dim=runtime.sequence_dim,
        batch_dims=runtime.batch_dims,
        core_dims=runtime.core_dims,
        output_var=out_var,
        validate=validate,
    )


__all__ = [
    "overlay_core",
]
