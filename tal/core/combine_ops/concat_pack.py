from __future__ import annotations

"""Concat-sequence packing/masking ownership."""

import numpy as np
import pandas as pd
import xarray as xr

from ..param_ops.guards import dataset_namespace_names, mark_reserved_coord, unique_temp_dim
from .concat_overlap import validate_grouped_sort_payload
from .concat_plan import ConcatSequencePlan
from .concat_sort import apply_overlap_sort as _apply_overlap_sort_rows
from .types import SequenceConcatOptions


def _build_packing_index(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segments, batch = lengths.shape
    total = lengths.sum(axis=0)
    width = int(total.max(initial=0))
    seg_idx = np.zeros((batch, width), dtype="int64")
    sample_idx = np.zeros((batch, width), dtype="int64")
    valid = np.zeros((batch, width), dtype=bool)
    starts = np.cumsum(lengths, axis=0) - lengths
    for seg in range(segments):
        for row in range(batch):
            length = int(lengths[seg, row])
            if length <= 0:
                continue
            start = int(starts[seg, row])
            stop = start + length
            seg_idx[row, start:stop] = seg
            sample_idx[row, start:stop] = np.arange(length, dtype="int64")
            valid[row, start:stop] = True
    return seg_idx, sample_idx, valid


def _mask_invalid_slots(
    ds: xr.Dataset,
    *,
    valid: xr.DataArray,
    sequence_dim: str,
    protected_names: tuple[str, ...] = (),
) -> xr.Dataset:
    out = ds
    for name, var in list(out.data_vars.items()):
        if sequence_dim in var.dims:
            out[name] = var.where(valid.broadcast_like(var))
    for name, coord in list(out.coords.items()):
        if name in {sequence_dim, "valid", "sample_index", *protected_names} or sequence_dim not in coord.dims:
            continue
        out = out.assign_coords({name: coord.where(valid.broadcast_like(coord))})
    return out


def _normalize_runtime_reserved_coords(
    ds: xr.Dataset,
    *,
    valid: xr.DataArray,
    sequence_dim: str,
) -> xr.Dataset:
    updates: dict[str, xr.DataArray] = {"valid": mark_reserved_coord(valid.astype(bool), name="valid")}
    if "sample_index" in ds.coords:
        sample = ds.coords["sample_index"]
        if sequence_dim in sample.dims:
            mask = valid.broadcast_like(sample)
            sample = sample.where(mask, other=np.int64(-1))
        updates["sample_index"] = mark_reserved_coord(sample.astype("int64"), name="sample_index")
    return ds.assign_coords(updates)


def _broadcast_sequence_only_payload(
    ds: xr.Dataset,
    *,
    flat_dim: str,
    sequence_dim: str,
) -> xr.Dataset:
    labels = ds.coords[flat_dim]
    out = ds
    for name, var in list(out.data_vars.items()):
        if tuple(var.dims) != (sequence_dim,):
            continue
        out[name] = var.expand_dims({flat_dim: labels}).transpose(flat_dim, sequence_dim)
    for name, coord in list(out.coords.items()):
        if name in {sequence_dim, flat_dim} or tuple(coord.dims) != (sequence_dim,):
            continue
        expanded = coord.expand_dims({flat_dim: labels}).transpose(flat_dim, sequence_dim)
        out = out.assign_coords({name: expanded})
    return out


def _sequence_coord_names(
    aligned: list[xr.Dataset],
    *,
    sequence_dim: str,
    flat_dim: str,
) -> list[str]:
    return sorted(
        {
            str(name)
            for ds in aligned
            for name, coord in ds.coords.items()
            if name not in {sequence_dim, flat_dim} and sequence_dim in coord.dims
        }
    )


def _concat_segment_inputs(
    aligned: list[xr.Dataset],
    *,
    size_name: str | None,
    sequence_coord_names: list[str],
) -> list[xr.Dataset]:
    out: list[xr.Dataset] = []
    for ds in aligned:
        prepared = ds.drop_vars(size_name, errors="ignore") if size_name else ds
        prepared = prepared.reset_coords(
            names=[name for name in sequence_coord_names if name in prepared.coords],
            drop=False,
        )
        out.append(prepared)
    return out


def _packed_concat_dataset(
    aligned: list[xr.Dataset],
    *,
    sequence_dim: str,
    flat_dim: str,
    labels: pd.Index,
    lengths: np.ndarray,
    fill_value: object,
    size_name: str | None,
) -> tuple[xr.Dataset, xr.DataArray]:
    seg_idx, sample_idx, valid_arr = _build_packing_index(lengths)
    names: set[str] = {flat_dim, sequence_dim}
    for ds in aligned:
        names.update(dataset_namespace_names(ds))
    out_dim = unique_temp_dim("__tal_out", taken_dims=tuple(sorted(names)))
    seg_dim = unique_temp_dim("__segment__", taken_dims=tuple(sorted(names | {out_dim})))
    coords = {flat_dim: labels, out_dim: np.arange(seg_idx.shape[1])}
    seg_da = xr.DataArray(seg_idx, dims=(flat_dim, out_dim), coords=coords)
    smp_da = xr.DataArray(sample_idx, dims=(flat_dim, out_dim), coords=coords)
    valid = xr.DataArray(valid_arr, dims=(flat_dim, out_dim), coords=coords).rename({out_dim: sequence_dim})
    sequence_coord_names = _sequence_coord_names(aligned, sequence_dim=sequence_dim, flat_dim=flat_dim)
    concat_inputs = _concat_segment_inputs(aligned, size_name=size_name, sequence_coord_names=sequence_coord_names)
    stacked_ds = xr.concat(
        concat_inputs,
        dim=seg_dim,
        join="outer",
        coords="different",
        data_vars="all",
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        fill_value=fill_value,
    )
    out = stacked_ds.isel({seg_dim: seg_da, sequence_dim: smp_da}).drop_vars(seg_dim, errors="ignore")
    out = out.drop_vars(sequence_dim, errors="ignore").rename_dims({out_dim: sequence_dim})
    out = out.drop_vars(out_dim, errors="ignore").assign_coords({sequence_dim: np.arange(out.sizes[sequence_dim], dtype="int64")})
    for name in sequence_coord_names:
        if name in out.data_vars:
            out = out.set_coords(name)
    return out, valid


def _apply_valid_mask(
    out: xr.Dataset,
    *,
    valid: xr.DataArray,
    sequence_dim: str,
    flat_dim: str,
    size_name: str | None,
) -> xr.Dataset:
    masked = _broadcast_sequence_only_payload(out, flat_dim=flat_dim, sequence_dim=sequence_dim)
    masked = _mask_invalid_slots(
        masked,
        valid=valid,
        sequence_dim=sequence_dim,
        protected_names=((size_name,) if size_name else ()),
    )
    return _normalize_runtime_reserved_coords(masked, valid=valid, sequence_dim=sequence_dim)


def _apply_overlap_sort(
    out: xr.Dataset,
    *,
    overlap: str,
    param_name: str | None,
    batch_dims: tuple[str, ...],
    flat_dim: str,
    sequence_dim: str,
    size_name: str | None,
) -> xr.Dataset:
    if overlap != "sort":
        return out
    if param_name is None:
        raise ValueError("concat_sequence: overlap='sort' requires param_coord on all inputs.")
    if batch_dims:
        validate_grouped_sort_payload(
            out,
            flat_dim=flat_dim,
            sequence_dim=sequence_dim,
            owner="concat_sequence",
        )
    if param_name not in out.coords:
        if int(out.sizes.get(sequence_dim, 0)) == 0:
            return out
        raise ValueError("concat_sequence: overlap='sort' requires a canonical param_coord output.")
    sorted_out = _apply_overlap_sort_rows(
        out,
        overlap=overlap,
        flat_dim=flat_dim,
        sequence_dim=sequence_dim,
        param_name=param_name,
    )
    remasked = _mask_invalid_slots(
        sorted_out,
        valid=sorted_out.coords["valid"],
        sequence_dim=sequence_dim,
        protected_names=((size_name,) if size_name else ()),
    )
    return _normalize_runtime_reserved_coords(
        remasked,
        valid=remasked.coords["valid"],
        sequence_dim=sequence_dim,
    )


def build_packed_concat_dataset(
    plan: ConcatSequencePlan,
    *,
    fill_value: object,
) -> tuple[xr.Dataset, xr.DataArray]:
    """Build packed concat output and validity mask from an execution plan.

    Parameters
    ----------
    plan : ConcatSequencePlan
        Resolved runtime context/payload used by this orchestration boundary.
    fill_value : object, optional
        Behavior flag/policy controlling boundary semantics.

    Returns
    -------
    tuple[xr.Dataset, xr.DataArray]
        Tuple of output values produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return _packed_concat_dataset(
        plan.aligned,
        sequence_dim=plan.sequence_dim,
        flat_dim=plan.flat_dim,
        labels=plan.labels,
        lengths=plan.lengths,
        fill_value=fill_value,
        size_name=plan.size_name,
    )


def apply_concat_valid_mask(
    ds: xr.Dataset,
    *,
    valid: xr.DataArray,
    plan: ConcatSequencePlan,
) -> xr.Dataset:
    """Apply concat valid mask and runtime reserved metadata normalization.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    valid : xr.DataArray, optional
        Validity/mask payload used by this operation.
    plan : ConcatSequencePlan, optional
        Resolved runtime context/payload used by this orchestration boundary.

    Returns
    -------
    xr.Dataset
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return _apply_valid_mask(
        ds,
        valid=valid,
        sequence_dim=plan.sequence_dim,
        flat_dim=plan.flat_dim,
        size_name=plan.size_name,
    )


def apply_concat_overlap_sort(
    ds: xr.Dataset,
    *,
    opts: SequenceConcatOptions,
    plan: ConcatSequencePlan,
) -> xr.Dataset:
    """Apply overlap sort path with grouped payload guard and remasking.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    opts : SequenceConcatOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    plan : ConcatSequencePlan, optional
        Resolved runtime context/payload used by this orchestration boundary.

    Returns
    -------
    xr.Dataset
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return _apply_overlap_sort(
        ds,
        overlap=opts.overlap,
        param_name=plan.param_name,
        batch_dims=plan.batch_dims,
        flat_dim=plan.flat_dim,
        sequence_dim=plan.sequence_dim,
        size_name=plan.size_name,
    )


__all__ = [
    "apply_concat_overlap_sort",
    "apply_concat_valid_mask",
    "build_packed_concat_dataset",
]
