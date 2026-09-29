"""Spatial numerical placeholders for structurally unreachable rows."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import xarray as xr

from tal.core.metadata_optional import canonicalize_optional_names
from tal.core.orchestration.alignment import align_exact_for_plan
from tal.core.orchestration.finalize import transfer_dataset_attrs
from tal.core.orchestration.indexing import dimension_coordinates
from tal.core.orchestration.schema_finalize import (
    CoreSchemaFinalizeSpec,
    stamp_core_schema,
)
from tal.core.orchestration.topology import ResolvedTopologyPlan
from tal.core.param_ops.guards import (
    mark_generated_size_coord,
    mark_reserved_coord,
    reserved_coord_is_owned,
)
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.core.schema_validate.finalize import transfer_dataarray_metadata
from tal.core.validity_finalize import assign_sequence_size_from_valid_mask
from tal.core.validity_mask import resolve_validated_structural_mask_base


def numerical_valid_mask(ds: xr.Dataset) -> xr.DataArray | None:
    """Intersect declared structural validity with operation-owned validity."""
    _, sequence_dim, _, _ = read_roles(ds)
    structural = resolve_validated_structural_mask_base(
        ds, sequence_dim=sequence_dim,
        sequence_size_coord=read_sequence_size_coord_name(ds),
    )
    if "valid" not in ds.coords or not reserved_coord_is_owned(ds, name="valid"):
        return structural
    temporal = ds.coords["valid"].reset_coords(drop=True).astype(bool)
    return temporal if structural is None else temporal & structural


def combined_numerical_mask(
    *datasets: xr.Dataset, topology: ResolvedTopologyPlan, owner: str,
) -> xr.DataArray | None:
    masks = tuple(numerical_valid_mask(ds) for ds in datasets)
    padded_join = topology.batch_join in {"outer", "left", "right"} or (
        topology.primary_key == "sequence" and topology.sequence_join in {"outer", "left", "right"}
    )
    if not any(valid is not None for valid in masks) and not padded_join:
        return None
    _require_output_validity_names(datasets, topology=topology, owner=owner)
    if topology.primary_key != "sequence" or topology.sequence_join != "exact" or topology.batch_join != "exact":
        masks = _align_numerical_masks(masks, topology=topology, owner=owner)
    mask = None
    for valid in masks:
        if valid is not None:
            mask = valid if mask is None else mask & valid
    return None if mask is None else mask.reset_coords(drop=True)


def _require_output_validity_names(
    datasets: tuple[xr.Dataset, ...], *, topology: ResolvedTopologyPlan, owner: str,
) -> None:
    """Protect surviving payload, dimension, and native-index identities."""
    names = {read_sequence_size_coord_name(ds) for ds in datasets} - {None}
    if not names or any("valid" in ds.coords and reserved_coord_is_owned(ds, name="valid") for ds in datasets):
        names.add("valid")
    dimensions = {*topology.batch_dims, *topology.output_core_dims}
    if topology.sequence_dim is not None:
        dimensions.add(topology.sequence_dim)
    protected = {topology.operands[0].data.name, *dimensions}
    for operand in topology.operands:
        protected.update(
            name for _, coordinates in operand.data.xindexes.group_by_index()
            if all(set(variable.dims) <= dimensions for variable in coordinates.values())
            for name in coordinates
        )
    collisions = names & protected
    if collisions:
        raise ValueError(f"{owner}: numerical validity metadata conflicts with protected output names {sorted(collisions)!r}.")


def _align_numerical_masks(
    masks: tuple[xr.DataArray | None, ...], *, topology: ResolvedTopologyPlan, owner: str,
) -> tuple[xr.DataArray, ...]:
    """Apply the payload's selected key/join policy to its declared coverage."""
    operands = tuple(
        replace(operand, data=_mask_operand(valid, operand.data, core=operand.semantic.core_dims),
                semantic=replace(operand.semantic, core_dims=()))
        for valid, operand in zip(masks, topology.operands, strict=True)
    )
    aligned = align_exact_for_plan(replace(topology, operands=operands), owner=owner, what="numerical validity")
    return tuple(valid.fillna(False).astype(bool).reset_coords(drop=True) for valid in aligned)


def _mask_operand(valid: xr.DataArray | None, data: xr.DataArray, *, core: tuple[str, ...]) -> xr.DataArray:
    """Expand partial coverage by shape, retaining the operand's native indexes."""
    if valid is None:
        valid = xr.DataArray(True)
    missing = {dim: size for dim, size in data.sizes.items() if dim not in core and dim not in valid.dims}
    valid = valid.expand_dims(missing)
    coordinates = data.coords.to_dataset().drop_dims(core, errors="ignore").coords
    return valid.assign_coords(coordinates)


def finalize_numerical_result(
    source: xr.Dataset, result: xr.Dataset, valid: xr.DataArray | None,
    *, topology: ResolvedTopologyPlan, other: xr.Dataset, owner: str,
) -> xr.Dataset:
    """Reconcile binary output coverage through the existing validity owners."""
    out = transfer_dataset_attrs(source, result, validate=False)
    _, _, _, core = read_roles(source)
    sequence, batch = topology.sequence_dim, topology.batch_dims
    size_name = read_sequence_size_coord_name(source) or read_sequence_size_coord_name(other)
    coverage = None if valid is None else _broadcast_output_mask(valid, out, sequence=sequence, batch=batch)
    out, size_name = assign_sequence_size_from_valid_mask(
        out, valid=coverage, sequence_dim=sequence, batch_dims=batch, sequence_size_coord=size_name,
    )
    out = _finalize_size_metadata(source, other, out, name=size_name)
    temporal = any(
        "valid" in ds.coords and reserved_coord_is_owned(ds, name="valid") for ds in (source, other)
    )
    if valid is not None and (temporal or size_name is None):
        if "valid" in out.data_vars or "valid" in out.dims or "valid" in out.xindexes:
            raise ValueError(f"{owner}: generated validity conflicts with protected output name 'valid'.")
        out = out.assign_coords(valid=mark_reserved_coord(valid.reset_coords(drop=True), name="valid"))
    param = read_param_coord_name(source) or read_param_coord_name(other)
    out, param, size_name = canonicalize_optional_names(
        out, sequence_dim=sequence, batch_dims=batch, param_name=param, size_name=size_name,
    )
    return stamp_core_schema(out, spec=CoreSchemaFinalizeSpec(
        sequence, batch, core, param, size_name,
    ))


def _finalize_size_metadata(
    source: xr.Dataset, other: xr.Dataset, output: xr.Dataset, *, name: str | None,
) -> xr.Dataset:
    """Preserve unchanged eager size declarations; mark newly computed counts."""
    if name is None:
        return output
    original = source.coords[name] if name in source.coords else other.coords[name]
    size = output.coords[name]
    if original.chunks is None and size.chunks is None and original.variable.equals(size.variable):
        return output.assign_coords({name: transfer_dataarray_metadata(original, size)})
    return mark_generated_size_coord(output, name=name)


def _broadcast_output_mask(
    valid: xr.DataArray, out: xr.Dataset, *, sequence: str | None, batch: tuple[str, ...],
) -> xr.DataArray:
    """Retain each resolved batch axis, even when only one operand has validity."""
    dims = (*batch, sequence) if sequence is not None else batch
    missing = {dim: out.sizes[dim] for dim in dims if dim not in valid.dims}
    return valid.expand_dims(missing).assign_coords(dimension_coordinates(out, dims=tuple(missing)))


def safe_rotation_values(
    values: xr.DataArray,
    *,
    core_dims: tuple[str, ...],
    valid: xr.DataArray | None,
) -> xr.DataArray:
    """Use identity only where the declared domain makes payload unreachable."""
    valid = _payload_mask(values, valid)
    if valid is None:
        return values
    identity = np.array([0.0, 0.0, 0.0, 1.0]) if len(core_dims) == 1 else np.eye(3)
    neutral = xr.DataArray(identity, dims=core_dims, coords={dim: values.coords[dim] for dim in core_dims})
    return values.where(valid, neutral)


def mask_numerical_result(values: xr.DataArray, valid: xr.DataArray | None) -> xr.DataArray:
    """Do not expose internal placeholders as computed samples."""
    valid = _payload_mask(values, valid)
    return values if valid is None else values.where(valid)


def _payload_mask(values: xr.DataArray, valid: xr.DataArray | None) -> xr.DataArray | None:
    """A broadcast-constant payload is reached if any corresponding row is valid."""
    if valid is None:
        return None
    absent = tuple(dim for dim in valid.dims if dim not in values.dims)
    return valid.any(dim=absent) if absent else valid
