from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from tal.core.schema import set_roles
from tal.core.schema_read import read_roles, validate_schema_if_needed
from tal.core.typed_lifecycle import TypedLifecycleContext, TypedLifecycleSpec

_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")


@dataclass(frozen=True)
class ArrayInitOptions:
    core_dims: tuple[str, ...] | None = None


def _coerce_array_init_options(options: object | None, *, owner: str) -> ArrayInitOptions:
    if options is None:
        return ArrayInitOptions()
    if isinstance(options, ArrayInitOptions):
        return options
    raise TypeError(f"{owner}: Array lifecycle options must be ArrayInitOptions or None.")


def declared_roles(ds: xr.Dataset, *, owner: str) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    candidate = validate_schema_if_needed(ds)
    declared, sequence_dim, batch_dims, core_dims = read_roles(candidate)
    if not declared:
        raise ValueError(f"{owner}: Array requires declared roles before setting core dims.")
    return sequence_dim, batch_dims, core_dims


def normalize_core_dims(dims: tuple[object, ...], *, owner: str) -> tuple[str, ...]:
    if not dims:
        raise ValueError(f"{owner}: expected at least one core dim.")
    if any(not isinstance(dim, str) for dim in dims):
        raise TypeError(f"{owner}: core dims must be strings.")
    out = tuple(dims)
    if any(not dim for dim in out):
        raise ValueError(f"{owner}: core dims must be non-empty strings.")
    if len(set(out)) != len(out):
        raise ValueError(f"{owner}: core dims must be unique; got {out!r}.")
    return out


def apply_array_init_options(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
    opts = _coerce_array_init_options(ctx.options, owner=ctx.owner)
    if opts.core_dims is None:
        return ds
    sequence_dim, batch_dims, _ = declared_roles(ds, owner=ctx.owner)
    normalized = normalize_core_dims(tuple(opts.core_dims), owner=ctx.owner)
    return set_roles(
        ds,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=normalized,
        validate=True,
    )


def enforce_array_invariants(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
    _ = (ds, ctx)
    return None


def enforce_vector_invariants(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
    _, _, core_dims = declared_roles(ds, owner=ctx.owner)
    if len(core_dims) != 1:
        raise ValueError(f"{ctx.owner}: Vector requires exactly one core dim; got {core_dims!r}.")


def matrix_core_dims(ds: xr.Dataset, *, owner: str) -> tuple[str, str]:
    _, _, core_dims = declared_roles(ds, owner=owner)
    if len(core_dims) != 2:
        raise ValueError(f"{owner}: Matrix requires exactly two core dims; got {core_dims!r}.")
    row_dim, col_dim = core_dims
    if row_dim == col_dim:
        raise ValueError(f"{owner}: Matrix core dims must be distinct; got {core_dims!r}.")
    return row_dim, col_dim


def enforce_matrix_invariants(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
    _ = matrix_core_dims(ds, owner=ctx.owner)


def enforce_vector3_invariants(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
    _, _, core_dims = declared_roles(ds, owner=ctx.owner)
    if len(core_dims) != 1:
        raise ValueError(f"{ctx.owner}: Vector3 requires exactly one core dim; got {core_dims!r}.")
    axis = core_dims[0]
    if int(ds.sizes.get(axis, -1)) != 3:
        raise ValueError(f"{ctx.owner}: Vector3 core axis {axis!r} must have length 3.")
    if axis not in ds.coords:
        raise ValueError(f"{ctx.owner}: Vector3 core axis {axis!r} must have labels ('x', 'y', 'z').")
    labels = tuple(ds.coords[axis].to_index().tolist())
    if labels != _XYZ_LABELS:
        raise ValueError(f"{ctx.owner}: Vector3 core axis labels must equal {_XYZ_LABELS!r}; got {labels!r}.")


ARRAY_LIFECYCLE = TypedLifecycleSpec(
    type_name="Array",
    owner_prefix="linalg.array",
    apply_init_options=apply_array_init_options,
    enforce=enforce_array_invariants,
)
VECTOR_LIFECYCLE = TypedLifecycleSpec(
    type_name="Vector",
    owner_prefix="linalg.vector",
    apply_init_options=apply_array_init_options,
    enforce=enforce_vector_invariants,
)
MATRIX_LIFECYCLE = TypedLifecycleSpec(
    type_name="Matrix",
    owner_prefix="linalg.matrix",
    apply_init_options=apply_array_init_options,
    enforce=enforce_matrix_invariants,
)
VECTOR3_LIFECYCLE = TypedLifecycleSpec(
    type_name="Vector3",
    owner_prefix="linalg.vector3",
    apply_init_options=apply_array_init_options,
    enforce=enforce_vector3_invariants,
)


__all__ = [
    "ARRAY_LIFECYCLE",
    "MATRIX_LIFECYCLE",
    "VECTOR3_LIFECYCLE",
    "VECTOR_LIFECYCLE",
    "ArrayInitOptions",
    "apply_array_init_options",
    "declared_roles",
    "enforce_array_invariants",
    "enforce_matrix_invariants",
    "enforce_vector3_invariants",
    "enforce_vector_invariants",
    "matrix_core_dims",
    "normalize_core_dims",
]
