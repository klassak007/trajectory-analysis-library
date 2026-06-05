from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from tal.utils.block_rows import BlockInputSpec, prepare_block_rows


@dataclass(frozen=True)
class ScanAxisSpec:
    """Describe one explicit ordered scan axis.

    Parameters
    ----------
    name
        Caller-owned label for diagnostics and documentation.
    kind
        Mechanical axis kind: ``"scan"``, ``"topology"``, or ``"core_scan"``.
    """

    name: str
    kind: Literal["scan", "topology", "core_scan"]


@dataclass(frozen=True)
class ScanInputSpec:
    """Describe one input block with ordered and core dimensions.

    Parameters
    ----------
    name
        Caller-owned diagnostic name for the block.
    ordered_ndim
        Number of trailing ordered dimensions before the core dimensions.
    core_ndim
        Number of trailing core dimensions after the ordered dimensions.
    dtype
        Optional NumPy dtype used when coercing the input block.
    """

    name: str
    ordered_ndim: int
    core_ndim: int
    dtype: object | None = None


@dataclass(frozen=True)
class ScanRows:
    """Prepared row arrays and explicit ordered-axis shape metadata.

    Parameters
    ----------
    row_arrays
        Broadcast input arrays reshaped with a leading row dimension.
    outer_shape
        Broadcast shape before the ordered dimensions.
    ordered_shape
        Shared shape of the explicit ordered axes.
    core_shapes
        Per-input core shapes after the ordered dimensions.
    output_shapes
        Per-output shapes built from ``outer_shape``, ``ordered_shape``, and
        the requested output core shapes.
    """

    row_arrays: tuple[np.ndarray, ...]
    outer_shape: tuple[int, ...]
    ordered_shape: tuple[int, ...]
    core_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]


def _as_tuple_shapes(shapes: Sequence[tuple[int, ...]] | None) -> tuple[tuple[int, ...], ...] | None:
    if shapes is None:
        return None
    return tuple(tuple(int(size) for size in shape) for shape in shapes)


def _validate_axes(axes: tuple[ScanAxisSpec, ...], *, owner: str) -> None:
    if not axes:
        raise ValueError(f"{owner}: ordered_axes must not be empty.")
    valid_kinds = {"scan", "topology", "core_scan"}
    for axis in axes:
        if axis.kind not in valid_kinds:
            raise ValueError(f"{owner}: scan axis {axis.name!r} has unsupported kind {axis.kind!r}.")


def _validate_axis_name(name: object, *, owner: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{owner}: topology_axis must be a non-empty string.")
    return name


def _validate_specs(specs: tuple[ScanInputSpec, ...], axes: tuple[ScanAxisSpec, ...], *, owner: str) -> None:
    for spec in specs:
        if spec.ordered_ndim < 0 or spec.core_ndim < 0:
            raise ValueError(f"{owner}: block {spec.name!r} ordered_ndim and core_ndim must be >= 0.")
        if spec.ordered_ndim != len(axes):
            raise ValueError(f"{owner}: block {spec.name!r} ordered_ndim must match ordered_axes.")


def _ordered_shape(array: np.ndarray, spec: ScanInputSpec) -> tuple[int, ...]:
    end = array.ndim - spec.core_ndim if spec.core_ndim else array.ndim
    start = end - spec.ordered_ndim
    return tuple(int(size) for size in array.shape[start:end])


def _core_shape(array: np.ndarray, spec: ScanInputSpec) -> tuple[int, ...]:
    if spec.core_ndim == 0:
        return ()
    return tuple(int(size) for size in array.shape[-spec.core_ndim :])


def _validate_ordered_shapes(shapes: tuple[tuple[int, ...], ...], *, owner: str) -> tuple[int, ...]:
    ordered = shapes[0]
    for shape in shapes[1:]:
        if shape != ordered:
            raise ValueError(f"{owner}: ordered axes must have matching lengths across blocks.")
    return ordered


def prepare_scan_rows(
    blocks: Sequence[object],
    specs: Sequence[ScanInputSpec],
    *,
    ordered_axes: Sequence[ScanAxisSpec],
    output_core_shapes: Sequence[tuple[int, ...]],
    owner: str,
    broadcast_shapes: Sequence[tuple[int, ...]] | None = None,
) -> ScanRows:
    """Prepare blocks for row-local scans over explicit ordered axes.

    Parameters
    ----------
    blocks
        Input array-like blocks with trailing ordered and core dimensions.
    specs
        One ``ScanInputSpec`` per block.
    ordered_axes
        Explicit ordered axes shared by every block.
    output_core_shapes
        Output core shapes appended after the ordered axes.
    owner
        Error-message prefix for the caller-owned boundary.
    broadcast_shapes
        Optional additional outer shapes that participate in broadcasting.

    Returns
    -------
    ScanRows
        Row arrays plus outer, ordered, core, and output shape metadata.

    Raises
    ------
    ValueError
        If ordered axes are empty or inconsistent, block/spec counts mismatch,
        ranks are invalid, dtype coercion fails, or shapes cannot broadcast.

    Examples
    --------
    >>> import numpy as np
    >>> from tal.utils import numba as tal_numba
    >>> rows = tal_numba.prepare_scan_rows(
    ...     (np.zeros((2, 5, 3)),),
    ...     (tal_numba.ScanInputSpec("values", 1, 1, np.float64),),
    ...     ordered_axes=(tal_numba.ScanAxisSpec("sequence", "scan"),),
    ...     output_core_shapes=((3,),),
    ...     owner="docs",
    ... )
    >>> rows.outer_shape, rows.ordered_shape, rows.core_shapes
    ((2,), (5,), ((3,),))
    """

    block_tuple = tuple(blocks)
    spec_tuple = tuple(specs)
    axis_tuple = tuple(ordered_axes)
    _validate_axes(axis_tuple, owner=owner)
    if not block_tuple and not spec_tuple:
        raise ValueError(f"{owner}: scan inputs must not be empty.")
    _validate_specs(spec_tuple, axis_tuple, owner=owner)
    block_specs = tuple(
        BlockInputSpec(spec.name, spec.ordered_ndim + spec.core_ndim, spec.dtype) for spec in spec_tuple
    )
    prepared = prepare_block_rows(
        block_tuple,
        block_specs,
        output_core_shape=(),
        owner=owner,
        broadcast_shapes=_as_tuple_shapes(broadcast_shapes),
    )
    ordered_shapes = tuple(_ordered_shape(array, spec) for array, spec in zip(prepared.row_arrays, spec_tuple))
    ordered = _validate_ordered_shapes(ordered_shapes, owner=owner)
    core_shapes = tuple(_core_shape(array, spec) for array, spec in zip(prepared.row_arrays, spec_tuple))
    output_shapes = tuple(
        prepared.outer_shape + ordered + tuple(int(size) for size in core_shape)
        for core_shape in output_core_shapes
    )
    return ScanRows(prepared.row_arrays, prepared.outer_shape, ordered, core_shapes, output_shapes)


def prepare_topology_rows(
    blocks: Sequence[object],
    specs: Sequence[ScanInputSpec],
    *,
    topology_axis: str,
    output_core_shapes: Sequence[tuple[int, ...]],
    owner: str,
    broadcast_shapes: Sequence[tuple[int, ...]] | None = None,
) -> ScanRows:
    """Prepare blocks for row-local work over one topology axis.

    Parameters
    ----------
    blocks
        Input array-like blocks with one trailing topology axis before any core
        dimensions.
    specs
        One ``ScanInputSpec`` per block. Each spec must use ``ordered_ndim=1``.
    topology_axis
        Explicit caller-owned topology-axis name.
    output_core_shapes
        Output core shapes appended after the topology axis.
    owner
        Error-message prefix for the caller-owned boundary.
    broadcast_shapes
        Optional additional outer shapes that participate in broadcasting.

    Returns
    -------
    ScanRows
        Row arrays plus outer, topology, core, and output shape metadata.

    Raises
    ------
    ValueError
        If ``topology_axis`` is empty, block/spec counts mismatch, ranks are
        invalid, dtype coercion fails, or shapes cannot broadcast.

    Examples
    --------
    >>> import numpy as np
    >>> from tal.utils import numba as tal_numba
    >>> rows = tal_numba.prepare_topology_rows(
    ...     (np.zeros((2, 4, 3)),),
    ...     (tal_numba.ScanInputSpec("links", 1, 1, np.float64),),
    ...     topology_axis="chain",
    ...     output_core_shapes=((3,),),
    ...     owner="docs",
    ... )
    >>> rows.outer_shape, rows.ordered_shape
    ((2,), (4,))
    """

    axis_name = _validate_axis_name(topology_axis, owner=owner)
    return prepare_scan_rows(
        blocks,
        specs,
        ordered_axes=(ScanAxisSpec(axis_name, "topology"),),
        output_core_shapes=output_core_shapes,
        owner=owner,
        broadcast_shapes=broadcast_shapes,
    )


__all__ = ["ScanAxisSpec", "ScanInputSpec", "ScanRows", "prepare_scan_rows", "prepare_topology_rows"]
