from __future__ import annotations

from collections.abc import Sequence

from ...core import assemble_core as _core_assemble_core
from ...core import block_core as _core_block_core
from ...core import stack_core as _core_stack_core
from ...core.dataset_ownership import analysis_object_dataset
from ..array import Array


def _rewrap_array_output(result: "AnalysisObject") -> Array:
    if isinstance(result, Array):
        return result
    return Array._from_validated(analysis_object_dataset(result))


def assemble_core(
    values: object,
    *,
    core_dims: Sequence[str],
    core_labels: Sequence[Sequence[object]] | None = None,
    output_var: str | None = None,
    validate: bool = True,
) -> Array:
    """Assemble nested AO-like layout into new core dimensions.

    Parameters
    ----------
    values : object
        Input values consumed by this operation.
    core_dims : Sequence[str], optional
        Core-dimension labels/shape metadata used for structural operations.
    core_labels : Sequence[Sequence[object]] | None, optional
        Core-dimension labels/shape metadata used for structural operations.
    output_var : str | None, optional
        Output naming metadata used during finalization.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    Array
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    out = _core_assemble_core(
        values,
        core_dims=core_dims,
        core_labels=core_labels,
        output_var=output_var,
        validate=validate,
    )
    return _rewrap_array_output(out)


def stack_core(
    values: object,
    *,
    core_dim: str,
    core_labels: Sequence[object] | None = None,
    output_var: str | None = None,
    validate: bool = True,
) -> Array:
    """Stack AO-like values along one core dimension.

    Parameters
    ----------
    values : object
        Input values consumed by this operation.
    core_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    core_labels : Sequence[object] | None, optional
        Core-dimension labels/shape metadata used for structural operations.
    output_var : str | None, optional
        Output naming metadata used during finalization.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    Array
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    out = _core_stack_core(
        values,
        core_dim=core_dim,
        core_labels=core_labels,
        output_var=output_var,
        validate=validate,
    )
    return _rewrap_array_output(out)


def block_core(
    values: object,
    *,
    row_dim: str,
    col_dim: str,
    row_labels: Sequence[object] | None = None,
    col_labels: Sequence[object] | None = None,
    output_var: str | None = None,
    validate: bool = True,
) -> Array:
    """Assemble nested 2D layout into row/col core dimensions.

    Parameters
    ----------
    values : object
        Input values consumed by this operation.
    row_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    col_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    row_labels : Sequence[object] | None, optional
        Core-dimension labels/shape metadata used for structural operations.
    col_labels : Sequence[object] | None, optional
        Core-dimension labels/shape metadata used for structural operations.
    output_var : str | None, optional
        Output naming metadata used during finalization.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    Array
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    out = _core_block_core(
        values,
        row_dim=row_dim,
        col_dim=col_dim,
        row_labels=row_labels,
        col_labels=col_labels,
        output_var=output_var,
        validate=validate,
    )
    return _rewrap_array_output(out)


__all__ = [
    "assemble_core",
    "block_core",
    "stack_core",
]
