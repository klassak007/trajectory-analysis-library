from __future__ import annotations

from collections.abc import Sequence

from ...core import CoreConcatOptions
from ...core import concat_core as _core_concat_core
from ..array import Array


def _first_input_array_type(values: Sequence[object]) -> type[Array] | None:
    if not values:
        return None
    first = values[0]
    if isinstance(first, Array):
        return first.__class__
    return None


def _rewrap_array_output(
    result: "AnalysisObject",
    *,
    input_values: Sequence[object],
) -> Array:
    if isinstance(result, Array):
        return result
    first_type = _first_input_array_type(input_values)
    if first_type is not None:
        return first_type._from_validated(result.unsafe_data)
    return Array._from_validated(result.unsafe_data)


def concat_core(
    aos: Sequence[object],
    *,
    opts: CoreConcatOptions | None = None,
    validate: bool = True,
) -> Array:
    """Concatenate AO-like values along an existing core dimension.

    Parameters
    ----------
    aos : Sequence[object]
        AO-like inputs consumed by this orchestration boundary.
    opts : CoreConcatOptions | None, optional
        Optional options controlling policy and numeric behavior for this operation.
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
    out = _core_concat_core(
        aos,
        opts=opts,
        validate=validate,
    )
    return _rewrap_array_output(out, input_values=aos)


__all__ = [
    "concat_core",
]
