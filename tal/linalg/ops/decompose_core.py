from __future__ import annotations

from ...core import CoreDecomposeOptions
from ...core import decompose_core as _core_decompose_core
from ..array import Array


def _rewrap_array_output(result: "AnalysisObject") -> Array:
    if isinstance(result, Array) and result.__class__ is Array:
        return result
    return Array._from_validated(result.unsafe_data)


def decompose_core(
    ao: object,
    *,
    opts: CoreDecomposeOptions,
    validate: bool = True,
) -> dict[tuple[object, ...], Array]:
    """Decompose selected core dimensions into keyed array outputs.

    Parameters
    ----------
    ao : object
        AnalysisObject-like input value.
    opts : CoreDecomposeOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    dict[tuple[object, ...], Array]
        Mapping-like result produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    out = _core_decompose_core(ao, opts=opts, validate=validate)
    return {key: _rewrap_array_output(value) for key, value in out.items()}


__all__ = [
    "decompose_core",
]
