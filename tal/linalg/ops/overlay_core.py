from __future__ import annotations

from collections.abc import Sequence

from ...core import CoreOverlayOptions
from ...core import overlay_core as _core_overlay_core
from ..array import Array


def _base_array_type(base: object) -> type[Array] | None:
    if isinstance(base, Array):
        return base.__class__
    return None


def _rewrap_overlay_output(result: "AnalysisObject", *, base: object) -> Array:
    if isinstance(result, Array):
        return result
    base_type = _base_array_type(base)
    if base_type is not None:
        try:
            return base_type._from_validated(result.unsafe_data)
        except (TypeError, ValueError):
            pass
    return Array._from_validated(result.unsafe_data)


def overlay_core(
    base: object,
    patches: Sequence[object] | object,
    *,
    opts: CoreOverlayOptions,
    validate: bool = True,
) -> Array:
    """Overlay patches along one core dimension of a base array.

    Parameters
    ----------
    base : object
        Input dataset/source value processed by this operation.
    patches : Sequence[object] | object
        Operand/component input consumed by this operation.
    opts : CoreOverlayOptions, optional
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
    out = _core_overlay_core(base, patches, opts=opts, validate=validate)
    return _rewrap_overlay_output(out, base=base)


__all__ = [
    "overlay_core",
]
