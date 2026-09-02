from __future__ import annotations

from ..core.analysis_object import AnalysisObject
from ..core.dataset_ownership import analysis_object_dataset
from ..core.schema_read import read_roles
from .array import Array


def resolve_binary_output_array_type(
    left: object,
    *,
    fallback: type[Array] = Array,
) -> type[Array]:
    if isinstance(left, Array):
        return left.__class__
    return fallback


def resolve_binary_output_array_type_by_core_arity(
    left: object,
    *,
    output_core_dims: tuple[str, ...],
    fallback: type[Array] = Array,
) -> type[Array]:
    left_cls = resolve_binary_output_array_type(left, fallback=fallback)
    if not isinstance(left, Array):
        return fallback

    builtin_output = _builtin_output_array_type_for_core_arity(output_core_dims)
    if _should_preserve_custom_left_for_arity_route(left, left_cls=left_cls, output_core_dims=output_core_dims):
        return left_cls
    return builtin_output


def _builtin_output_array_type_for_core_arity(output_core_dims: tuple[str, ...]) -> type[Array]:
    from .matrix import Matrix
    from .vector import Vector

    core_arity = len(output_core_dims)
    if core_arity == 2:
        return Matrix
    if core_arity == 1:
        return Vector
    return Array


def _left_declared_core_arity(left: Array) -> int | None:
    try:
        declared, sequence_dim, _, core_dims = read_roles(analysis_object_dataset(left))
    except Exception:  # pragma: no cover - defensive schema-read fallback
        return None
    if not declared:
        return None
    _ = sequence_dim
    return len(core_dims)


def _should_preserve_custom_left_for_arity_route(
    left: Array,
    *,
    left_cls: type[Array],
    output_core_dims: tuple[str, ...],
) -> bool:
    from .matrix import Matrix
    from .vector import Vector
    from .vector3 import Vector3

    builtin_types = (Array, Vector, Vector3, Matrix)
    if left_cls in builtin_types:
        return False
    if issubclass(left_cls, Vector3):
        return False
    if not issubclass(left_cls, (Vector, Matrix)):
        return True
    left_arity = _left_declared_core_arity(left)
    if left_arity is None:
        return False
    return left_arity == len(output_core_dims)


def rewrap_binary_output_array(
    finalized: AnalysisObject,
    *,
    output_cls: type[Array],
) -> Array:
    if type(finalized) is output_cls:
        return finalized
    return output_cls._from_validated(analysis_object_dataset(finalized))


__all__ = [
    "resolve_binary_output_array_type",
    "resolve_binary_output_array_type_by_core_arity",
    "rewrap_binary_output_array",
]
