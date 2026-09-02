from __future__ import annotations

import numpy as np

from tal.spatial.kernels import pose_kernels, rotation_apply_kernels
from tal.spatial.kernels.scipy_buffers import writable_scipy_vectors


class _WritableVectorRotation:
    @classmethod
    def from_quat(cls, values: np.ndarray) -> _WritableVectorRotation:
        return cls()

    def inv(self) -> _WritableVectorRotation:
        return self

    def apply(self, values: np.ndarray) -> np.ndarray:
        if not values.flags.writeable:
            raise ValueError("Rotation.apply requires a writable vector buffer")
        return values.copy()


def _readonly_rows(values: list[float], *, rows: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(values, dtype=np.float64), (rows, len(values)))


def test_spatial_perf_002_scipy_vector_buffer_copies_only_when_required() -> None:
    """ID: SPATIAL_PERF_002_scipy_vector_buffer_copies_only_when_required."""
    writable = np.asarray([[1.0, 2.0, 3.0]])
    readonly = _readonly_rows([1.0, 2.0, 3.0], rows=2)

    assert writable_scipy_vectors(writable) is writable
    prepared = writable_scipy_vectors(readonly)
    assert prepared.flags.writeable
    assert not np.shares_memory(prepared, readonly)


def test_spatial_perf_003_rotation_kernels_prepare_readonly_broadcast_vectors(
    monkeypatch,
) -> None:
    """ID: SPATIAL_PERF_003_rotation_kernels_prepare_readonly_broadcast_vectors."""
    monkeypatch.setattr(rotation_apply_kernels, "SciRotation", _WritableVectorRotation)
    monkeypatch.setattr(pose_kernels, "SciRotation", _WritableVectorRotation)
    vectors = _readonly_rows([1.0, 2.0, 3.0], rows=2)
    translations = _readonly_rows([0.0, 0.5, 1.0], rows=2)
    quaternions = _readonly_rows([0.0, 0.0, 0.0, 1.0], rows=2)

    rotated = rotation_apply_kernels.rotate_vec3_kernel(vectors, quaternions)
    composed = pose_kernels.compose_translation_kernel(vectors, translations, quaternions)
    inverted = pose_kernels.inverse_translation_kernel(translations, quaternions)

    np.testing.assert_array_equal(rotated, vectors)
    np.testing.assert_array_equal(composed, vectors + translations)
    np.testing.assert_array_equal(inverted, -translations)
