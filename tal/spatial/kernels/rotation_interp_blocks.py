from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

SLERP_BLOCK_SIZE = 65_536


@dataclass(frozen=True)
class SlerpBlock:
    """Bounded endpoint rows in public batch/query order."""

    start: int
    left: np.ndarray
    right: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray


def validate_slerp_shapes(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray, valid: np.ndarray, *, owner: str) -> None:
    if q0.shape != q1.shape:
        raise ValueError(f"{owner}: q0 and q1 must share shape.")
    if not q0.shape or q0.shape[-1] != 4:
        raise ValueError(f"{owner}: expected trailing quaternion dim length 4.")
    if alpha.shape != valid.shape:
        raise ValueError(f"{owner}: alpha and valid must share shape.")
    if q0.shape[:-1] != alpha.shape:
        raise ValueError(f"{owner}: alpha/valid must match q0/q1 non-core shape.")


def iter_slerp_blocks(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray, valid: np.ndarray) -> Iterator[SlerpBlock]:
    """Copy only bounded slices, including when input flattening needs a copy."""
    for start in range(0, alpha.size, SLERP_BLOCK_SIZE):
        stop = min(start + SLERP_BLOCK_SIZE, alpha.size)
        yield SlerpBlock(
            start,
            _flat_slice(q0, 4 * start, 4 * stop).reshape(-1, 4),
            _flat_slice(q1, 4 * start, 4 * stop).reshape(-1, 4),
            _flat_slice(alpha, start, stop),
            _flat_slice(valid, start, stop),
        )


def _flat_slice(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    if values.flags.c_contiguous:
        return values.reshape(-1)[start:stop]
    return values.flat[start:stop]


__all__ = ["SLERP_BLOCK_SIZE", "SlerpBlock", "iter_slerp_blocks", "validate_slerp_shapes"]
