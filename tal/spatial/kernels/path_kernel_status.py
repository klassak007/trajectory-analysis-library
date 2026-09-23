from __future__ import annotations

from dataclasses import dataclass

PATH_STATUS_OK = 0
PATH_STATUS_INVALID_QUATERNION = 1
PATH_STATUS_INVALID_ALPHA = 2
PATH_STATUS_INVALID_BRACKET = 3
PATH_STATUS_INVALID_DIRECTION = 4


@dataclass(frozen=True)
class PathKernelFailure:
    edge: int
    row: int
    status: int


def raise_path_kernel_failure(failure: PathKernelFailure) -> None:
    """Raise the stable numerical diagnostic for one fused path failure."""
    descriptions = {
        PATH_STATUS_INVALID_QUATERNION: "quaternion norm must be finite and > 0",
        PATH_STATUS_INVALID_ALPHA: "finite alpha values must be within [0, 1]",
        PATH_STATUS_INVALID_BRACKET: "parameter bracket index is out of range",
        PATH_STATUS_INVALID_DIRECTION: "path direction must be +1 or -1",
    }
    detail = descriptions.get(failure.status, "unknown fused path failure")
    raise ValueError(f"edge {failure.edge}, query {failure.row}: {detail}")


__all__ = [
    "PATH_STATUS_INVALID_ALPHA",
    "PATH_STATUS_INVALID_BRACKET",
    "PATH_STATUS_INVALID_DIRECTION",
    "PATH_STATUS_INVALID_QUATERNION",
    "PATH_STATUS_OK",
    "PathKernelFailure",
    "raise_path_kernel_failure",
]
