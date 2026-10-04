from __future__ import annotations

import numpy as np

LOCAL_POLY_BACKEND_OWNER = "spatial.kinematics.temporal.local_poly_backend"


def validate_local_poly_shapes(values: object, param: object, valid: object, *, owner: str) -> None:
    value_shape = np.shape(values)
    param_shape = np.shape(param)
    if len(value_shape) < 2:
        raise ValueError(f"{owner}: values must include sequence and core dimensions.")
    if param_shape != np.shape(valid):
        raise ValueError(f"{owner}: param/valid shapes must match; got {param_shape!r} vs {np.shape(valid)!r}.")
    if value_shape[:-1] != param_shape:
        raise ValueError(
            f"{owner}: values non-core shape must match param shape; got {value_shape[:-1]!r} vs {param_shape!r}."
        )


def validate_local_poly_options(window: int, poly_order: int, *, owner: str) -> tuple[int, int]:
    if isinstance(window, bool) or not isinstance(window, int) or window < 1 or window % 2 == 0:
        raise ValueError(f"{owner}: window must be a positive odd integer.")
    if isinstance(poly_order, bool) or not isinstance(poly_order, int) or poly_order < 1:
        raise ValueError(f"{owner}: poly_order must be a positive integer.")
    if poly_order >= window:
        raise ValueError(f"{owner}: poly_order must be less than window.")
    return int(window), int(poly_order)


__all__ = ["LOCAL_POLY_BACKEND_OWNER", "validate_local_poly_options", "validate_local_poly_shapes"]
