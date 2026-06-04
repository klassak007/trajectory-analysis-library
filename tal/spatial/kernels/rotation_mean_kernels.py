from __future__ import annotations

import numpy as np

_QUAT_SIZE = 4


def quat_mean_kernel(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if array.shape[-1] != _QUAT_SIZE:
        raise ValueError("rotation mean kernel requires final quaternion axis length 4.")
    n = int(array.shape[-2])
    batch = array.shape[:-2]
    target_weight_shape = batch + (n,)
    try:
        w = np.broadcast_to(weight, target_weight_shape)
    except ValueError as exc:
        raise ValueError(
            "spatial.rotation.mean: quaternion mean kernel requires weights broadcastable to "
            f"shape {target_weight_shape!r}, got {weight.shape!r}."
        ) from exc
    flat_q = array.reshape((-1, n, _QUAT_SIZE))
    flat_w = w.reshape((-1, n))
    out = np.full((flat_q.shape[0], _QUAT_SIZE), np.nan, dtype=np.float64)
    for index in range(flat_q.shape[0]):
        q_row = flat_q[index]
        w_row = flat_w[index]
        valid = np.isfinite(w_row) & np.all(np.isfinite(q_row), axis=1)
        if not np.any(valid):
            continue
        qv = q_row[valid]
        wv = w_row[valid]
        total = float(wv.sum())
        if not np.isfinite(total) or total <= 0:
            continue
        gram = (qv[:, :, None] * qv[:, None, :] * wv[:, None, None]).sum(axis=0)
        eigvals, eigvecs = np.linalg.eigh(gram)
        quat = eigvecs[:, int(np.argmax(eigvals))]
        norm = float(np.linalg.norm(quat))
        if norm <= 0 or not np.isfinite(norm):
            continue
        quat = quat / norm
        if quat[3] < 0:
            quat = -quat
        out[index] = quat
    return out.reshape(batch + (_QUAT_SIZE,))


__all__ = ["quat_mean_kernel"]
