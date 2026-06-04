from __future__ import annotations

import numpy as np

from ._fixed_size_constants import STATUS_INVALID_QUAT, STATUS_OK


def normalize_quat_components(x, y, z, w):
    total = x * x + y * y + z * z + w * w
    if not np.isfinite(total) or total <= 0.0:
        return STATUS_INVALID_QUAT, 0.0, 0.0, 0.0, 0.0
    norm = np.sqrt(total)
    return STATUS_OK, x / norm, y / norm, z / norm, w / norm


def normalize_quat_array(values, row):
    x = values[row, 0]
    y = values[row, 1]
    z = values[row, 2]
    w = values[row, 3]
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z) or not np.isfinite(w):
        return STATUS_INVALID_QUAT, 0.0, 0.0, 0.0, 0.0
    total = x * x + y * y + z * z + w * w
    if not np.isfinite(total) or total <= 0.0:
        return STATUS_INVALID_QUAT, 0.0, 0.0, 0.0, 0.0
    norm = np.sqrt(total)
    return STATUS_OK, x / norm, y / norm, z / norm, w / norm


def normalize_quat_row(values, row, idx):
    x = values[row, idx, 0]
    y = values[row, idx, 1]
    z = values[row, idx, 2]
    w = values[row, idx, 3]
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z) or not np.isfinite(w):
        return STATUS_INVALID_QUAT, 0.0, 0.0, 0.0, 0.0
    total = x * x + y * y + z * z + w * w
    if not np.isfinite(total) or total <= 0.0:
        return STATUS_INVALID_QUAT, 0.0, 0.0, 0.0, 0.0
    norm = np.sqrt(total)
    return STATUS_OK, x / norm, y / norm, z / norm, w / norm


def normalize_quat_tuple(quat):
    x, y, z, w = quat
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z) or not np.isfinite(w):
        return STATUS_INVALID_QUAT, 0.0, 0.0, 0.0, 0.0
    total = x * x + y * y + z * z + w * w
    if not np.isfinite(total) or total <= 0.0:
        return STATUS_INVALID_QUAT, 0.0, 0.0, 0.0, 0.0
    norm = np.sqrt(total)
    return STATUS_OK, x / norm, y / norm, z / norm, w / norm


def quat_multiply(right, left):
    rx, ry, rz, rw = right
    lx, ly, lz, lw = left
    return (
        rw * lx + rx * lw + ry * lz - rz * ly,
        rw * ly - rx * lz + ry * lw + rz * lx,
        rw * lz + rx * ly - ry * lx + rz * lw,
        rw * lw - rx * lx - ry * ly - rz * lz,
    )


def rotate_vec3(quat, vector):
    qx, qy, qz, qw = quat
    vx, vy, vz = vector
    uvx = qy * vz - qz * vy
    uvy = qz * vx - qx * vz
    uvz = qx * vy - qy * vx
    uuvx = qy * uvz - qz * uvy
    uuvy = qz * uvx - qx * uvz
    uuvz = qx * uvy - qy * uvx
    return (
        vx + 2.0 * (qw * uvx + uuvx),
        vy + 2.0 * (qw * uvy + uuvy),
        vz + 2.0 * (qw * uvz + uuvz),
    )


def inverse_pose(translation, quat):
    inverse_quat = (-quat[0], -quat[1], -quat[2], quat[3])
    qx, qy, qz, qw = inverse_quat
    vx, vy, vz = translation
    uvx = qy * vz - qz * vy
    uvy = qz * vx - qx * vz
    uvz = qx * vy - qy * vx
    uuvx = qy * uvz - qz * uvy
    uuvy = qz * uvx - qx * uvz
    uuvz = qx * uvy - qy * uvx
    rotated = (
        vx + 2.0 * (qw * uvx + uuvx),
        vy + 2.0 * (qw * uvy + uuvy),
        vz + 2.0 * (qw * uvz + uuvz),
    )
    return (-rotated[0], -rotated[1], -rotated[2]), inverse_quat


def compose_pose(left_t, left_q, right_t, right_q):
    qx, qy, qz, qw = right_q
    vx, vy, vz = left_t
    uvx = qy * vz - qz * vy
    uvy = qz * vx - qx * vz
    uvz = qx * vy - qy * vx
    uuvx = qy * uvz - qz * uvy
    uuvy = qz * uvx - qx * uvz
    uuvz = qx * uvy - qy * uvx
    rotated = (
        vx + 2.0 * (qw * uvx + uuvx),
        vy + 2.0 * (qw * uvy + uuvy),
        vz + 2.0 * (qw * uvz + uuvz),
    )
    out_t = (rotated[0] + right_t[0], rotated[1] + right_t[1], rotated[2] + right_t[2])
    rx, ry, rz, rw = right_q
    lx, ly, lz, lw = left_q
    out_x = rw * lx + rx * lw + ry * lz - rz * ly
    out_y = rw * ly - rx * lz + ry * lw + rz * lx
    out_z = rw * lz + rx * ly - ry * lx + rz * lw
    out_w = rw * lw - rx * lx - ry * ly - rz * lz
    total = out_x * out_x + out_y * out_y + out_z * out_z + out_w * out_w
    norm = np.sqrt(total)
    return out_t, (out_x / norm, out_y / norm, out_z / norm, out_w / norm)


__all__ = [
    "compose_pose",
    "inverse_pose",
    "normalize_quat_array",
    "normalize_quat_components",
    "normalize_quat_row",
    "normalize_quat_tuple",
    "quat_multiply",
    "rotate_vec3",
]
