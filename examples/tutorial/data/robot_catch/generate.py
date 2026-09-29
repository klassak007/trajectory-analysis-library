"""Regenerate the deterministic raw logs; no TAL analysis lives in this file."""

from pathlib import Path

import numpy as np
import pandas as pd


def reference_motion(time, throw):
    """Moving robot reference point and yaw in the fixed camera frame."""
    position = np.column_stack(
        (0.5 * time + 0.18 * throw, 0.2 * time + 0.2 * np.sin(throw), np.full_like(time, 2.3))
    )
    yaw = -0.6 + 0.22 * throw + 0.8 * time
    return position, yaw


def camera_rotation(angle):
    """Planar rotation; the camera's z axis is vertical."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ball_log(throw):
    crossing = 0.9 + 0.06 * throw
    time = np.round(np.arange(0.35, crossing + 0.26 + 0.02 * (throw % 3), 0.01), 2)
    center, _ = reference_motion(time, throw)
    angle = -0.6 + 0.22 * throw + 0.8 * crossing
    dt = time - crossing
    flight = np.column_stack((2.0 * dt, np.zeros_like(dt), -1.5 * dt - 4.905 * dt**2))
    values = center + flight @ camera_rotation(angle).T
    return pd.DataFrame(values, columns=[f"ball.position.{axis}" for axis in "xyz"]).assign(time=time)


def gripper_log(throw, lateral_error):
    time = np.round(np.arange(21) / 10, 2)
    position, yaw = reference_motion(time, throw)
    crossing = 0.9 + 0.06 * throw
    angle = -0.6 + 0.22 * throw + 0.8 * crossing
    offset = camera_rotation(angle) @ [0.0, lateral_error, 0.01 * np.cos(throw)]
    position -= offset
    quat = np.column_stack((np.zeros_like(time), np.zeros_like(time), np.sin(yaw / 2), np.cos(yaw / 2)))
    values = np.column_stack((position, quat))
    columns = [f"gripper.position.{axis}" for axis in "xyz"]
    columns += [f"gripper.rotation.{axis}" for axis in "xyzw"]
    return pd.DataFrame(values, columns=columns).assign(time=time)


def main():
    root = Path(__file__).parent
    (root / "camera").mkdir(exist_ok=True)
    (root / "robot").mkdir(exist_ok=True)
    errors = {
        "predictive": [0.025, -0.035, 0.045, -0.055, 0.065, 0.100],
        "reactive": [0.040, 0.110, 0.160, 0.200, 0.130, 0.180],
    }
    trials = []
    for controller, offsets in errors.items():
        for throw, offset in enumerate(offsets):
            trial = f"{controller}_{throw + 1:02d}"
            trials.append({"trial": trial, "controller": controller, "throw": throw + 1})
            ball_log(throw).to_csv(root / "camera" / f"{trial}.csv", index=False, float_format="%.12g")
            gripper_log(throw, offset).to_csv(root / "robot" / f"{trial}.csv", index=False, float_format="%.12g")
    pd.DataFrame(trials).to_csv(root / "trials.csv", index=False)


if __name__ == "__main__":
    main()
