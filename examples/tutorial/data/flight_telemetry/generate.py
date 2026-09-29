"""Write deterministic flight logs, without importing TAL or doing analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

OFFSETS = [
    (-0.5, 0.4), (0.5, -0.6), (0.3, 1.2), (-0.4, -1.7), (2.5, 0.3), (2.8, 1.6),
    (0.6, 0.7), (-0.8, -0.8), (0.2, 1.8), (-2.6, -0.5), (0.0, 0.0), (0.0, 0.0),
]


def pose_columns(time, trial, *, drone):
    yaw = 0.3 + 0.04 * trial + 0.05 * time
    ship = np.column_stack((0.8 * time + 0.1 * trial, 0.2 * time, 0.1 * time))
    prefix, position = "ship", ship
    if drone:
        crossing = 8.025 + 0.1 * trial
        dx, dy = OFFSETS[trial]
        x, y = dx + 0.35 * (time - crossing), dy + 0.03 * (time - crossing)
        local_rotated = np.column_stack((np.cos(yaw) * x - np.sin(yaw) * y,
                                        np.sin(yaw) * x + np.cos(yaw) * y, crossing - time))
        prefix, position = "drone", ship + local_rotated
        yaw = yaw + 0.08
    quaternion = np.column_stack((np.zeros_like(time), np.zeros_like(time), np.sin(yaw / 2), np.cos(yaw / 2)))
    columns = [f"{prefix}.position.{axis}" for axis in "xyz"]
    columns += [f"{prefix}.rotation.{axis}" for axis in "xyzw"]
    return pd.DataFrame(np.column_stack((position, quaternion)), columns=columns).assign(time=time)


def write_trial(root, trial):
    label = f"flight_{trial + 1:02d}"
    end = 4.5 + 0.5 * (trial - 10) if trial >= 10 else 10.0 + 0.1 * trial
    time = np.round(np.arange(round(end / 0.05) + 1) * 0.05, 2)
    observations = pose_columns(time, trial, drone=True)
    observations = observations.drop(columns=[f"drone.rotation.{axis}" for axis in "xyzw"])
    observations["altitude"] = observations["drone.position.z"]
    observations.to_csv(root / "observations" / f"{label}.csv", index=False, float_format="%.14g")
    provider_time = np.arange(25) * 0.5
    for kind in ("ship", "drone"):
        pose = pose_columns(provider_time, trial, drone=kind == "drone")
        pose.to_csv(root / f"{kind}_poses" / f"{label}.csv", index=False, float_format="%.14g")
    return {"trial": label, "guidance": "baseline" if trial < 6 else "revised",
            "weight": 1.0 + 0.1 * (trial % 3)}


def main():
    root = Path(__file__).parent
    for folder in ("observations", "ship_poses", "drone_poses", "sensors"):
        (root / folder).mkdir(exist_ok=True)
    pd.DataFrame([write_trial(root, trial) for trial in range(12)]).to_csv(root / "trials.csv", index=False)
    for name, time, bias in (("imu", np.arange(241) * 0.05, 0.0),
                             ("gps", 0.025 + np.arange(60) * 0.2, 0.05)):
        pd.DataFrame({"time": time, "speed": 2.0 + 0.3 * time + bias}).to_csv(
            root / "sensors" / f"{name}.csv", index=False, float_format="%.14g",
        )


if __name__ == "__main__":
    main()
