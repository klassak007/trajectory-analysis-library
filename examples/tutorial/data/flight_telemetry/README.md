# Simulated flight telemetry

This fixture is deterministic, synthetic, and independent of TAL. `generate.py` uses
only NumPy/pandas to write raw logs; construction, layouts, recipes, frame registration,
and analysis remain visible in the notebooks. No external data or random seed is needed.

- `trials.csv`: twelve trial IDs, prescribed baseline/revised scenario labels, and
  illustrative importance weights. No success or failure labels are stored.
- `observations/`: 20 Hz drone positions in world coordinates and world altitude.
  Ten recordings end at 10.0–10.9 s; two end at 4.5 and 5.0 s. TAL's CSV reader
  creates the ragged representation and validity declarations.
- `ship_poses/` and `drone_poses/`: full-coverage 2 Hz position and quaternion logs
  from 0 through 12 s. Provider coverage is independent of observation duration.
- `sensors/`: a 20 Hz IMU estimate and a 5 Hz GPS estimate starting 25 ms later.
  Both already share a clock origin. GPS has a prescribed +0.05 m/s bias.

For zero-based trial `i`, ship position is `(0.8t + 0.1i, 0.2t, 0.1t)` and yaw is
`0.3 + 0.04i + 0.05t` radians. Drone height relative to the deck is
`8.025 + 0.1i - t`. Horizontal motion uses the explicitly listed offsets in the
generator; ship yaw rotates those local offsets into world coordinates. Drone yaw
adds 0.08 radians to ship yaw. All times are seconds and distances metres.

The nominal zero-height instant lies between observation samples. Tutorial 14 uses
the **first observed transition** to height ≤ 0 and derives lateral/longitudinal
categories from that sample. Two recordings stop before any crossing is observed.
The fixture includes in-bounds crossings and each miss category, but the notebooks
must derive them from observations. Prescribed guidance labels are not inferred
outcomes, and changing a corridor does not change the simulated trajectories.

Regenerate from either supported working directory:

```sh
# Repository root
python examples/tutorial/data/flight_telemetry/generate.py
```

```sh
# examples/tutorial
python data/flight_telemetry/generate.py
```
