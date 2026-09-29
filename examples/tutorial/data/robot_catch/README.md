# Robot-catching telemetry fixture

These small, deterministic **simulated** logs support
[the alternate capstone](../../15_capstone_robot_catch.ipynb). They are not
measurements of a real robot or evidence for a controller's performance.

- `trials.csv` explicitly pairs six throws across `predictive` and `reactive`
  controllers. Matching throw numbers have identical camera observations.
- `camera/*.csv` contains ball xyz observations at 100 Hz. Recordings have
  different lengths. A filename stem is the trial label.
- `robot/*.csv` contains the gripper's position and xyzw quaternion relative to
  the fixed camera, at 10 Hz from 0 through 2 seconds. Every pose log covers its
  camera recording. The gripper translates and rotates about the vertical axis.
- Position is in metres; time is in seconds. The camera's z axis is vertical.
  The quaternion maps gripper-frame vectors into camera-frame vectors.

For zero-based throw index `i`, nominal crossing time is `0.9 + 0.06*i`.
The moving reference position is
`[0.5*t + 0.18*i, 0.2*t + 0.2*sin(i), 2.3]`; yaw is
`-0.6 + 0.22*i + 0.8*t` radians. The ball follows a ballistic arc about this
reference with displacement `[2*dt, 0, -1.5*dt - 4.905*dt**2]`, rotated by
the yaw at nominal crossing. Here `dt` is time minus nominal crossing.

Each controller's gripper has a prescribed constant offset from that reference.
Lateral offsets are `[0.025, -0.035, 0.045, -0.055, 0.065, 0.100]` m for
predictive and `[0.040, 0.110, 0.160, 0.200, 0.130, 0.180]` m for reactive;
vertical offsets are `0.01*cos(i)` m. Both offsets use the crossing orientation.
The names label two illustrative error patterns, not implemented control laws.

The notebook independently derives closest sampled distances, eligibility, and
approach events from the logs. It uses a 0.08 m capture sphere and a separate
0.35 m approach sphere. At the default capture radius, 5/6 predictive and 1/6
reactive trials are geometrically eligible. All 12 enter the approach sphere.
These are sampled geometry results, not collision detection or grasp simulation.

To regenerate the CSV files, from the repository root run:

```sh
python examples/tutorial/data/robot_catch/generate.py
```

The generator only writes raw data. It imports no TAL code and performs no
registration, synchronization, event extraction, grouping, or analysis. The
notebook reads the checked-in files directly and needs no helper import.
