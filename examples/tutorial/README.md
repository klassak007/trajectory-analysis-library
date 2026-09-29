# TAL tutorials

These notebooks use short, visible TAL workflows to answer practical questions.
Run each notebook independently, from top to bottom. Notebooks 01–14 use deterministic
simulated flight telemetry; 15 is the unchanged robot-catching alternative.

## Launch this checkout

From the repository root, activate the intended environment and install the checkout:

```sh
conda activate tal
python -m pip install -e '.[notebooks]'
python -m ipykernel install --user --name tal --display-name 'Python (tal)'
python -m jupyterlab examples/tutorial
```

In a worktree, an existing editable install may point to a different checkout. To avoid
changing that install, launch with an explicit import root instead:

```sh
conda activate tal
PYTHONPATH="$PWD" python -m jupyterlab examples/tutorial
```

Select **Python (tal)**. In a scratch cell, verify both the executable and the source:

```python
import sys
import tal
print(sys.executable)
print(tal.__file__)
```

`tal.__file__` must point to the `tal/__init__.py` inside the checkout you intend to run.
Restart an existing kernel after changing installs or launch settings. The notebooks
support a working directory of either the repository root or `examples/tutorial`;
they do not alter `sys.path`. Dependencies come from the existing `notebooks` extra:
Jupyter, Matplotlib, NetworkX, HoloViews, hvPlot, and their dependencies. Dask and Zarr
are existing core dependencies; no ROS, geo, or astro installation is needed.

## Learning path

| Notebook | Practical question |
| --- | --- |
| [01 Quickstart](01_quickstart_trajectory_workflow.ipynb) | Which flights got close to the deck? |
| [02 Construction](02_construction_and_semantics.ipynb) | What do fields, roles, validity, and ownership declare? |
| [03 Selection](03_indexing_and_selection.ipynb) | Do I need positions, labels, recorded samples, or interpolation? |
| [04 Synchronization](04_timebase_resample_synchronize.ipynb) | Do asynchronously sampled sensors agree? |
| [05 Events](05_conditions_events_and_windows.ipynb) | When did each approach get low? |
| [06 Grouping](06_groupby_concat_and_ragged.ipynb) | How do flight scenarios compare? |
| [07 Linear algebra](07_linalg_array_matrix_vector.ipynb) | How do I calibrate labeled sensor channels? |
| [08 Spatial objects](08_spatial_position_rotation_pose.ipynb) | How do fields become positions, orientations, and Poses? |
| [09 Kinematics](09_kinematics_velocity_acceleration.ipynb) | Can I recover known motion by differentiating and integrating? |
| [10 Topology](10_frames_and_topology.ipynb) | Which frame path connects two sensors? |
| [11 Frame analysis](11_framegraph_and_spatial_types.ipynb) | Am I changing a relation or just its coordinate basis? |
| [12 Lazy persistence](12_lazy_data_and_persistence.ipynb) | When is data read, computed, persisted, and closed? |
| [13 Visualization](13_visualization_holoviews_explorer.ipynb) | Which view answers my question without overplotting? |
| [14 Landing](14_capstone_autonomous_landing.ipynb) | Would a wider landing corridor recover the observed misses? |
| [15 Robot catch](15_capstone_robot_catch.ipynb) | Why did the robot miss? |

Start with 01–04; use 05–13 as focused references. The landing investigation combines
these concepts. The robot-catching notebook remains a shorter contrasting capstone.

## Data and interpretation

See [flight telemetry provenance](data/flight_telemetry/README.md) and
[robot-catching provenance](data/robot_catch/README.md). Both are simulations, not
recorded experiments. Run the flight generator with
`python examples/tutorial/data/flight_telemetry/generate.py` from the repository root.
It writes only deterministic raw logs and catalog fields, with no TAL operations.

Timestamps are seconds; position is in metres; quaternions use x, y, z, w order.
These are declared fixture conventions, not unit inference or conversion by TAL.
The capstone's threshold crossing is sampled geometry, not contact dynamics.

Saved outputs show the default analysis. Widget callbacks may need a live notebook
kernel. Zarr output uses temporary directories and is removed before notebook 12 ends.
See [verification and known library findings](REVIEW.md) for current limitations.
