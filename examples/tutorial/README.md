# TAL tutorials

These notebooks teach TAL through small worked comparisons, visible output topology,
and plots, then apply the concepts to deterministic simulated logs. Every variation
is fully worked out; there are no user exercises. Run each notebook independently,
from top to bottom. Notebook 01 gives a quick result; 02–13 teach specific topics;
14–15 combine them into investigations.

## Launch this checkout

From the repository root, activate the intended environment and install the checkout:

```sh
conda activate tal
python -m pip install -e '.[notebooks]'
python -m ipykernel install --user --name tal --display-name 'Python (tal)'
PYTHONPATH="$PWD" python -m jupyterlab examples/tutorial
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

| Notebook | Topics |
| --- | --- |
| [01 Quickstart](01_quickstart_trajectory_workflow.ipynb) | Loading, inspecting, selecting, reducing, and evaluating trajectories |
| [02 Construction](02_construction_and_semantics.ipynb) | Roles, parameters, validity, layout specifications, and ownership |
| [03 Selection and arithmetic](03_indexing_and_selection.ipynb) | Positions, labels, per-run axis/auxiliary queries, unindexed N-D grids, lazy typed labels, masks, alignment, and broadcasting |
| [04 Parameter operations](04_timebase_resample_synchronize.ipynb) | Evaluation, resampling, synchronization, active domains, and datetime queries |
| [05 Conditions and events](05_conditions_events_and_windows.ipynb) | Conditions, lazy AO reuse, boundaries, intervals, packed clocks, selected layouts, native anchors, empty windows, and invalid-anchor packing |
| [06 Ragged statistics](06_groupby_concat_and_ragged.ipynb) | Reductions, populations, grouping, bins, weights, concatenation, and merge |
| [07 Linear algebra](07_linalg_array_matrix_vector.ipynb) | Labeled arrays, vectors, matrices, core assembly, contraction, and systems |
| [08 Spatial types](08_spatial_position_rotation_pose.ipynb) | Field recipes, representations, components, transform algebra, and interpolation |
| [09 Kinematics](09_kinematics_velocity_acceleration.ipynb) | Differentiation, irregular sampling, smoothing, integration, and combined motion |
| [10 Frame graphs](10_frames_and_topology.ipynb) | Topology, frame metadata, oriented paths, folds, snapshots, and remaps |
| [11 Frame-aware operations](11_framegraph_and_spatial_types.ipynb) | Association, registration, relation/basis changes, providers, and kinematic support |
| [12 Lazy execution and storage](12_lazy_data_and_persistence.ipynb) | Planning, compute/persist, Zarr, CSV, graph context, and resource lifetime |
| [13 Visualization](13_visualization_holoviews_explorer.ipynb) | Coordinates, validity, overlays, selectors, grouping, components, and exploration |
| [14 Landing capstone](14_capstone_autonomous_landing.ipynb) | Moving frames, sampled crossings, corridor geometry, and width/length sensitivity |
| [15 Robot-catching capstone](15_capstone_robot_catch.ipynb) | Relative distance, paired/grouped outcomes, anchor populations, and radius sensitivity |

Start with 01–04; use 05–13 as focused references. Each notebook states its topics,
learning outcomes, and prerequisites. Read the setup and purpose before executing a
cell, then compare its result with the interpretation that follows. Foundational
examples precede advanced worked variations and recorded-log applications.
Spatial field recipes are concentrated in 08; generic graph topology in 10
is separate from spatial providers and physical support in 11.

The two capstones use different geometry and comparison populations. Their plots and
worked sensitivities interpret results produced through visible TAL operations.

## Reading the figures

Figures connect inputs to results: sample/component boxes explain roles, sampling rugs
expose clock differences, matched panels explain transformations, and paired dots
compare the same trial identities. Filled observation markers and open estimate markers
are distinguished where those meanings matter. Padding uses gray shading or hatching;
missing observations remain explicit. Figure captions explain local color meanings.

Notebook 02 introduces each layout field before wrapping data. Notebook 04 uses a
curved synthetic signal to reveal interpolation error, then a linear log fixture as
an exact control. Notebook 09 pairs exact polynomial controls with a nonlinear,
irregularly sampled path. The capstones show representative geometry before population
summaries, retaining unobserved outcomes and matched throw identities.

The repository-local `presentation.py` only supplies plot defaults and display cleanup.
TAL calls, data construction, and independent checks remain visible in the notebooks.
The explicit import root in the launch commands makes this module available from both
supported working directories; the notebooks do not change Python's search path.

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
