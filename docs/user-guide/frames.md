(frames-transforms)=
# Frames

Frames give names to coordinate systems and relationships. TAL keeps frame IDs
as AO metadata, while `FrameGraph` manages runtime topology for path finding,
diagnostics, and spatial transform resolution.

## Minimal Example

<!-- example-id: UG-FRAMES-BASIC -->
```python
import numpy as np
import xarray as xr
from tal.core import AnalysisObject
from tal.frames import FrameGraph, find_path, fold_path, render_snapshot_ascii, snapshot_from_seeds

with FrameGraph() as graph:
    world = graph.get_or_create_frame("world")
    ship = graph.get_or_create_frame("ship", parent=world)
    drone = graph.get_or_create_frame("drone", parent=ship)

path = find_path(drone, world)
steps = fold_path(
    path,
    edge_value_fn=lambda child, parent: [(child.id, parent.id)],
    compose=lambda acc, value: acc + value,
    inverse=lambda value: [(value[0][1], value[0][0])],
    identity=lambda: [],
)

snapshot = snapshot_from_seeds(("drone",), graph=graph)
ascii_tree = render_snapshot_ascii(snapshot)

ao = AnalysisObject.from_data(
    xr.Dataset({"value": ("sample", np.array([1.0, 2.0]))}, coords={"sample": [0, 1]}),
    sequence_dim="sample",
    core_dims=(),
    validate=True,
)
retagged = ao.frames.retag(parent="world", child="drone")
parent, child = retagged.frames.ids()
bound_parent, bound_child = retagged.frames.bind(graph=graph, create_missing=True)
renamed = retagged.frames.rename_frame("drone", "drone_0", graph=graph)
```

The AO carries only metadata until it is bound or used by a spatial operation.
The graph is the runtime object that owns topology.

Active graph selection is task-local. A synchronous `with graph:` block may
span asyncio `await` points without sharing its context-entry state with sibling
tasks. This context isolation does not make concurrent topology mutation atomic.

## Working With Frame Topology

- Use `FrameGraph.get_or_create_frame(...)` to build topology.
- Use `find_path(src, dst)` to resolve an oriented path.
- Use `fold_path(...)` to compose edge payloads along that path.
- Use `snapshot_from_seeds(...)` and `render_snapshot_ascii(...)` for quick
  diagnostics.
- Use `draw_frame_graph(...)` when optional drawing dependencies are available.

Rendered diagnostics follow the graph's parent-to-child topology. Optional
graph drawing requires the plotting dependencies used by `draw_frame_graph`.

## Working With AO Frame Metadata

`ao.frames.retag(...)`, `.ids()`, `.bind(...)`, `.remap_ids(...)`, and
`.rename_frame(...)` manage frame IDs attached to an AO. `retag(...)` is
metadata-only; it does not transform values.

Topology changes, such as reparenting or renaming frames in a graph, require
explicit conflict policies so graph edits remain deterministic.

Spatial kinematics can attach motion and inertial metadata to frame edges when
path solving needs more than topology. That metadata is consumed by spatial
operations, not by the frame graph registry itself.

## What Usually Goes Wrong

- Frames from different graphs cannot be path-solved together.
- Frozen graphs reject mutation.
- `ao.frames.bind(...)` fails if metadata references missing frames and
  `create_missing=False`.
- Frame ID remaps must be injective.
- Reparent or rename conflicts fail closed unless an explicit replace policy is
  requested.

## Quick Checks

- Inspect `path.nodes` and `steps`.
- Inspect `ascii_tree`.
- Compare `retagged.frames.ids()` and `renamed.frames.ids()` after graph edits.

## See Also

- {doc}`spatial`
- {doc}`viewing`
- API: {doc}`../api/frames`
