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
retagged = ao.frames.retag(parent="ship", child="drone")
parent, child = retagged.frames.ids()
resolved_parent, resolved_child = retagged.frames.resolve(graph)
drone.rename("drone_0")
renamed = retagged.frames.remap_ids({"drone": "drone_0"})
```

The AO carries frame metadata while the graph owns topology. Resolution is a
read-only lookup; graph mutation and metadata remapping remain explicit,
separate operations.

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

`ao.frames.retag(...)`, `.ids()`, `.remap_ids(...)`, and `.resolve(...)` manage
or inspect frame IDs attached to an AO. `retag(...)` and `remap_ids(...)` are
metadata-only; `resolve(...)` never changes graph topology.

Topology changes, such as reparenting or renaming frames in a graph, require
explicit conflict policies so graph edits remain deterministic.

Spatial kinematics can attach motion and inertial metadata to frame edges when
path solving needs more than topology. That metadata is consumed by spatial
operations, not by the frame graph registry itself.

## What Usually Goes Wrong

- Frames from different graphs cannot be path-solved together.
- Frozen graphs reject mutation.
- `ao.frames.resolve(...)` fails if a present metadata ID is not registered in
  the selected graph.
- Frame ID remaps must be injective.
- Reparent or rename conflicts fail closed unless an explicit replace policy is
  requested.

## Quick Checks

- Inspect `path.nodes` and `steps`.
- Inspect `ascii_tree`.
- Compare `retagged.frames.ids()` and `renamed.frames.ids()` after separate
  graph and metadata edits.

## See Also

- {doc}`spatial`
- {doc}`viewing`
- API: {doc}`../api/frames`

## Bound spatial transforms

Use `tal.spatial.bind_pose(graph, parent, child, pose)` to attach a pose value
or a `(child, parent)` callback to an edge. Transform with
`position.to_frame(destination, graph=graph)` or solve with
`Pose.solve_path_transform(source, destination, graph=graph)`. Rotation paths
use the bound pose's rotation component. See the self-contained
[bound-pose example](../api/types/path_solve.md#minimal-example).

The graph endpoints are the complete edge relation. Each provider frame tag,
when present, must match its corresponding endpoint; the graph supplies omitted
tags. The numerical representation must be in the edge-parent basis. Call
`pose.express_in(parent, graph=graph)` before binding a third-frame value. Lazy
provider payloads are non-owning: keep their owning AO open through dependent
computation. Replacing or removing a binding never closes it.

This differs from `ao.frames.resolve(...)`, which only looks up metadata IDs
and never installs a provider. Registering a pose leaves motion and inertial
declarations unchanged.

## Passive spatial association

Spatial constructors accept frame declarations and an optional graph without
mutating that graph. For example, `Position(data, parent="camera", graph=graph)`
associates a newly constructed position, while
`position.with_graph(other_graph)` returns a distinct associated alias.

`position.graph` is wrapper-local runtime context used by later graph-required
operations when no explicit graph is supplied. It is not serialized into the
Dataset, and ordinary construction or algebra never falls back to the active
graph. Association is separate from `bind_pose(...)`: the former remembers a
graph. For a canonical parent/child Pose, `pose.register()` installs that Pose
as the associated edge provider and returns the same Pose. Use
`bind_pose(...)` for callable or exact unparameterized providers.
