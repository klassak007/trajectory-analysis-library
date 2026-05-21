(api-frames)=
# Frames

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

TAL separates frame metadata from runtime frame topology. AOs can carry
parent/child frame IDs in schema metadata, while `FrameGraph` owns runtime graph
relationships, path finding, snapshots, and optional drawing.

```{contents}
:local:
:depth: 2
```

## Runtime Graph

```python
from tal.frames import Frame, FrameGraph, get_active_frame_graph, get_or_create_frame
```

`FrameGraph` owns `Frame` nodes. It supports context-scoped active graphs,
explicit reparent and rename policies, and optional freezing to prevent further
mutation.

```python
from tal.frames import find_path, fold_path
```

`find_path(src, dst)` returns a canonical `FramePath`. `fold_path(...)` composes
edge payloads along the path using caller-provided callbacks, so the frame
runtime stays payload-agnostic.

When a path traverses an edge opposite its stored direction, `fold_path(...)`
uses the caller-provided `inverse` callback. The frame layer never guesses how
to invert a domain payload.

## Snapshots and Drawing

```python
from tal.frames import (
    FrameGraphDrawOptions,
    draw_frame_graph,
    render_snapshot_ascii,
    snapshot_from_seeds,
    snapshot_to_networkx,
)
```

Snapshots are immutable diagnostics. They are useful for checking graph
structure without mutating graph state. Drawing requires optional `networkx` and
`matplotlib`.

## AO Frame Metadata

```python
ao.frames.ids()
ao.frames.retag(parent="world", child="camera")
ao.frames.remap_ids({"camera": "cam0"})
ao.frames.bind(graph=graph, create_missing=True)
ao.frames.rename_frame("camera", "cam0", graph=graph)
```

Frame IDs are stored under `ds.attrs["tal"]["ext"]["frames"]`. Metadata writes
use the schema writer path. Binding connects those IDs to concrete `Frame`
objects in a runtime graph; it does not numerically transform AO values.

Functional helpers are also available:

```python
from tal.utils.frame_ops import frame_bind, frame_ids, frame_remap_ids, frame_rename, frame_retag
from tal.utils.frame_schema import get_frames, set_frames
```

## Invariants

- Frame IDs are strings or absent.
- Frame ID remaps must be injective.
- Frames from different graphs cannot be path-solved together.
- Frozen graphs reject mutation.
- Graph conflicts require explicit `on_conflict` policies.
- Metadata-only retagging does not transform spatial payload values.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/frames
   :nosignatures:

   tal.frames.FrameGraph
   tal.frames.Frame
   tal.frames.FramePath
   tal.frames.FrameSnapshot
   tal.frames.PathStep
   tal.frames.SnapshotIssue
   tal.frames.find_path
   tal.frames.fold_path
   tal.frames.FrameGraphDrawOptions
   tal.frames.draw_frame_graph
   tal.frames.get_active_frame_graph
   tal.frames.get_or_create_frame
   tal.frames.render_snapshot_ascii
   tal.frames.snapshot_from_seeds
   tal.frames.snapshot_to_networkx
   tal.utils.frame_ops.frame_ids
   tal.utils.frame_ops.frame_retag
   tal.utils.frame_ops.frame_remap_ids
   tal.utils.frame_ops.frame_bind
   tal.utils.frame_ops.frame_rename
   tal.utils.frame_ops.FramesAccessor.ids
   tal.utils.frame_ops.FramesAccessor.retag
   tal.utils.frame_ops.FramesAccessor.remap_ids
   tal.utils.frame_ops.FramesAccessor.bind
   tal.utils.frame_ops.FramesAccessor.rename_frame
```

## See Also

- {doc}`schema`
- User guide: {doc}`../user-guide/frames`
- Spatial types: {doc}`types/index`
