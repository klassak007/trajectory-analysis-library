(api-schema)=
# Schema API

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

TAL stores semantic metadata in one canonical location:

```python
ds.attrs["tal"]
```

The schema records roles, parameter coordinates, validity metadata, and
extension namespaces such as frames and components. TAL v3 is a clean schema;
there is no legacy compatibility mode.

```{contents}
:local:
:depth: 2
```

## Canonical Shape

```yaml
tal:
  version: 1
  core:
    roles:
      sequence_dim: "<dim>"
      batch_dims: ["<dim>", "..."]
      core_dims: ["<dim>", "..."]
    param_coord:
      name: "<coord>"
    validity:
      sequence_size_coord: "<coord>"
      layout: "left_packed"
  ext:
    frames:
      parent: "<frame_id>"
      child: "<frame_id>"
    <namespace>:
      ...
```

`roles`, `param_coord`, `validity`, and `ext` blocks are present only when the
corresponding semantics are declared.

## Writer and Validator API

```python
set_roles(ds, *, sequence_dim=UNSET, batch_dims=UNSET, core_dims=UNSET, validate=True)
set_param_coord(ds, *, name, validate=True)
set_validity(ds, *, sequence_size_coord, layout="left_packed", validate=True)
merge_schema(ds, patch, *, validate=True)
validate_schema(ds)
```

All writer APIs are atomic candidate-update operations. On failure they raise
`SchemaError` and do not partially mutate the caller dataset.

## Role Semantics

- `core_dims` is required when roles are declared and may be empty.
- `sequence_dim` is optional.
- `batch_dims` is optional.
- All declared role dimensions must exist in the dataset.
- Role dimensions cannot overlap.

`set_roles(...)` is a partial update. Omitted fields preserve existing values;
provided fields replace values; explicit empty `batch_dims` or `core_dims`
clears that role.

## Parameter and Validity Semantics

`param_coord` requires `sequence_dim`. Its coordinate dims must be either:

- `(sequence_dim,)`
- `(*batch_dims, sequence_dim)`

`validity` also requires `sequence_dim`. A `sequence_size_coord` must have dims
equal to `batch_dims`, or be scalar when there are no batch dims. TAL currently
uses the `left_packed` validity layout.

## Merge Semantics

`merge_schema(ds, patch, ...)` expects `patch` to be the TAL payload itself, not
a wrapper with a top-level `"tal"` key.

- mappings merge recursively
- `None` deletes keys
- lists replace rather than merge

## Frame Metadata

Frame IDs live under `tal.ext.frames`:

```yaml
ext:
  frames:
    parent: world
    child: camera
```

Use `tal.utils.frame_schema.get_frames(...)` and
`tal.utils.frame_schema.set_frames(...)` rather than editing the metadata
payload directly.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/schema
   :nosignatures:

   tal.core.set_roles
   tal.core.set_param_coord
   tal.core.set_validity
   tal.core.merge_schema
   tal.core.validate_schema
   tal.utils.frame_schema.get_frames
   tal.utils.frame_schema.set_frames
```

## See Also

- {doc}`analysis-object`
- {doc}`components`
- {doc}`frames`
