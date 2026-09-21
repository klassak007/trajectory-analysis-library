(api-field-recipes)=
# `SpatialFieldRecipe`

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

`tal.spatial.SpatialFieldRecipe` is the immutable return type of
`Position.fields(...)`, `Rotation.fields(...)`, and `Pose.fields(...)`. A
recipe stores explicit field selectors only. Each `build(...)` call supplies
its own source, optional literal prefix, source layout, frame declarations, and
passive graph association.

Recipes do not infer layouts or component meanings, retain a source, open a
reader, or mutate a frame graph. Construct recipes through the typed `fields`
methods rather than calling the recipe class directly.

Existing CSV and ROS readers return schema-bearing AOs that can be passed
directly to `build(...)`; omit `source_layout` for those results. Reading and
field construction remain separate operations: recipes never accept paths,
reopen a reader, or widen the fields selected during ingestion.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/field_recipes
   :nosignatures:

   tal.spatial.SpatialFieldRecipe
   tal.spatial.SpatialFieldRecipe.build
```

## See Also

- User guide: {doc}`../../user-guide/creating_trajectory_objects`
- I/O adapters: {doc}`../io`
- {doc}`position`
- {doc}`rotation`
- {doc}`pose`
