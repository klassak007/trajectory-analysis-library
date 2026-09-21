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
- {doc}`position`
- {doc}`rotation`
- {doc}`pose`
