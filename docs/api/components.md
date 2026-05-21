(api-components)=
# Components

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

Components let an AO name meaningful pieces inside a core dimension. Use them
when a payload has stable substructure, such as linear and angular parts of a
six-vector, named joints, marker groups, or pose components.

Component metadata is stored in the TAL schema and is repaired or pruned by
structural AO operations.

## Functional API

```python
from tal.core import (
    ComponentComposeOptions,
    ComponentExtractOptions,
    ComponentPatchOptions,
    ComponentRegistryOptions,
    ComponentSpec,
    compose_components,
    define_components,
    extract_components,
    patch_components,
    read_components,
)
```

```python
define_components(ao, *, opts=ComponentRegistryOptions(...), validate=True)
read_components(ao)
extract_components(ao, *, opts=ComponentExtractOptions(...), validate=True)
patch_components(base, components, *, opts=ComponentPatchOptions(...), validate=True)
compose_components(components, *, opts=ComponentComposeOptions(...), validate=True)
```

## Accessor API

The same operations are available from `ao.components`:

```python
ao.components.define(opts=..., validate=True)
ao.components.registry()
ao.components.extract(opts=..., validate=True)
ao.components.patch(components, opts=..., validate=True)
ao.components.compose(components, opts=..., validate=True)
```

## Registry Contract

- Registry entries are keyed by component name.
- Each entry declares `core_dim`, `labels`, and optional `var`.
- Labels are strict JSON-scalar values and must be unique.
- Component label sets must be disjoint within each `core_dim`.
- Extraction is label-driven; there is no positional fallback.
- Patching requires exact component label-set matching and an explicit overlap
  policy.
- Composition order follows registry insertion order.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/components
   :nosignatures:

   tal.core.ComponentSpec
   tal.core.ComponentRegistryOptions
   tal.core.ComponentComposeOptions
   tal.core.ComponentExtractOptions
   tal.core.ComponentPatchOptions
   tal.core.compose_components
   tal.core.define_components
   tal.core.extract_components
   tal.core.patch_components
   tal.core.read_components
   tal.core.component_ops.accessor.ComponentsAccessor.define
   tal.core.component_ops.accessor.ComponentsAccessor.registry
   tal.core.component_ops.accessor.ComponentsAccessor.extract
   tal.core.component_ops.accessor.ComponentsAccessor.patch
   tal.core.component_ops.accessor.ComponentsAccessor.compose
```

## See Also

- {doc}`analysis-object`
- User guide: {doc}`../user-guide/core_concepts`
