(api-domain-extensions)=
# Domain Extension Helper API

This page collects the TAL core APIs used by domain extension packages. These
helpers support typed `AnalysisObject` subclasses, operand coercion, runtime
context resolution, schema finalization, and schema inspection.

Use these helpers from domain packages. `tal.core` remains domain-neutral and
does not import extension packages.

```{contents}
:local:
:depth: 2
```

## Typed Lifecycle

`TypedAnalysisObject` is the subclass scaffold for domain-specific AO types.
`TypedLifecycleSpec` declares how a subtype coerces input, applies constructor
options, normalizes metadata, and enforces subtype invariants. The default hook
helpers cover no-op or identity behavior when a type does not need custom
logic.

```{eval-rst}
.. autoclass:: tal.core.typed_lifecycle.TypedAnalysisObject

.. autoclass:: tal.core.typed_lifecycle.TypedLifecycleContext

.. autoclass:: tal.core.typed_lifecycle.TypedLifecycleSpec

.. autofunction:: tal.core.typed_lifecycle.default_coerce_source

.. autofunction:: tal.core.typed_lifecycle.identity_init_options

.. autofunction:: tal.core.typed_lifecycle.identity_normalize

.. autofunction:: tal.core.typed_lifecycle.no_op_enforce
```

## Typed Rewrap Boundaries

Typed operations normally finalize through core helpers and then rewrap into the
domain subclass. These classmethods are lifecycle-aware boundaries for that
return path. They are underscored because they are not general constructors;
use them only after an operation has deliberately selected its validation
policy.

```{eval-rst}
.. automethod:: tal.core.typed_lifecycle.TypedAnalysisObject._from_validated

.. automethod:: tal.core.typed_lifecycle.TypedAnalysisObject._from_unvalidated
```

## Input Coercion

Use `coerce_operand(...)` for user-facing operation operands and
`coerce_analysis_object_input(...)` for AO-like internal boundaries that do not
need operand labels or scalar policy.

```{eval-rst}
.. autofunction:: tal.core.orchestration.inputs.coerce_operand

.. autofunction:: tal.core.orchestration.inputs.coerce_analysis_object_input
```

## Dataset Context

Dataset context helpers resolve schema roles, parameter coordinates, validity
metadata, selected variables, and selected payload arrays at an operation
boundary.

```{eval-rst}
.. autoclass:: tal.core.orchestration.context.DatasetContextOptions

.. autoclass:: tal.core.orchestration.context.DatasetContext

.. autofunction:: tal.core.orchestration.context.resolve_dataset_context

.. autofunction:: tal.core.orchestration.context.resolve_dataset_contexts
```

## Finalization

Use finalization helpers to return schema-consistent AOs after a domain kernel
has produced an xarray dataset. `finalize_like(...)` preserves and repairs
structure from a source AO. `finalize_with_schema(...)` stamps an explicit core
schema before final validation.

```{eval-rst}
.. autofunction:: tal.core.orchestration.finalize.finalize_like

.. autoclass:: tal.core.orchestration.schema_finalize.CoreSchemaFinalizeSpec

.. autofunction:: tal.core.orchestration.schema_finalize.finalize_with_schema
```

## Schema Readers

Schema readers provide lightweight inspection of TAL metadata without mutating
the source dataset. `read_roles(...)` is commonly used by subtype invariant
hooks to confirm declared sequence, batch, and core dimensions.

```{eval-rst}
.. autofunction:: tal.core.schema_read.read_roles
```

## See Also

- {doc}`analysis-object`
- {doc}`schema`
- {doc}`../developer-guide/domain_extensions`
