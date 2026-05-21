(api-domain-extensions)=
# Extension Author API

```{important}
For package authors building TAL domain extensions; application code should
normally use the User API.
```

This page collects the TAL core APIs used by domain extension packages. These
helpers support typed `AnalysisObject` subclasses, operand coercion, runtime
context resolution, topology planning, schema finalization, and schema
inspection.

Use these helpers from domain packages by full module path. `tal.core` remains
domain-neutral and does not import extension packages.

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

Supported import paths:

- `tal.core.typed_lifecycle.TypedAnalysisObject`
- `tal.core.typed_lifecycle.TypedLifecycleSpec`
- `tal.core.typed_lifecycle.TypedLifecycleContext`
- `tal.core.typed_lifecycle.SourceCoercer`
- `tal.core.typed_lifecycle.DatasetHook`
- `tal.core.typed_lifecycle.EnforceHook`
- `tal.core.typed_lifecycle.default_coerce_source`
- `tal.core.typed_lifecycle.identity_init_options`
- `tal.core.typed_lifecycle.identity_normalize`
- `tal.core.typed_lifecycle.no_op_enforce`

```{eval-rst}
.. autoclass:: tal.core.typed_lifecycle.TypedAnalysisObject

.. autoclass:: tal.core.typed_lifecycle.TypedLifecycleContext

.. autoclass:: tal.core.typed_lifecycle.TypedLifecycleSpec

.. autodata:: tal.core.typed_lifecycle.SourceCoercer
   :no-value:

.. autodata:: tal.core.typed_lifecycle.DatasetHook
   :no-value:

.. autodata:: tal.core.typed_lifecycle.EnforceHook
   :no-value:

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

Supported import paths:

- `tal.core.typed_lifecycle.TypedAnalysisObject._from_validated`
- `tal.core.typed_lifecycle.TypedAnalysisObject._from_unvalidated`

```{eval-rst}
.. automethod:: tal.core.typed_lifecycle.TypedAnalysisObject._from_validated

.. automethod:: tal.core.typed_lifecycle.TypedAnalysisObject._from_unvalidated
```

## Input Coercion

Use `coerce_operand(...)` for user-facing operation operands and
`coerce_analysis_object_input(...)` for AO-like internal boundaries that do not
need operand labels or scalar policy. Use
`normalize_analysis_object_inputs(...)` for variadic AO-like boundaries that
need deterministic index-aware diagnostics.

Supported import paths:

- `tal.core.orchestration.inputs.coerce_operand`
- `tal.core.orchestration.inputs.coerce_analysis_object_input`
- `tal.core.orchestration.inputs.normalize_analysis_object_inputs`

```{eval-rst}
.. autofunction:: tal.core.orchestration.inputs.coerce_operand

.. autofunction:: tal.core.orchestration.inputs.coerce_analysis_object_input

.. autofunction:: tal.core.orchestration.inputs.normalize_analysis_object_inputs
```

## Dataset Context

Dataset context helpers resolve schema roles, parameter coordinates, validity
metadata, selected variables, and selected payload arrays at an operation
boundary.

Supported import paths:

- `tal.core.orchestration.context.DatasetContextOptions`
- `tal.core.orchestration.context.DatasetContext`
- `tal.core.orchestration.context.resolve_dataset_context`
- `tal.core.orchestration.context.resolve_dataset_contexts`

```{eval-rst}
.. autoclass:: tal.core.orchestration.context.DatasetContextOptions

.. autoclass:: tal.core.orchestration.context.DatasetContext

.. autofunction:: tal.core.orchestration.context.resolve_dataset_context

.. autofunction:: tal.core.orchestration.context.resolve_dataset_contexts
```

## Operation Context

Operation context helpers adapt normalized AO inputs into combine-style or
parameter-runtime context objects. Use these when a domain operation needs the
same role, parameter-coordinate, or validity semantics as TAL's built-in
combine and parameter operations.

Supported import paths:

- `tal.core.orchestration.resolve.resolve_combine_contexts`
- `tal.core.orchestration.resolve.resolve_param_runtime_context`

```{eval-rst}
.. autofunction:: tal.core.orchestration.resolve.resolve_combine_contexts

.. autofunction:: tal.core.orchestration.resolve.resolve_param_runtime_context
```

## Topology Planning

Topology helpers support advanced domain operations that need TAL's semantic
alignment, broadcast, flattening, and topology restoration behavior.

Supported import paths:

- `tal.core.orchestration.topology.SemanticTopology`
- `tal.core.orchestration.topology.TopologyPolicy`
- `tal.core.orchestration.topology.TopologyOperand`
- `tal.core.orchestration.topology.ResolvedTopologyPlan`
- `tal.core.orchestration.topology.resolve_unary_topology`
- `tal.core.orchestration.topology.resolve_binary_topology`
- `tal.core.orchestration.topology.resolve_nary_topology`
- `tal.core.orchestration.topology.realize_operands_for_plan`
- `tal.core.orchestration.topology.flatten_param_contexts`
- `tal.core.orchestration.topology.flatten_query_for_batch_plan`
- `tal.core.orchestration.topology.restore_dataset_batch_topology`
- `tal.core.orchestration.topology.restore_dataset_multi_batch`

```{eval-rst}
.. autoclass:: tal.core.orchestration.topology.SemanticTopology

.. autoclass:: tal.core.orchestration.topology.TopologyPolicy

.. autoclass:: tal.core.orchestration.topology.TopologyOperand

.. autoclass:: tal.core.orchestration.topology.ResolvedTopologyPlan

.. autofunction:: tal.core.orchestration.topology.resolve_unary_topology

.. autofunction:: tal.core.orchestration.topology.resolve_binary_topology

.. autofunction:: tal.core.orchestration.topology.resolve_nary_topology

.. autofunction:: tal.core.orchestration.topology.realize_operands_for_plan
```

## Finalization

Use finalization helpers to return schema-consistent AOs after a domain kernel
has produced an xarray dataset. `finalize_like(...)` preserves and repairs
structure from a source AO. `finalize_with_schema(...)` stamps an explicit core
schema before final validation.

Supported import paths:

- `tal.core.orchestration.finalize.finalize_like`
- `tal.core.orchestration.finalize.finalize_many_like`
- `tal.core.orchestration.finalize.restore_and_finalize`
- `tal.core.orchestration.finalize.rewrap_unvalidated_like`
- `tal.core.orchestration.schema_finalize.CoreSchemaFinalizeSpec`
- `tal.core.orchestration.schema_finalize.finalize_with_schema`
- `tal.core.orchestration.schema_finalize.clear_core_schema_blocks`

```{eval-rst}
.. autofunction:: tal.core.orchestration.finalize.finalize_like

.. autofunction:: tal.core.orchestration.finalize.finalize_many_like

.. autofunction:: tal.core.orchestration.finalize.rewrap_unvalidated_like

.. autoclass:: tal.core.orchestration.schema_finalize.CoreSchemaFinalizeSpec

.. autofunction:: tal.core.orchestration.schema_finalize.finalize_with_schema

.. autofunction:: tal.core.orchestration.schema_finalize.clear_core_schema_blocks
```

## Schema And Validity Inspection

Schema and validity helpers provide lightweight inspection and repair of TAL
metadata without interpreting domain-specific extension namespaces.
`read_roles(...)` is commonly used by subtype invariant hooks to confirm
declared sequence, batch, and core dimensions.

Supported import paths:

- `tal.core.schema_read.read_roles`
- `tal.core.schema_read.read_param_coord_name`
- `tal.core.schema_read.read_sequence_size_coord_name`
- `tal.core.schema_read.validate_schema_if_needed`
- `tal.core.validity_finalize.assign_sequence_size_from_valid_mask`
- `tal.core.validity_finalize.set_left_packed_validity_or_prune_from_size_coord`
- `tal.core.validity_finalize.reconcile_sequence_validity_after_structure`

```{eval-rst}
.. autofunction:: tal.core.schema_read.read_roles

.. autofunction:: tal.core.schema_read.read_param_coord_name

.. autofunction:: tal.core.schema_read.read_sequence_size_coord_name

.. autofunction:: tal.core.schema_read.validate_schema_if_needed

.. autofunction:: tal.core.validity_finalize.assign_sequence_size_from_valid_mask

.. autofunction:: tal.core.validity_finalize.set_left_packed_validity_or_prune_from_size_coord

.. autofunction:: tal.core.validity_finalize.reconcile_sequence_validity_after_structure
```

## See Also

- {doc}`analysis-object`
- {doc}`schema`
- {doc}`../developer-guide/domain_extensions`
