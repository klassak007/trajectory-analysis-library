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

## Parameter Query Kernels

TAL 0.2.0 supports the following author API for domain kernels. TAL parameter
evaluation and `tal_extensions.geo` interpolation share this owner. Resolve a
`ParamRuntimeContext`, prepare a query, execute a kernel using `plan.query`,
`plan.query_dim`, and `plan.param_map`, then finalize its Dataset with the same
plan. `plan.trajectory` describes whether the output retains sequence semantics.
Topology and output-name plans remain private to TAL.

`ParamQueryOptions` combines existing `ParamEvalOptions`, `output_intent`
(`grid` or `trajectory`), and `mapped_dataset` (whole Dataset or selected payload).
Deterministic name conflicts are rejected before map construction. The existing
parameter-map coordinate boundary can evaluate parameter/query coordinates;
payload interpolation and finalization preserve Dask laziness. Neither kernels
nor callers may mutate the resolved source or arrays retained by the plan.
Preserve surviving source coordinates and indexes during kernel assembly.
Finalization enforces coordinate ownership even with `validate=False`.

Supported import paths:

- `tal.core.param_ops.query.ParamQueryOptions`
- `tal.core.param_ops.query.ParamQueryPlan`
- `tal.core.param_ops.query.prepare_param_query`
- `tal.core.param_ops.query.finalize_param_query`
- `tal.core.param_ops.ParamEvalOptions`
- `tal.core.param_ops.types.ParamRuntimeContext`
- `tal.core.param_engine.map_apply.gather_along_sequence`
- `tal.core.param_engine.map_apply.empty_mapped_value`

```{eval-rst}
.. autoclass:: tal.core.param_ops.query.ParamQueryOptions

.. py:class:: tal.core.param_ops.query.ParamQueryPlan

   Opaque request-local query state returned by ``prepare_param_query``.
   Read ``query``, ``query_dim``, ``param_map``, and ``trajectory`` to execute
   the kernel. Pass the unchanged plan to ``finalize_param_query``. Plan
   construction and private state belong to TAL.

.. autofunction:: tal.core.param_ops.query.prepare_param_query

.. autofunction:: tal.core.param_ops.query.finalize_param_query
```

## Other Owners Consumed By First-Party Extensions

Support at these existing owner paths covers only the listed symbols. This does
not expose their private helpers, broaden core responsibilities, or establish
compatibility across TAL versions. TAL Extensions 0.1.0 requires TAL 0.2.0.

| Owner | Supported symbols and purpose |
| --- | --- |
| `tal.core.dataset_ownership` | `analysis_object_dataset`: privileged, borrowed Dataset inspection; extensions must never mutate it |
| `tal.core.orchestration.inputs` | `query_coord_from_other_input`: labeled query extraction |
| `tal.core.orchestration.runtime_checks` | `require_exact_labels`, `require_explicit_unique_dim_labels`, `require_single_core_dim_with_length`, `require_var_contains_dims`, `select_single_numeric_var`: typed representation checks |
| `tal.core.orchestration.resolve` | `effective_batch_dims`, `effective_sequence_dim`: explicit role overrides |
| `tal.core.orchestration.lazy` | `is_chunked_dataarray`: reject intentionally unsupported eager-backend input |
| `tal.core.orchestration.finalize` | `transfer_dataset_attrs`: detached source metadata transfer |
| `tal.core.combine_ops.align` | `align_contexts`: resolved binary alignment |
| `tal.core.combine_ops.types` | `AlignOptions`, `CombineContext`: immutable alignment declarations |
| `tal.core.metadata_optional` | `shared_optional_name`: reconcile optional coordinate declarations |
| `tal.core.schema` | `merge_schema`: detached schema updates at extension metadata owners |
| `tal.core.param_ops.guards` | `validate_query_dim_name`, `assert_query_dim_safe`, `assert_reserved_metadata_safe`: query name validation |
| `tal.spatial.conversion.finalize` | `allocate_free_dim_name`, `dataset_dim_names`: spatial conversion dimensions |
| `tal.spatial.metadata` | `set_position_rep`, `set_expressed_in`: spatial metadata owner |
| `tal.utils.frame_schema` | `get_frames`, `set_frames`: frame metadata owner |

Spatial conversion helpers remain in spatial; extension-only math remains in
the extension domain. The separately installed geo and astro implementations
live under `extensions/src/tal_extensions`, and their schema namespaces remain
`tal.ext.geo` and `tal.ext.astro`. Installing or importing either distribution
does not register `Position.geo`; applications explicitly call
`tal_extensions.geo.register_position_accessor()` when they use that accessor.

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
