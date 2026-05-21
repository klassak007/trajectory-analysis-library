(api-internal-api)=
# Internal API Policy

> **Audience:** Internal API Policy. This page defines which implementation
> details are intentionally outside TAL's supported public documentation
> surface.

TAL documents two supported API audiences: the User API and the Extension
Author API. Everything else is internal unless it is explicitly documented on
one of those pages.

## Unsupported Implementation Details

Undocumented helpers whose names begin with `_` are unsupported implementation
details. The only documented underscored exception is the Extension Author API
rewrap boundary pair:

- `tal.core.typed_lifecycle.TypedAnalysisObject._from_validated`
- `tal.core.typed_lifecycle.TypedAnalysisObject._from_unvalidated`

Internal implementation modules are also unsupported unless a symbol from them
is explicitly documented on a User API or Extension Author API page. This
includes low-level schema validation modules, parameter-engine modules,
orchestration intent modules, topology batch plumbing, lazy-array detection
helpers, runtime-check modules, and domain-local kernel/finalize helpers.

## Spatial Metadata Plumbing

Spatial metadata helpers that are not exported from `tal.spatial` are internal
implementation details. Application code should use documented spatial types
such as `Position`, `Rotation`, `Pose`, `Velocity`, and `Acceleration`.

## Stability

Internal APIs may change without deprecation. Code outside TAL should depend on
the documented User API or Extension Author API instead.
