(api-covariance)=
# `Covariance`

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

TAL does not currently provide a dedicated covariance typed wrapper. Represent
covariance-like payloads with `tal.linalg.Array` or `tal.linalg.Matrix` using
explicit core dimensions and labels.

When covariance-specific behavior is added, it should preserve the same
principles as the rest of `tal.linalg`: explicit payload axes, label-safe
alignment, and schema-aware finalization.

## See Also

- {doc}`array`
- {doc}`matrix`
