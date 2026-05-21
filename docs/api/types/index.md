(api-types-index)=
# Typed Analysis Objects

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

Typed AOs are `AnalysisObject` subclasses that interpret core dimensions as a
specific mathematical or spatial payload. They keep the same sequence, batch,
parameter, validity, and frame semantics as base AOs while adding stricter
payload invariants.

```{contents}
:local:
:depth: 2
```

```{toctree}
:maxdepth: 1
:caption: Spatial

position
rotation
pose
velocity
acceleration
path_solve
```

```{toctree}
:maxdepth: 1
:caption: Linear Algebra

array
vector
vector3
matrix
covariance
```

Use typed wrappers when the payload meaning should affect validation or math:
vectors and matrices for linear algebra, rotations and poses for rigid-body
geometry, and velocity/acceleration families for kinematics.
