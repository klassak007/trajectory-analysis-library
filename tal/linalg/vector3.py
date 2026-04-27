from __future__ import annotations

import xarray as xr

from ..core.analysis_object import AnalysisObject
from ..core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
    validate_schema_if_needed,
)
from .array import Array
from .component_context import (
    build_reference_component_context,
    coerce_component_or_scalar,
    promote_scalar_component,
)
from .vector import Vector

_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")


def _normalize_xyz_components(
    x: object,
    y: object,
    z: object,
    *,
    owner: str,
) -> tuple[AnalysisObject, AnalysisObject, AnalysisObject]:
    raw = (("x", x), ("y", y), ("z", z))
    coerced = [coerce_component_or_scalar(value, owner=owner, label=label) for label, value in raw]
    reference = next((item for item in coerced if item is not None), None)
    if reference is None:
        raise ValueError(f"{owner}: at least one AO-like component is required when using scalar components.")
    spec = build_reference_component_context(reference, owner=owner)
    out: list[AnalysisObject] = []
    for (label, value), item in zip(raw, coerced):
        if item is not None:
            out.append(item)
            continue
        out.append(promote_scalar_component(value, reference=spec, owner=f"{owner}: {label}"))
    return out[0], out[1], out[2]


def _semantic_dim_order(ds: xr.Dataset, *, owner: str) -> tuple[str, ...]:
    declared, sequence_dim, batch_dims, core_dims = read_roles(ds)
    if not declared:
        raise ValueError(f"{owner}: from_xyz requires declared roles.")
    seen: set[str] = set()
    ordered: list[str] = []
    role_dims: tuple[str, ...]
    if sequence_dim is None:
        role_dims = tuple(batch_dims) + tuple(core_dims)
    else:
        role_dims = (sequence_dim, *batch_dims, *core_dims)
    for dim in role_dims:
        if dim in seen:
            continue
        seen.add(dim)
        ordered.append(dim)
    ordered.extend(dim for dim in ds.dims if dim not in seen)
    return tuple(ordered)


class Vector3(Vector):
    """Fixed-axis 3D vector subtype over ``Vector``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    XYZ_LABELS: tuple[str, str, str] = _XYZ_LABELS

    def __init__(
        self,
        data: "Array | AnalysisObject | xr.Dataset | xr.DataArray",
        *,
        core_dims: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(data, core_dims=core_dims)
        self._enforce_array_invariants(owner="Vector3.__init__")

    def _enforce_array_invariants(self, *, owner: str) -> None:
        _, _, core_dims = self._declared_roles(owner=owner)
        if len(core_dims) != 1:
            raise ValueError(f"{owner}: Vector3 requires exactly one core dim; got {core_dims!r}.")
        axis = core_dims[0]
        if int(self.unsafe_data.sizes.get(axis, -1)) != 3:
            raise ValueError(f"{owner}: Vector3 core axis {axis!r} must have length 3.")
        if axis not in self.unsafe_data.coords:
            raise ValueError(f"{owner}: Vector3 core axis {axis!r} must have labels ('x', 'y', 'z').")
        labels = tuple(self.unsafe_data.coords[axis].to_index().tolist())
        if labels != self.XYZ_LABELS:
            raise ValueError(f"{owner}: Vector3 core axis labels must equal {self.XYZ_LABELS!r}; got {labels!r}.")
        return None

    @classmethod
    def from_xyz(
        cls,
        x: object,
        y: object,
        z: object,
        *,
        axis: str = "axis",
        output_var: str | None = None,
        validate: bool = True,
    ) -> "Vector3":
        """Build a ``Vector3`` from x/y/z components.

        Parameters
        ----------
        x, y, z
            AO-like components or scalars (at least one AO-like required).
        axis
            Output core-axis name for xyz labels.
        output_var
            Optional output variable name.
        validate
            Whether to validate schema on output.

        Returns
        -------
        Vector3
            Vector3 payload with labels ``('x', 'y', 'z')``.

        Notes
        -----
        Scalar components are promoted using reference topology from AO-like
        components. Role/topology conflicts fail closed.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.linalg import Vector3
        >>> x = AnalysisObject.from_data(
        ...     xr.Dataset({"x": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> vec = Vector3.from_xyz(x, 0.0, 1.0, axis="axis", output_var="vec3")
        >>> tuple(vec.unsafe_data.coords["axis"].to_numpy().tolist())
        ('x', 'y', 'z')
        """
        owner = "linalg.vector3.from_xyz"
        if not isinstance(axis, str) or not axis:
            raise ValueError(f"{owner}: axis must be a non-empty string.")
        x_ao, y_ao, z_ao = _normalize_xyz_components(x, y, z, owner=owner)
        from .ops import stack_core

        assembled = stack_core(
            [x_ao, y_ao, z_ao],
            core_dim=axis,
            core_labels=cls.XYZ_LABELS,
            output_var=output_var,
            validate=validate,
        )
        semantic_order = _semantic_dim_order(assembled.unsafe_data, owner=owner)
        assembled = assembled.transpose(*semantic_order, validate=validate)
        if validate:
            return cls._from_validated(assembled.unsafe_data)
        return cls._from_unvalidated(assembled.unsafe_data)

    def _component_scalar_array(self, label: str, *, owner: str) -> Array:
        ds = validate_schema_if_needed(self.unsafe_data)
        declared, sequence_dim, batch_dims, core_dims = read_roles(ds)
        if not declared:
            raise ValueError(f"{owner}: Vector3 requires declared roles.")
        axis = core_dims[0]
        selected = ds.sel({axis: label}, drop=True)
        component = AnalysisObject.from_data(
            selected,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            core_dims=(),
            param_coord=read_param_coord_name(ds),
            sequence_size_coord=read_sequence_size_coord_name(ds),
            validate=True,
        )
        return Array._from_validated(component.unsafe_data)

    @property
    def x(self) -> Array:
        """Return ``x`` component as scalar-core ``Array``.

        Returns
        -------
        Array
            Resolved property value.
        """
        return self._component_scalar_array("x", owner="Vector3.x")

    @property
    def y(self) -> Array:
        """Return ``y`` component as scalar-core ``Array``.

        Returns
        -------
        Array
            Resolved property value.
        """
        return self._component_scalar_array("y", owner="Vector3.y")

    @property
    def z(self) -> Array:
        """Return ``z`` component as scalar-core ``Array``.

        Returns
        -------
        Array
            Resolved property value.
        """
        return self._component_scalar_array("z", owner="Vector3.z")


__all__ = [
    "Vector3",
]
