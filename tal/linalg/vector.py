from __future__ import annotations

import xarray as xr

from ..core.analysis_object import AnalysisObject
from .array import Array
from .lifecycle import VECTOR_LIFECYCLE


class Vector(Array):
    """Typed linalg vector subtype over ``Array``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    LIFECYCLE = VECTOR_LIFECYCLE

    def set_core_dims(self, *dims: str) -> "Vector":
        """Set vector core dimension (exactly one required).

        Parameters
        ----------
        *dims : str
            Core-dimension labels/shape metadata used for structural operations.

        Returns
        -------
        Vector
            Result of applying this operation with TAL semantic constraints preserved.

        Notes
        -----
        Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
        """
        if len(dims) != 1:
            raise ValueError(f"Vector.set_core_dims: Vector requires exactly one core dim; got {dims!r}.")
        out = super().set_core_dims(*dims)
        return out

    def dot(
        self,
        other: "Array | AnalysisObject | xr.Dataset | xr.DataArray",
    ) -> "Array":
        """Compute vector dot product.

        Parameters
        ----------
        other
            Right operand with matching vector core dim.

        Returns
        -------
        Array
            Scalar-core array result.

        Notes
        -----
        Dot is strict on matching core-dimension names and uses label-aware
        alignment.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.linalg import Vector
        >>> left = Vector(AnalysisObject.from_data(
        ...     xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... ))
        >>> right = Vector(AnalysisObject.from_data(
        ...     xr.Dataset({"v": (("sample", "axis"), [[4.0, 5.0, 6.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... ))
        >>> left.dot(right).unsafe_data["datavar"].to_numpy().tolist()
        [32.0]
        """
        from .ops import dot

        return dot(self, other)

    def norm(
        self,
        *,
        ord: int | float | None = 2,
    ) -> "Array":
        """Compute vector norm.

        Parameters
        ----------
        ord
            Norm order passed through linalg norm owner.

        Returns
        -------
        Array
            Scalar-core array result.

        Notes
        -----
        Norm is strict on one declared vector core dimension.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.linalg import Vector
        >>> vec = Vector(AnalysisObject.from_data(
        ...     xr.Dataset({"v": (("sample", "axis"), [[3.0, 4.0, 0.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... ))
        >>> vec.norm().unsafe_data["datavar"].to_numpy().tolist()
        [5.0]
        """
        from .ops import norm

        return norm(self, ord=ord)


__all__ = [
    "Vector",
]
