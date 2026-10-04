from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from ..core.analysis_object import AnalysisObject
from ..core.dataset_ownership import analysis_object_dataset
from .array import Array
from .lifecycle import MATRIX_LIFECYCLE, matrix_core_dims

if TYPE_CHECKING:
    from .ops.pinv import PInvOptions
    from .ops.solve import SolveOptions
    from .vector import Vector


class Matrix(Array):
    """Typed linalg matrix subtype over ``Array``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    LIFECYCLE = MATRIX_LIFECYCLE

    def _matrix_core_dims(self, *, owner: str) -> tuple[str, str]:
        return matrix_core_dims(analysis_object_dataset(self), owner=owner)

    def set_core_dims(self, *dims: str) -> "Matrix":
        """Set matrix core dimensions (exactly two distinct names).

        Parameters
        ----------
        *dims : str
            Core-dimension labels/shape metadata used for structural operations.

        Returns
        -------
        Matrix
            Result of applying this operation with TAL semantic constraints preserved.

        Notes
        -----
        Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
        """
        if len(dims) != 2:
            raise ValueError(f"Matrix.set_core_dims: Matrix requires exactly two core dims; got {dims!r}.")
        out = super().set_core_dims(*dims)
        return out

    @property
    def T(self) -> "Matrix":
        """Return matrix transpose with swapped declared core dimensions.

        Returns
        -------
        Matrix
            Resolved property value.
        """
        row_dim, col_dim = self._matrix_core_dims(owner="Matrix.T")
        dims = list(analysis_object_dataset(self).dims)
        row_idx = dims.index(row_dim)
        col_idx = dims.index(col_dim)
        dims[row_idx], dims[col_idx] = dims[col_idx], dims[row_idx]
        transposed = self.transpose(*dims, validate=True)
        return transposed.set_matrix_axes(col_dim, row_dim)

    def solve(
        self,
        rhs: "Array | AnalysisObject | xr.Dataset | xr.DataArray",
        *,
        opts: "SolveOptions | None" = None,
    ) -> "Vector | Matrix":
        """Solve ``self @ x = rhs`` with strict role-driven semantics.

        Parameters
        ----------
        rhs : Array | AnalysisObject | xr.Dataset | xr.DataArray
            Right-hand-side vector or matrix in ``A @ x = rhs``.
        opts : SolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``method`` (``'auto'``, ``'solve'``, or ``'lstsq'``) and ``rcond`` for least-squares truncation.

        Returns
        -------
        Vector | Matrix
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.linalg import Matrix, Vector
        >>> from tal.linalg.ops.solve import SolveOptions
        >>> A = Matrix(AnalysisObject.from_data(
        ...     xr.Dataset({"A": (("sample", "row", "col"), [[[2.0, 0.0], [0.0, 3.0]]])}, coords={"sample": [0], "row": ["r0", "r1"], "col": ["c0", "c1"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("row", "col"),
        ...     validate=True,
        ... ))
        >>> b = Vector(AnalysisObject.from_data(
        ...     xr.Dataset({"b": (("sample", "row"), [[8.0, 15.0]])}, coords={"sample": [0], "row": ["r0", "r1"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("row",),
        ...     validate=True,
        ... ))
        >>> A.solve(b, opts=SolveOptions(method="solve")).as_dataset()["datavar"].values.tolist()
        [[4.0, 5.0]]
        """
        from .ops import solve

        return solve(self, rhs, opts=opts)

    def inv(self) -> "Matrix":
        """Return strict square-matrix inverse.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.linalg import Matrix
        >>> A = Matrix(AnalysisObject.from_data(
        ...     xr.Dataset(
        ...         {"A": (("sample", "row", "col"), [[[2.0, 0.0], [0.0, 4.0]]])},
        ...         coords={"sample": [0], "row": ["r0", "r1"], "col": ["c0", "c1"]},
        ...     ),
        ...     sequence_dim="sample",
        ...     core_dims=("row", "col"),
        ...     validate=True,
        ... ))
        >>> A.inv().as_dataset()["datavar"].to_numpy().tolist()
        [[[0.5, 0.0], [0.0, 0.25]]]
        """
        from .ops import inv

        return inv(self)

    def pinv(
        self,
        *,
        opts: "PInvOptions | None" = None,
    ) -> "Matrix":
        """Return role-driven Moore-Penrose pseudoinverse.

        Parameters
        ----------
        opts : PInvOptions | None, optional
            When ``None``, defaults are used. Key fields are ``rcond`` and ``hermitian``.

        Returns
        -------
        Matrix
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.linalg import Matrix
        >>> from tal.linalg.ops.pinv import PInvOptions
        >>> A = Matrix(AnalysisObject.from_data(
        ...     xr.Dataset({"A": (("sample", "row", "col"), [[[2.0, 0.0], [0.0, 4.0]]])}, coords={"sample": [0], "row": ["r0", "r1"], "col": ["c0", "c1"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("row", "col"),
        ...     validate=True,
        ... ))
        >>> A.pinv(opts=PInvOptions(rcond=1e-8)).as_dataset()["datavar"].values.tolist()
        [[[0.5, 0.0], [0.0, 0.25]]]
        """
        from .ops import pinv

        return pinv(self, opts=opts)


__all__ = [
    "Matrix",
]
