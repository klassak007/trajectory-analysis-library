from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from ..core.analysis_object import AnalysisObject
from ..core.typed_lifecycle import TypedAnalysisObject
from .lifecycle import ARRAY_LIFECYCLE, ArrayInitOptions, declared_roles, normalize_core_dims


class Array(TypedAnalysisObject):
    """Typed linalg array subtype over ``AnalysisObject``.

    Notes
    -----
    Array operations are label-aware and preserve xarray semantics: alignment is
    by dimension names and coordinate labels.

    Operator Families
    -----------------
    ``+``, ``-``, and ``@`` delegate to linalg owner functions
    (``add``, ``sub``, ``matmul``) after strict role/topology planning.
    """

    def __init__(
        self,
        data: "Array | AnalysisObject | xr.Dataset | xr.DataArray",
        *,
        core_dims: tuple[str, ...] | None = None,
    ) -> None:
        self._init_typed(data, options=ArrayInitOptions(core_dims=core_dims))

    LIFECYCLE = ARRAY_LIFECYCLE

    def _declared_roles(self, *, owner: str) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
        return declared_roles(self.unsafe_data, owner=owner)

    def _declared_role_context(self, *, owner: str) -> tuple[str | None, tuple[str, ...]]:
        sequence_dim, batch_dims, _ = self._declared_roles(owner=owner)
        return sequence_dim, batch_dims

    def set_core_dims(self, *dims: str) -> "Array":
        """Set declared core dimensions for the array.

        Parameters
        ----------
        *dims : str
            Existing dataset dimensions to treat as mathematical payload axes.

        Returns
        -------
        Array
            Array with the same data and updated core-dimension role metadata.

        Notes
        -----
        The dimensions must already exist and must be unique. Sequence and
        batch roles are preserved.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.linalg import Array
        >>> arr = Array(AnalysisObject.from_data(
        ...     xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... ))
        >>> arr.set_core_dims("axis").unsafe_data["v"].dims
        ('sample', 'axis')
        """
        owner = "Array.set_core_dims"
        sequence_dim, batch_dims = self._declared_role_context(owner=owner)
        normalized = normalize_core_dims(tuple(dims), owner=owner)
        out = self.set_roles(
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            core_dims=normalized,
            validate=True,
        )
        return out

    def set_vector_axis(self, dim: str) -> "Array":
        """Convenience alias for setting one core dimension.

        Parameters
        ----------
        dim : str
            Existing dimension to use as the vector axis.

        Returns
        -------
        Array
            Array with ``dim`` as its only core dimension.

        Notes
        -----
        This method is equivalent to ``set_core_dims(dim)``.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.core.schema_read import read_roles
        >>> from tal.linalg import Array
        >>> arr = Array(AnalysisObject.from_data(
        ...     xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... ))
        >>> read_roles(arr.set_vector_axis("axis").unsafe_data)[3]
        ('axis',)
        """
        return self.set_core_dims(dim)

    def set_matrix_axes(self, row_dim: str, col_dim: str) -> "Array":
        """Convenience alias for setting two distinct core dimensions.

        Parameters
        ----------
        row_dim : str
            Existing dimension to use as the matrix row axis.
        col_dim : str
            Existing dimension to use as the matrix column axis.

        Returns
        -------
        Array
            Array with ``(row_dim, col_dim)`` as core dimensions.

        Notes
        -----
        ``row_dim`` and ``col_dim`` must be different names because xarray
        dimensions are name-addressed.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.core.schema_read import read_roles
        >>> from tal.linalg import Array
        >>> arr = Array(AnalysisObject.from_data(
        ...     xr.Dataset({"A": (("sample", "row", "col"), [[[1.0, 0.0], [0.0, 1.0]]])}, coords={"sample": [0], "row": ["r0", "r1"], "col": ["c0", "c1"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("row", "col"),
        ...     validate=True,
        ... ))
        >>> read_roles(arr.set_matrix_axes("row", "col").unsafe_data)[3]
        ('row', 'col')
        """
        if not isinstance(row_dim, str):
            raise TypeError("Array.set_matrix_axes: row_dim must be str.")
        if not isinstance(col_dim, str):
            raise TypeError("Array.set_matrix_axes: col_dim must be str.")
        if row_dim == col_dim:
            raise ValueError("Array.set_matrix_axes: row_dim and col_dim must be distinct.")
        return self.set_core_dims(row_dim, col_dim)

    def as_core(self, *dims: str) -> "Array":
        """Alias for :meth:`Array.set_core_dims`.

        Parameters
        ----------
        *dims : str
            Existing dimensions to treat as core dimensions.

        Returns
        -------
        Array
            Array with updated core-dimension role metadata.

        Notes
        -----
        This is a readability alias for fluent code.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.core.schema_read import read_roles
        >>> from tal.linalg import Array
        >>> arr = Array(AnalysisObject.from_data(
        ...     xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... ))
        >>> read_roles(arr.as_core("axis").unsafe_data)[3]
        ('axis',)
        """
        return self.set_core_dims(*dims)

    def axis(self, dim: str) -> "Array":
        """Alias for :meth:`Array.set_vector_axis`.

        Parameters
        ----------
        dim : str
            Existing dimension to use as the vector axis.

        Returns
        -------
        Array
            Array with ``dim`` as its only core dimension.

        Notes
        -----
        This alias is useful when declaring vector semantics inline.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.core.schema_read import read_roles
        >>> from tal.linalg import Array
        >>> arr = Array(AnalysisObject.from_data(
        ...     xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... ))
        >>> read_roles(arr.axis("axis").unsafe_data)[3]
        ('axis',)
        """
        return self.set_vector_axis(dim)

    def rc(self, row_dim: str, col_dim: str) -> "Array":
        """Alias for :meth:`Array.set_matrix_axes`.

        Parameters
        ----------
        row_dim : str
            Existing dimension to use as the matrix row axis.
        col_dim : str
            Existing dimension to use as the matrix column axis.

        Returns
        -------
        Array
            Array with ``(row_dim, col_dim)`` as core dimensions.

        Notes
        -----
        This alias is useful when declaring matrix semantics inline.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.core.schema_read import read_roles
        >>> from tal.linalg import Array
        >>> arr = Array(AnalysisObject.from_data(
        ...     xr.Dataset({"A": (("sample", "row", "col"), [[[1.0, 0.0], [0.0, 1.0]]])}, coords={"sample": [0], "row": ["r0", "r1"], "col": ["c0", "c1"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("row", "col"),
        ...     validate=True,
        ... ))
        >>> read_roles(arr.rc("row", "col").unsafe_data)[3]
        ('row', 'col')
        """
        return self.set_matrix_axes(row_dim, col_dim)

    def __matmul__(self, other: "Array | AnalysisObject | xr.Dataset | xr.DataArray") -> "Array":
        """Matrix multiply via :func:`tal.linalg.matmul`."""
        from .ops import matmul

        return matmul(self, other)

    def __add__(self, other: "Array | AnalysisObject | xr.Dataset | xr.DataArray") -> "Array":
        """Elementwise add via :func:`tal.linalg.add`."""
        from .ops import add

        return add(self, other)

    def __radd__(self, other: "Array | AnalysisObject | xr.Dataset | xr.DataArray") -> "Array":
        """Right-hand elementwise add via :func:`tal.linalg.add`."""
        from .ops import add

        return add(other, self)

    def __sub__(self, other: "Array | AnalysisObject | xr.Dataset | xr.DataArray") -> "Array":
        """Elementwise subtract via :func:`tal.linalg.sub`."""
        from .ops import sub

        return sub(self, other)

    def __rsub__(self, other: "Array | AnalysisObject | xr.Dataset | xr.DataArray") -> "Array":
        """Right-hand elementwise subtract via :func:`tal.linalg.sub`."""
        from .ops import sub

        return sub(other, self)

    @classmethod
    def assemble_core(
        cls,
        values: object,
        *,
        core_dims: Sequence[str],
        core_labels: Sequence[Sequence[object]] | None = None,
        output_var: str | None = None,
        validate: bool = True,
    ) -> "Array":
        """Assemble nested layout into new core dimensions.

        Parameters
        ----------
        values : object
            Nested AO-like layout to assemble.
        core_dims : Sequence[str]
            New core dimensions represented by the nesting levels.
        core_labels : Sequence[Sequence[object]] | None, optional
            Optional coordinate labels for each new core dimension.
        output_var : str | None, optional
            Output variable name.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Array
            Array with assembled core dimensions.

        Notes
        -----
        This classmethod delegates to :func:`tal.linalg.assemble_core` and wraps
        the result in ``cls``.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.linalg import Array
        >>> x = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> y = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> Array.assemble_core([x, y], core_dims=("axis",), core_labels=(("x", "y"),)).unsafe_data.sizes["axis"]
        2
        """
        from .ops import assemble_core

        out = assemble_core(
            values,
            core_dims=core_dims,
            core_labels=core_labels,
            output_var=output_var,
            validate=validate,
        )
        if cls is Array:
            return out
        return cls._from_validated(out.unsafe_data)

    @classmethod
    def stack_core(
        cls,
        values: object,
        *,
        core_dim: str,
        core_labels: Sequence[object] | None = None,
        output_var: str | None = None,
        validate: bool = True,
    ) -> "Array":
        """Stack AO-like values along one core dimension.

        Parameters
        ----------
        values : object
            AO-like values to stack.
        core_dim : str
            New core dimension.
        core_labels : Sequence[object] | None, optional
            Optional coordinate labels for ``core_dim``.
        output_var : str | None, optional
            Output variable name.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Array
            Array with a new one-dimensional core axis.

        Notes
        -----
        This is a one-dimensional convenience wrapper around
        :meth:`assemble_core`.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.linalg import Array
        >>> x = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> y = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> Array.stack_core([x, y], core_dim="axis", core_labels=("x", "y")).unsafe_data.sizes["axis"]
        2
        """
        from .ops import stack_core

        out = stack_core(
            values,
            core_dim=core_dim,
            core_labels=core_labels,
            output_var=output_var,
            validate=validate,
        )
        if cls is Array:
            return out
        return cls._from_validated(out.unsafe_data)

    @classmethod
    def block_core(
        cls,
        values: object,
        *,
        row_dim: str,
        col_dim: str,
        row_labels: Sequence[object] | None = None,
        col_labels: Sequence[object] | None = None,
        output_var: str | None = None,
        validate: bool = True,
    ) -> "Array":
        """Assemble nested 2D layout into row/col core dimensions.

        Parameters
        ----------
        values : object
            Two-dimensional nested AO-like layout.
        row_dim : str
            New row core dimension.
        col_dim : str
            New column core dimension.
        row_labels : Sequence[object] | None, optional
            Optional coordinate labels for ``row_dim``.
        col_labels : Sequence[object] | None, optional
            Optional coordinate labels for ``col_dim``.
        output_var : str | None, optional
            Output variable name.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Array
            Array with row and column core dimensions.

        Notes
        -----
        This is a two-dimensional convenience wrapper around
        :meth:`assemble_core`.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.linalg import Array
        >>> x = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> y = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> out = Array.block_core([[x], [y]], row_dim="row", col_dim="col")
        >>> (out.unsafe_data.sizes["row"], out.unsafe_data.sizes["col"])
        (2, 1)
        """
        from .ops import block_core

        out = block_core(
            values,
            row_dim=row_dim,
            col_dim=col_dim,
            row_labels=row_labels,
            col_labels=col_labels,
            output_var=output_var,
            validate=validate,
        )
        if cls is Array:
            return out
        return cls._from_validated(out.unsafe_data)

    @classmethod
    def concat_core(
        cls,
        aos: Sequence[object],
        *,
        opts: "CoreConcatOptions | None" = None,
        validate: bool = True,
    ) -> "Array":
        """Concatenate AO-like values along existing core dimension.

        Parameters
        ----------
        aos : Sequence[object]
            AO-like inputs consumed by this orchestration boundary.
        opts : CoreConcatOptions | None, optional
            When ``None``, defaults are used. Key fields are ``core_dim``
            (concatenation axis), ``core_labels`` (labels to assign along
            that axis), and ``output_var`` (override output variable name).
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Array
            Result of applying this operation with TAL semantic constraints preserved.

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
        >>> from tal.core import AnalysisObject, CoreConcatOptions
        >>> from tal.linalg import Array
        >>> x = Array(AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[1.0]])}, coords={"sample": [0], "axis": ["x"]}), sequence_dim="sample", core_dims=("axis",), validate=True))
        >>> y = Array(AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[2.0]])}, coords={"sample": [0], "axis": ["y"]}), sequence_dim="sample", core_dims=("axis",), validate=True))
        >>> out = Array.concat_core([x, y], opts=CoreConcatOptions(core_dim="axis"))
        >>> out.unsafe_data.sizes["axis"]
        2
        """
        from .ops import concat_core

        out = concat_core(
            aos,
            opts=opts,
            validate=validate,
        )
        if cls is Array:
            return out
        return cls._from_validated(out.unsafe_data)

    def decompose_core(
        self,
        *,
        opts: "CoreDecomposeOptions",
        validate: bool = True,
    ) -> dict[tuple[object, ...], "Array"]:
        """Decompose selected core dimensions into keyed outputs.

        Parameters
        ----------
        opts : CoreDecomposeOptions
            Required decomposition options. Set ``core_dims`` to the dims to split; ``key_mode`` controls whether output keys use indices or labels; ``output_var`` overrides the emitted variable name.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        dict[tuple[object, ...], 'Array']
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
        >>> from tal.core import AnalysisObject, CoreDecomposeOptions
        >>> from tal.linalg import Array
        >>> arr = Array(AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}), sequence_dim="sample", core_dims=("axis",), validate=True))
        >>> sorted(arr.decompose_core(opts=CoreDecomposeOptions(core_dims=("axis",), key_mode="label")))
        [('x',), ('y',)]
        """
        from .ops import decompose_core

        return decompose_core(
            self,
            opts=opts,
            validate=validate,
        )

    def overlay_core(
        self,
        patches: Sequence[object] | object,
        *,
        opts: "CoreOverlayOptions",
        validate: bool = True,
    ) -> "Array":
        """Overlay patches along one core dimension.

        Parameters
        ----------
        patches : Sequence[object] | object
            Patch payloads applied by overlay operations.
        opts : CoreOverlayOptions
            Required overlay options. Set ``core_dim`` to the overlay axis; ``on_overlap`` chooses error vs replace behavior for duplicate labels; ``output_var`` overrides the emitted variable name.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Array
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
        >>> from tal.core import AnalysisObject, CoreOverlayOptions
        >>> from tal.linalg import Array
        >>> base = Array(AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}), sequence_dim="sample", core_dims=("axis",), validate=True))
        >>> patch = Array(AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[9.0]])}, coords={"sample": [0], "axis": ["y"]}), sequence_dim="sample", core_dims=("axis",), validate=True))
        >>> base.overlay_core([patch], opts=CoreOverlayOptions(core_dim="axis", on_overlap="replace")).unsafe_data["v"].sel(axis="y").item()
        9.0
        """
        from .ops import overlay_core

        return overlay_core(
            self,
            patches,
            opts=opts,
            validate=validate,
        )


__all__ = [
    "Array",
]
