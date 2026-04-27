from __future__ import annotations

from typing import Any

import xarray as xr

from tal.core import AnalysisObject

from .backends import (
    datatree_extract_template,
    detect_catalog_backend,
    normalize_catalog_payload,
    resolve_catalog_batch_dim,
)
from .extract import extract_catalog_to_analysis_object
from .guardrails import raise_compute_blocked
from .options import (
    CatalogBackendOption,
    CatalogExtractOptions,
    CatalogQueryOptions,
    coerce_catalog_extract_options,
    coerce_catalog_init_options,
    coerce_catalog_query_options,
    coerce_extract_variables,
)
from .query import run_catalog_query
from .selection import (
    catalog_group_labels,
    coerce_head_tail_n,
    normalize_label_selector,
    normalize_position_selector,
    select_by_labels,
    select_by_positions,
)
from .types import CatalogState, make_catalog_state


class Catalog:
    """Browse-only catalog surface for grouped log-style data.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.catalog import Catalog
    >>> from tal.core import AnalysisObject
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> catalog = Catalog(ao, backend="dataset", batch_dim="run")
    >>> catalog.group_labels
    ('a', 'b')
    """

    __array_priority__ = 1000

    def __init__(
        self,
        value: object,
        *,
        backend: CatalogBackendOption = "auto",
        batch_dim: str | None = None,
    ) -> None:
        opts = coerce_catalog_init_options(
            backend=backend,
            batch_dim=batch_dim,
            owner="Catalog.__init__",
        )
        resolved_backend = detect_catalog_backend(value, backend=opts.backend, owner="Catalog.__init__")
        payload = normalize_catalog_payload(value, backend=resolved_backend, owner="Catalog.__init__")
        resolved_batch_dim, resolved_payload = resolve_catalog_batch_dim(
            payload,
            backend=resolved_backend,
            batch_dim=opts.batch_dim,
            owner="Catalog.__init__",
        )
        template = datatree_extract_template(resolved_payload) if resolved_backend == "datatree" else None
        self._state = make_catalog_state(
            backend=resolved_backend,
            batch_dim=resolved_batch_dim,
            data=resolved_payload,
            template=template,
        )

    @classmethod
    def _from_state(cls, state: CatalogState) -> Catalog:
        out = cls.__new__(cls)
        out._state = state
        return out

    @property
    def backend(self) -> str:
        """Return the normalized backend name for this catalog.

        Returns
        -------
        str
            Resolved property value.
        """
        return self._state.backend

    @property
    def batch_dim(self) -> str:
        """Return the batch dimension used to index catalog groups.

        Returns
        -------
        str
            Resolved property value.
        """
        return self._state.batch_dim

    @property
    def group_labels(self) -> tuple[object, ...]:
        """Return the current ordered catalog group labels.

        Returns
        -------
        tuple[object, ...]
            Resolved property value.
        """
        return catalog_group_labels(self._state)

    @property
    def data(self) -> xr.Dataset | xr.DataTree:
        """Return a deep copy of the underlying browse payload.

        Returns
        -------
        xr.Dataset | xr.DataTree
            Resolved property value.
        """
        return self._state.data.copy(deep=True)

    def sel(self, selector: object) -> Catalog:
        """Select catalog groups by batch labels.

        Parameters
        ----------
        selector : object
            Batch label, iterable of labels, slice, or xarray-style label
            selector.

        Returns
        -------
        Catalog
            Catalog containing the selected groups in label order.

        Notes
        -----
        Selection is performed against the catalog batch dimension labels, not
        against row positions.

        Examples
        --------
        >>> catalog.sel("run_0")  # doctest: +SKIP
        """
        labels = normalize_label_selector(selector, batch_dim=self.batch_dim, owner="Catalog.sel")
        return self._from_state(select_by_labels(self._state, labels, owner="Catalog.sel"))

    def isel(self, selector: object) -> Catalog:
        """Select catalog groups by integer positions on the batch axis.

        Parameters
        ----------
        selector : object
            Integer position, iterable of positions, or positional slice.

        Returns
        -------
        Catalog
            Catalog containing the selected groups in positional order.

        Notes
        -----
        Use :meth:`sel` for label-based selection.

        Examples
        --------
        >>> catalog.isel(0)  # doctest: +SKIP
        """
        parsed = normalize_position_selector(selector, batch_dim=self.batch_dim, owner="Catalog.isel")
        return self._from_state(select_by_positions(self._state, parsed, owner="Catalog.isel"))

    def head(self, n: int = 5) -> Catalog:
        """Return the first ``n`` catalog groups.

        Parameters
        ----------
        n : int, optional
            Number of leading groups to return.

        Returns
        -------
        Catalog
            Catalog containing at most ``n`` leading groups.

        Notes
        -----
        ``n=0`` returns an empty catalog with the same backend type.

        Examples
        --------
        >>> catalog.head(3)  # doctest: +SKIP
        """
        size = coerce_head_tail_n(n, owner="Catalog.head")
        return self.isel(slice(0, size))

    def tail(self, n: int = 5) -> Catalog:
        """Return the last ``n`` catalog groups.

        Parameters
        ----------
        n : int, optional
            Number of trailing groups to return.

        Returns
        -------
        Catalog
            Catalog containing at most ``n`` trailing groups.

        Notes
        -----
        ``n=0`` returns an empty catalog with the same backend type.

        Examples
        --------
        >>> catalog.tail(3)  # doctest: +SKIP
        """
        size = coerce_head_tail_n(n, owner="Catalog.tail")
        return self.isel(slice(0, 0) if size == 0 else slice(-size, None))

    def query(
        self,
        where: object = None,
        *,
        opts: CatalogQueryOptions | None = None,
        **filters: object,
    ) -> Catalog:
        """Filter catalog groups by predicates.

        Parameters
        ----------
        where : object
            Predicate expression used to filter catalog groups.
            Default is ``None``.
        opts : object
            Query option overrides. Expected type is
            ``CatalogQueryOptions`` or ``None``. When ``None``, defaults are used:
            ``unknown_field_policy='error'`` and
            ``metadata_eager_policy='forbid'``.
            Set ``unknown_field_policy='ignore'`` to skip unknown fields
            in ``where``/``filters``. Set ``metadata_eager_policy='allow'``
            to permit metadata-dependent query evaluation.
        filters : object
            Keyword filter predicates applied against batch coordinates/metadata fields.

        Returns
        -------
        Catalog
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
        >>> from tal.catalog import Catalog
        >>> from tal.catalog.options import CatalogQueryOptions
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
        ...     sequence_dim="sample",
        ...     batch_dims=("run",),
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> catalog = Catalog(ao, backend="dataset", batch_dim="run")
        >>> catalog.query(kind="sim", opts=CatalogQueryOptions()).group_labels
        ('a',)
        """
        query_opts = coerce_catalog_query_options(opts, owner="Catalog.query")
        state = run_catalog_query(
            self._state,
            where=where,
            filters=filters,
            options=query_opts,
            owner="Catalog.query",
        )
        return self._from_state(state)

    def extract(
        self,
        variables: object = None,
        *,
        opts: CatalogExtractOptions | None = None,
    ) -> AnalysisObject:
        """Materialize selected variables into an AnalysisObject.

        Parameters
        ----------
        variables : object, optional
            Requested variables for catalog extraction.
        opts : CatalogExtractOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``CatalogExtractOptions`` key fields: ``ignore_missing_vars`` (default False), ``require_sequence_size_coord`` (default False), ``metadata_promotion`` (default CatalogMetadataPromotionOptions(scalar_target='batch_coord', nonscalar_target='none')), ``validate`` (default True).

        Returns
        -------
        AnalysisObject
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
        >>> from tal.catalog import Catalog
        >>> from tal.catalog.options import CatalogExtractOptions
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1]}),
        ...     sequence_dim="sample",
        ...     batch_dims=("run",),
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> catalog = Catalog(ao, backend="dataset", batch_dim="run")
        >>> extracted = catalog.extract("value", opts=CatalogExtractOptions())
        >>> extracted.unsafe_data["value"].sizes["run"]
        2
        """
        extract_opts = coerce_catalog_extract_options(opts, owner="Catalog.extract")
        names = coerce_extract_variables(variables, owner="Catalog.extract")
        return extract_catalog_to_analysis_object(
            self._state,
            variables=names,
            options=extract_opts,
            owner="Catalog.extract",
        )

    def __repr__(self) -> str:
        return (
            f"Catalog(backend={self.backend!r}, batch_dim={self.batch_dim!r}, "
            f"groups={len(self.group_labels)})"
        )

    def _blocked(self, operation: str) -> None:
        raise_compute_blocked(owner=f"Catalog.{operation}", operation=operation)

    def __array__(self, dtype: Any | None = None, copy: bool | None = None) -> Any:
        _ = dtype
        _ = copy
        self._blocked("__array__")

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        _ = ufunc
        _ = method
        _ = inputs
        _ = kwargs
        self._blocked("__array_ufunc__")

    def __array_function__(self, func: Any, types: tuple[type, ...], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        _ = func
        _ = types
        _ = args
        _ = kwargs
        self._blocked("__array_function__")

    def __add__(self, other: object) -> Any:
        _ = other
        self._blocked("__add__")

    def __radd__(self, other: object) -> Any:
        _ = other
        self._blocked("__radd__")

    def __sub__(self, other: object) -> Any:
        _ = other
        self._blocked("__sub__")

    def __rsub__(self, other: object) -> Any:
        _ = other
        self._blocked("__rsub__")

    def __mul__(self, other: object) -> Any:
        _ = other
        self._blocked("__mul__")

    def __rmul__(self, other: object) -> Any:
        _ = other
        self._blocked("__rmul__")

    def __truediv__(self, other: object) -> Any:
        _ = other
        self._blocked("__truediv__")

    def __rtruediv__(self, other: object) -> Any:
        _ = other
        self._blocked("__rtruediv__")

    def __pow__(self, other: object) -> Any:
        _ = other
        self._blocked("__pow__")

    def __rpow__(self, other: object) -> Any:
        _ = other
        self._blocked("__rpow__")

    def __matmul__(self, other: object) -> Any:
        _ = other
        self._blocked("__matmul__")

    def __rmatmul__(self, other: object) -> Any:
        _ = other
        self._blocked("__rmatmul__")

    def __neg__(self) -> Any:
        self._blocked("__neg__")

    def __pos__(self) -> Any:
        self._blocked("__pos__")

    def __abs__(self) -> Any:
        self._blocked("__abs__")


__all__ = ["Catalog"]
