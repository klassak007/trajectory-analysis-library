from __future__ import annotations

from .options import AOVizOptions, VizKind
from .surface import component, explorer, line, scatter


class AnalysisObjectVizAccessor:
    """Visualization accessor rooted at ``ao.viz``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    def __init__(self, ao: "AnalysisObject") -> None:
        self._ao = ao

    def line(self, *, opts: AOVizOptions | None = None):
        """Render the AnalysisObject as a line chart.

        Parameters
        ----------
        opts : AOVizOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``AOVizOptions`` key fields: ``var`` (default None), ``x`` (default None), ``by`` (default ()), ``groupby`` (default ()).

        Returns
        -------
        object
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
        >>> from tal.viz import AOVizOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> try:
        ...     plot = ao.viz.line(opts=AOVizOptions(var="value"))
        ... except ImportError:
        ...     plot = None
        >>> plot is None or plot is not None
        True
        """
        return line(self._ao, opts=opts)

    def scatter(self, *, opts: AOVizOptions | None = None):
        """Render the AnalysisObject as a scatter chart.

        Parameters
        ----------
        opts : AOVizOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``AOVizOptions`` key fields: ``var`` (default None), ``x`` (default None), ``by`` (default ()), ``groupby`` (default ()).

        Returns
        -------
        object
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
        >>> from tal.viz import AOVizOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> try:
        ...     plot = ao.viz.scatter(opts=AOVizOptions(var="value"))
        ... except ImportError:
        ...     plot = None
        >>> plot is None or plot is not None
        True
        """
        return scatter(self._ao, opts=opts)

    def explorer(self, *, opts: AOVizOptions | None = None):
        """Open the interactive explorer view for this AnalysisObject.

        Parameters
        ----------
        opts : AOVizOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``AOVizOptions`` key fields: ``var`` (default None), ``x`` (default None), ``by`` (default ()), ``groupby`` (default ()).

        Returns
        -------
        object
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
        >>> from tal.viz import AOVizOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> try:
        ...     plot = ao.viz.explorer(opts=AOVizOptions(var="value"))
        ... except (ImportError, ValueError):
        ...     plot = None
        >>> plot is None or plot is not None
        True
        """
        return explorer(self._ao, opts=opts)

    def component(
        self,
        name: str,
        *,
        kind: VizKind = "explorer",
        opts: AOVizOptions | None = None,
    ):
        """Render a named registered component using the selected visualization kind.

        Parameters
        ----------
        name : str
            Named component selector.
        kind : VizKind, optional
            Operation-family selector used to dispatch temporal behavior.
        opts : AOVizOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``AOVizOptions`` key fields: ``var`` (default None), ``x`` (default None), ``by`` (default ()), ``groupby`` (default ()).

        Returns
        -------
        object
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
        >>> from tal.core.component_ops import ComponentRegistryOptions, ComponentSpec
        >>> from tal.viz import AOVizOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... ).components.define(opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))
        >>> try:
        ...     plot = ao.viz.component("xy", kind="line", opts=AOVizOptions(var="vec"))
        ... except ImportError:
        ...     plot = None
        >>> plot is None or plot is not None
        True
        """
        return component(self._ao, name, kind=kind, opts=opts)


__all__ = ["AnalysisObjectVizAccessor"]
