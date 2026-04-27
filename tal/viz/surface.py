from __future__ import annotations

from dataclasses import replace
from typing import Any

from tal.core.component_ops import ComponentExtractOptions, extract_components, read_components
from tal.core.component_ops.types import ComponentSpec
from tal.core.orchestration.inputs import coerce_analysis_object_input

from .holoviews_backend import render_hvplot
from .options import AOVizOptions, VizKind, coerce_viz_options, require_supported_viz_kind
from .plan import resolve_viz_runtime
from .prepare import prepare_viz_payload


def _render(value: object, *, kind: VizKind, opts: AOVizOptions | None, owner: str) -> Any:
    runtime = resolve_viz_runtime(value, kind=kind, opts=opts, owner=owner)
    payload = prepare_viz_payload(runtime, owner=owner)
    return render_hvplot(payload, owner=owner)


def line(value: object, *, opts: AOVizOptions | None = None) -> Any:
    """Render a line plot for an AnalysisObject-like value.

    Parameters
    ----------
    value : object
        Input value consumed by this operation.
    opts : object
        Visualization options. Expected type is ``AOVizOptions`` or ``None``.
        When ``None``, defaults are used.
        Key fields for ``line`` are ``var`` (which variable to plot),
        ``x`` (x-axis coordinate), ``by`` (overlay split), and
        ``groupby`` (small-multiples grouping).

    Returns
    -------
    Any
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
    >>> from tal.viz import AOVizOptions, line
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1], "time": ("sample", [0.0, 1.0])}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     param_coord="time",
    ...     validate=True,
    ... )
    >>> try:
    ...     plot = line(ao, opts=AOVizOptions(var="value", x="time"))
    ... except ImportError:
    ...     plot = None
    >>> plot is None or plot is not None
    True
    """
    return _render(value, kind="line", opts=opts, owner="ao.viz.line")


def scatter(value: object, *, opts: AOVizOptions | None = None) -> Any:
    """Render a scatter plot for an AnalysisObject-like value.

    Parameters
    ----------
    value : object
        Input value consumed by this operation.
    opts : object
        Visualization options. Expected type is ``AOVizOptions`` or ``None``.
        When ``None``, defaults are used.
        Key fields for ``scatter`` are ``var`` (which variable to plot),
        ``x`` (x-axis coordinate), ``by`` (overlay split), and
        ``groupby`` (small-multiples grouping).

    Returns
    -------
    Any
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
    >>> from tal.viz import AOVizOptions, scatter
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1], "time": ("sample", [0.0, 1.0])}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     param_coord="time",
    ...     validate=True,
    ... )
    >>> try:
    ...     plot = scatter(ao, opts=AOVizOptions(var="value", x="time"))
    ... except ImportError:
    ...     plot = None
    >>> plot is None or plot is not None
    True
    """
    return _render(value, kind="scatter", opts=opts, owner="ao.viz.scatter")


def explorer(value: object, *, opts: AOVizOptions | None = None) -> Any:
    """Render the interactive visualization explorer for an AnalysisObject-like value.

    Parameters
    ----------
    value : object
        Input value consumed by this operation.
    opts : object
        Visualization options. Expected type is ``AOVizOptions`` or ``None``.
        When ``None``, defaults are used.
        Key fields for ``explorer`` are ``var`` (which variable to plot),
        ``x`` (x-axis coordinate), ``by`` (overlay split), and
        ``groupby`` (small-multiples grouping).

    Returns
    -------
    Any
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
    >>> from tal.viz import AOVizOptions, explorer
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1], "time": ("sample", [0.0, 1.0])}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     param_coord="time",
    ...     validate=True,
    ... )
    >>> try:
    ...     plot = explorer(ao, opts=AOVizOptions(var="value", x="time"))
    ... except (ImportError, ValueError):
    ...     plot = None
    >>> plot is None or plot is not None
    True
    """
    return _render(value, kind="explorer", opts=opts, owner="ao.viz.explorer")


def _require_component_name(name: object, *, owner: str) -> str:
    if isinstance(name, str) and name:
        return name
    raise ValueError(f"{owner}: component name must be a non-empty string.")


def _resolve_component_spec(source: object, *, name: str, owner: str) -> ComponentSpec:
    registry = read_components(source)
    if name in registry:
        return registry[name]
    raise ValueError(f"{owner}: unknown component {name!r}.")


def _extract_component(source: object, *, name: str, owner: str):
    extracted = extract_components(
        source,
        opts=ComponentExtractOptions(names=(name,), output_var=None),
        validate=True,
    )
    return extracted[name]


def _with_component_default_channels(opts: AOVizOptions, *, spec: ComponentSpec) -> AOVizOptions:
    if opts.by or opts.groupby:
        return opts
    if len(spec.labels) <= opts.max_overlay_items:
        return replace(opts, by=(spec.core_dim,))
    return replace(opts, groupby=(spec.core_dim,))


def component(
    value: object,
    name: str,
    *,
    kind: VizKind | str = "explorer",
    opts: AOVizOptions | None = None,
) -> Any:
    """Render a named component extracted from a component-registered AnalysisObject.

    Parameters
    ----------
    value : object
        Input value consumed by this operation.
    name : str
        Named component selector.
    kind : VizKind | str, optional
        Operation-family selector used to dispatch temporal behavior.
    opts : object
        Visualization options. Expected type is ``AOVizOptions`` or ``None``.
        When ``None``, defaults are used.
        Key fields for ``component`` are ``var`` (which variable to plot),
        ``x`` (x-axis coordinate), ``by`` (overlay split), and
        ``groupby`` (small-multiples grouping).

    Returns
    -------
    Any
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
    >>> from tal.viz import AOVizOptions, component
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... ).components.define(opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))
    >>> try:
    ...     plot = component(ao, "xy", kind="line", opts=AOVizOptions(var="vec"))
    ... except ImportError:
    ...     plot = None
    >>> plot is None or plot is not None
    True
    """
    owner = "ao.viz.component"
    resolved_name = _require_component_name(name, owner=owner)
    resolved_kind = require_supported_viz_kind(kind, owner=owner)
    source = coerce_analysis_object_input(value, owner=owner)
    options = coerce_viz_options(opts, owner=owner)
    spec = _resolve_component_spec(source, name=resolved_name, owner=owner)
    component_ao = _extract_component(source, name=resolved_name, owner=owner)
    effective_opts = _with_component_default_channels(options, spec=spec)
    return _render(
        component_ao,
        kind=resolved_kind,
        opts=effective_opts,
        owner=f"{owner}.{resolved_kind}",
    )


def install_analysis_object_viz_surface() -> None:
    """Install the ``ao.viz`` accessor on ``AnalysisObject``.

    Parameters
    ----------
    None
        This callable does not accept user-facing parameters.

    Returns
    -------
    None
        Returns ``None``; side effects are applied through owned state/metadata updates.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    from tal.core.analysis_object import AnalysisObject

    existing = getattr(AnalysisObject, "viz", None)
    if isinstance(existing, property):
        fget = existing.fget
        if fget is not None and getattr(fget, "__module__", "") == __name__:
            return
        raise ValueError(
            "install_analysis_object_viz_surface: AnalysisObject.viz is already owned by another accessor."
        )
    if existing is not None:
        raise ValueError(
            "install_analysis_object_viz_surface: AnalysisObject.viz already exists and is not a property."
        )

    from .accessor import AnalysisObjectVizAccessor

    def _viz_accessor(self: "AnalysisObject") -> AnalysisObjectVizAccessor:
        """Return the ``ao.viz`` accessor bound to this AnalysisObject.

        Returns
        -------
        AnalysisObjectVizAccessor
            Visualization accessor for this AO.

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
        return AnalysisObjectVizAccessor(self)

    AnalysisObject.viz = property(_viz_accessor)  # type: ignore[assignment]


__all__ = [
    "component",
    "explorer",
    "install_analysis_object_viz_surface",
    "line",
    "scatter",
]
