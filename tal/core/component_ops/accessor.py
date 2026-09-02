from __future__ import annotations

from collections.abc import Mapping

from .compose import compose_components
from .extract import extract_components
from .patch import patch_components
from .registry import define_components, read_components
from .types import (
    ComponentComposeOptions,
    ComponentExtractOptions,
    ComponentPatchOptions,
    ComponentRegistryOptions,
    ComponentSpec,
)


class ComponentsAccessor:
    """Component registry/runtime accessor rooted at ``ao.components``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    def __init__(self, ao: "AnalysisObject") -> None:
        self._ao = ao

    def define(
        self,
        *,
        opts: ComponentRegistryOptions,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Define/replace component registry metadata on the AO.

        Parameters
        ----------
        opts : ComponentRegistryOptions
            Required component registry options. ``registry`` provides the canonical component mapping; ``replace`` controls whether existing registry entries are replaced or merged.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

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
        >>> from tal.core import AnalysisObject
        >>> from tal.core.component_ops import ComponentRegistryOptions, ComponentSpec
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... )
        >>> tagged = ao.components.define(opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))
        >>> tagged.components.registry()["xy"].core_dim
        'axis'
        """
        return define_components(self._ao, opts=opts, validate=validate)

    def registry(self) -> Mapping[str, ComponentSpec]:
        """Read component registry as an immutable mapping.

        Returns
        -------
        Mapping[str, ComponentSpec]
            Component names mapped to their core-dimension, label, and optional
            variable specifications.

        Notes
        -----
        The returned mapping is a read-only view of component metadata. Use
        :meth:`define` to replace or merge registry entries.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.core.component_ops import ComponentRegistryOptions, ComponentSpec
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... )
        >>> tagged = ao.components.define(
        ...     opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))})
        ... )
        >>> tagged.components.registry()["xy"].labels
        ('x', 'y')
        """
        return read_components(self._ao)

    def extract(
        self,
        *,
        opts: ComponentExtractOptions | None = None,
        validate: bool = True,
    ) -> dict[str, "AnalysisObject"]:
        """Extract registry components into AO outputs keyed by component name.

        Parameters
        ----------
        opts : ComponentExtractOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``ComponentExtractOptions`` key fields: ``names`` (default None), ``output_var`` (default None).
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        dict[str, 'AnalysisObject']
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
        >>> from tal.core.component_ops import ComponentExtractOptions, ComponentRegistryOptions, ComponentSpec
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... )
        >>> tagged = ao.components.define(opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))
        >>> parts = tagged.components.extract(opts=ComponentExtractOptions(names=("xy",)))
        >>> parts["xy"].as_dataset()["vec"].sizes["axis"]
        2
        """
        return extract_components(self._ao, opts=opts, validate=validate)

    def patch(
        self,
        components: Mapping[str, object],
        *,
        opts: ComponentPatchOptions,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Patch component labels into a base AO using registry semantics.

        Parameters
        ----------
        components : Mapping[str, object]
            Mapping of component names to AO-like values to patch/compose.
        opts : ComponentPatchOptions
            Required patch options. ``on_overlap`` chooses conflict policy for label collisions and ``output_var`` overrides the patched variable name.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

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
        >>> from tal.core import AnalysisObject
        >>> from tal.core.component_ops import ComponentPatchOptions, ComponentRegistryOptions, ComponentSpec
        >>> base = AnalysisObject.from_data(
        ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... ).components.define(opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))
        >>> patch = AnalysisObject.from_data(
        ...     xr.Dataset({"vec": (("sample", "axis"), [[9.0, 8.0]])}, coords={"sample": [0], "axis": ["x", "y"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... )
        >>> out = base.components.patch({"xy": patch}, opts=ComponentPatchOptions(on_overlap="replace"))
        >>> out.as_dataset()["vec"].sel(axis="x").item()
        9.0
        """
        return patch_components(self._ao, components, opts=opts, validate=validate)

    def compose(
        self,
        components: Mapping[str, object],
        *,
        opts: ComponentComposeOptions,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Compose components into a single AO using registry semantics.

        Parameters
        ----------
        components : Mapping[str, object]
            Mapping of component names to AO-like values to patch/compose.
        opts : ComponentComposeOptions
            Required compose options. ``registry`` defines component dims/labels for assembly and ``output_var`` overrides the composed variable name.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

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
        >>> from tal.core import AnalysisObject
        >>> from tal.core.component_ops import ComponentComposeOptions, ComponentSpec
        >>> x = AnalysisObject.from_data(
        ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0]])}, coords={"sample": [0], "axis": ["x"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... )
        >>> y = AnalysisObject.from_data(
        ...     xr.Dataset({"vec": (("sample", "axis"), [[2.0]])}, coords={"sample": [0], "axis": ["y"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... )
        >>> opts = ComponentComposeOptions({"x": ComponentSpec("axis", ("x",)), "y": ComponentSpec("axis", ("y",))})
        >>> out = x.components.compose({"x": x, "y": y}, opts=opts)
        >>> tuple(out.as_dataset().coords["axis"].values.tolist())
        ('x', 'y')
        """
        return compose_components(components, opts=opts, validate=validate)


__all__ = ["ComponentsAccessor"]
