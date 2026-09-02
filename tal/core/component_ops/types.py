from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ComponentSpec:
    """Component registry entry (core dimension, labels, optional variable).

    Notes
    -----
    A component spec names a slice of one core dimension. ``var`` is optional
    when the source AO has exactly one data variable; provide it when a
    component belongs to a specific variable in a multi-variable dataset.

    Examples
    --------
    >>> from tal.core import ComponentSpec
    >>> spec = ComponentSpec(core_dim="axis", labels=("x", "y"))
    >>> spec.labels
    ('x', 'y')
    """

    core_dim: str
    labels: tuple[object, ...]
    var: str | None = None


@dataclass(frozen=True)
class ComponentRegistryOptions:
    """Options for registry define/replace operations.

    Notes
    -----
    Pass these options to ``define_components`` or ``ao.components.define`` to
    attach a component registry to an AO.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, ComponentRegistryOptions, ComponentSpec, define_components
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... )
    >>> opts = ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))})
    >>> define_components(ao, opts=opts).components.registry()["xy"].labels
    ('x', 'y')
    """

    registry: Mapping[str, ComponentSpec]
    replace: bool = True


@dataclass(frozen=True)
class ComponentExtractOptions:
    """Options for component extraction.

    Notes
    -----
    ``names=None`` extracts every registered component. ``output_var`` can
    rename each single-variable extracted component after finalization.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, ComponentExtractOptions, ComponentRegistryOptions, ComponentSpec, define_components, extract_components
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... )
    >>> tagged = define_components(ao, opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))
    >>> parts = extract_components(tagged, opts=ComponentExtractOptions(names=("xy",)))
    >>> parts["xy"].as_dataset().sizes["axis"]
    2
    """

    names: tuple[str, ...] | None = None
    output_var: str | None = None


@dataclass(frozen=True)
class ComponentPatchOptions:
    """Options for patching component payloads.

    Notes
    -----
    Use ``on_overlap="replace"`` when patch labels intentionally replace
    existing labels in the registered component slice.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, ComponentPatchOptions, ComponentRegistryOptions, ComponentSpec, define_components, patch_components
    >>> base = AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... )
    >>> tagged = define_components(base, opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))
    >>> patch = AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[10.0, 20.0]])}, coords={"sample": [0], "axis": ["x", "y"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... )
    >>> out = patch_components(tagged, {"xy": patch}, opts=ComponentPatchOptions(on_overlap="replace"))
    >>> out.as_dataset()["vec"].sel(axis="x").item()
    10.0
    """

    on_overlap: Literal["error", "replace"] = "error"
    output_var: str | None = None


@dataclass(frozen=True)
class ComponentComposeOptions:
    """Options for composing component payloads into one AO.

    Notes
    -----
    The registry describes where each named component should land in the
    composed core layout.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, ComponentComposeOptions, ComponentSpec, compose_components
    >>> x = AnalysisObject.from_data(xr.Dataset({"vec": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> y = AnalysisObject.from_data(xr.Dataset({"vec": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> opts = ComponentComposeOptions({"xy": ComponentSpec("axis", ("x", "y"))})
    >>> out = compose_components({"xy": AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... )}, opts=opts)
    >>> out.as_dataset().sizes["axis"]
    2
    """

    registry: Mapping[str, ComponentSpec]
    output_var: str | None = None
