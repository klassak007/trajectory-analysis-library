from __future__ import annotations

from collections.abc import Mapping

import xarray as xr

from ..analysis_object import AnalysisObject
from ..orchestration.finalize import finalize_like
from ..orchestration.inputs import coerce_analysis_object_input
from ..schema import merge_schema
from .options import coerce_component_extract_options
from .registry import read_components
from .runtime_checks import require_core_dim_in_data, select_component_var
from .types import ComponentExtractOptions, ComponentSpec


def _resolve_component_names(
    registry: Mapping[str, ComponentSpec],
    *,
    names: tuple[str, ...] | None,
    owner: str,
) -> tuple[str, ...]:
    if names is None:
        return tuple(registry.keys())
    missing = [name for name in names if name not in registry]
    if missing:
        raise ValueError(f"{owner}: unknown component names requested: {missing!r}.")
    return names


def _extract_component_dataset(
    ds: xr.Dataset,
    *,
    spec: ComponentSpec,
    var_name: str,
) -> xr.Dataset:
    data = ds[var_name]
    require_core_dim_in_data(
        data,
        dim=spec.core_dim,
        owner="components.extract",
        operand=f"selected var {var_name!r}",
    )
    selected = data.sel({spec.core_dim: list(spec.labels)})
    return selected.to_dataset(name=var_name)


def _attach_source_schema(
    ds_out: xr.Dataset,
    *,
    source: AnalysisObject,
) -> xr.Dataset:
    tal = source.unsafe_data.attrs.get("tal")
    if not isinstance(tal, Mapping):
        return ds_out
    return merge_schema(ds_out, patch=dict(tal), validate=False)


def _rename_output_var_after_finalize(
    result: AnalysisObject,
    *,
    source_var: str,
    output_var: str | None,
    validate: bool,
) -> AnalysisObject:
    if output_var is None or output_var == source_var:
        return result
    return result.rename({source_var: output_var}, validate=validate)


def extract_components(
    ao: object,
    *,
    opts: ComponentExtractOptions | None = None,
    validate: bool = True,
) -> dict[str, AnalysisObject]:
    """Extract registered components from an AO.

    Parameters
    ----------
    ao : object
        AnalysisObject-like input value.
    opts : ComponentExtractOptions | None, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    dict[str, AnalysisObject]
        Mapping-like result produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.component_ops import ComponentExtractOptions, ComponentRegistryOptions, ComponentSpec
    >>> from tal.core.component_ops import extract_components
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... ).components.define(opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))
    >>> parts = extract_components(ao, opts=ComponentExtractOptions(names=("xy",)))
    >>> parts["xy"].unsafe_data["vec"].sizes["axis"]
    2
    """
    owner = "components.extract"
    source = coerce_analysis_object_input(ao, owner=owner)
    options = coerce_component_extract_options(opts, owner=owner)
    registry = read_components(source)
    names = _resolve_component_names(registry, names=options.names, owner=owner)
    out: dict[str, AnalysisObject] = {}
    for name in names:
        spec = registry[name]
        var_name = select_component_var(source.unsafe_data, spec=spec, component_name=name, owner=owner)
        ds_out = _extract_component_dataset(
            source.unsafe_data,
            spec=spec,
            var_name=var_name,
        )
        ds_out = _attach_source_schema(ds_out, source=source)
        finalized = finalize_like(source, ds_out, validate=validate, owner=owner)
        out[name] = _rename_output_var_after_finalize(
            finalized,
            source_var=var_name,
            output_var=options.output_var,
            validate=validate,
        )
    return out


__all__ = ["extract_components"]
