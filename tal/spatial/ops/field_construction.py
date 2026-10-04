"""Plan, assemble, and finalize explicit spatial scalar-field construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import xarray as xr

from tal.core import AnalysisLayoutSpec, AnalysisObject
from tal.core.component_ops.registry import _read_registry_from_dataset
from tal.core.dataset_ownership import (
    analysis_object_dataset,
    couple_dataset_resource,
    isolate_external_dataset,
    metadata_isolated_dataset,
)
from tal.core.orchestration.schema_finalize import (
    CoreSchemaFinalizeSpec,
    finalize_with_schema,
)
from tal.core.orchestration.topology import STRICT_NON_CORE_POLICY
from tal.core.schema import _SchemaUpdatePlan, merge_schema
from tal.core.schema_errors import SchemaError, _schema_error_with_context
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.core.schema_update import prepare_ingress_target
from tal.core.schema_validate import prepare_schema_validation
from tal.core.selected_ingress import (
    PreparedSelectedIngress,
    commit_prepared_selected_ingress,
    prepare_selected_ingress,
)

from ..association import finalize_spatial_as
from ..construction import (
    SpatialConstructionOverrides,
    SpatialConstructionPlan,
    apply_spatial_construction,
    preflight_spatial_construction,
    prepare_spatial_construction,
)
from ..metadata import (
    get_expressed_in,
    get_instantaneous_inertial,
    get_kinematics_kind,
    get_position_intent,
    set_position_rep,
    set_rotation_rep,
    validate_spatial_roles,
)
from ..metadata.representation import _validate_spatial_representation
from .field_selectors import (
    ExpandedFieldSelection,
    expand_field_selector,
    require_expanded_fields,
)
from .pose_component_ops import (
    build_components_pose_dataset,
    finalize_components_pose_output,
)

if TYPE_CHECKING:
    from tal.core.component_ops.types import ComponentSpec

    from ..field_recipes import SpatialFieldRecipe


@dataclass(frozen=True)
class PreparedFieldSource:
    """One validated, owned scalar-field projection and its source context."""

    selected: AnalysisObject
    resource_source: xr.Dataset | None
    sequence_dim: str | None
    batch_dims: tuple[str, ...]
    param_name: str | None
    size_name: str | None


@dataclass(frozen=True)
class PreparedFieldBuild:
    """Copy-neutral typed field plan ready for one ownership commit."""

    ingress: PreparedSelectedIngress
    selections: tuple[ExpandedFieldSelection, ...]
    source_ao: AnalysisObject | None
    raw_source: xr.Dataset
    construction: SpatialConstructionPlan
    sequence_dim: str | None
    batch_dims: tuple[str, ...]
    param_name: str | None
    size_name: str | None


def _require_source_layout(value: object, *, owner: str) -> AnalysisLayoutSpec | None:
    if value is None or isinstance(value, AnalysisLayoutSpec):
        return value
    raise TypeError(f"{owner}: source_layout must be AnalysisLayoutSpec or None.")


def _normalize_source(
    source: object,
    *,
    owner: str,
) -> tuple[xr.Dataset, AnalysisObject | None]:
    if isinstance(source, AnalysisObject):
        return analysis_object_dataset(source), source
    if isinstance(source, xr.DataArray):
        raise TypeError(f"{owner}: source must be an AnalysisObject or xarray.Dataset, not DataArray.")
    if isinstance(source, xr.Dataset):
        try:
            return AnalysisObject._normalized_ingress_dataset(source), None
        except SchemaError:
            raise
        except (TypeError, ValueError) as error:
            raise type(error)(f"{owner}: {error}") from error
    raise TypeError(f"{owner}: source must be an AnalysisObject or xarray.Dataset.")


def _layout_plan(layout: AnalysisLayoutSpec) -> _SchemaUpdatePlan:
    return _SchemaUpdatePlan(
        sequence_dim=layout.sequence_dim,
        batch_dims=layout.batch_dims,
        core_dims=layout.core_dims,
        param_coord=layout.param_coord,
        sequence_size_coord=layout.sequence_size_coord,
        complete_target=True,
    )


def _resolve_source_schema(
    ds: xr.Dataset,
    *,
    source_layout: AnalysisLayoutSpec | None,
    owner: str,
) -> tuple[xr.Dataset, Mapping[str, ComponentSpec]]:
    if "tal" in ds.attrs:
        prepare_schema_validation(ds)
        registry = _read_registry_from_dataset(ds, owner=owner)
        if source_layout is not None:
            raise ValueError(f"{owner}: source_layout cannot accompany a declared TAL layout.")
        return ds, registry
    if source_layout is None:
        raise ValueError(f"{owner}: an untagged Dataset requires source_layout.")
    if source_layout.core_dims:
        raise ValueError(f"{owner}: source_layout.core_dims must be empty for scalar fields.")
    candidate = prepare_ingress_target(ds, ds, _layout_plan(source_layout))
    return candidate, _read_registry_from_dataset(candidate, owner=owner)


def _scalar_projection_plan(ds: xr.Dataset) -> _SchemaUpdatePlan:
    _, sequence_dim, batch_dims, _ = read_roles(ds)
    return _SchemaUpdatePlan(
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=(),
        param_coord=read_param_coord_name(ds),
        sequence_size_coord=read_sequence_size_coord_name(ds),
        complete_target=True,
    )


def _semantic_layout(ds: xr.Dataset) -> tuple[str | None, tuple[str, ...]]:
    _, sequence_dim, batch_dims, _ = read_roles(ds)
    return sequence_dim, batch_dims


def _require_real_scalar_channels(
    ds: xr.Dataset,
    names: tuple[str, ...],
    *,
    semantic_dims: tuple[str, ...],
    owner: str,
) -> None:
    expected = set(semantic_dims)
    for name in names:
        data = ds[name]
        if len(data.dims) != len(expected) or set(data.dims) != expected:
            raise ValueError(
                f"{owner}: selected field {name!r} must use exactly semantic dimensions "
                f"{semantic_dims!r}; got {data.dims!r}."
            )
        if data.dtype.kind not in "iuf":
            raise TypeError(f"{owner}: selected field {name!r} must have a real numeric dtype.")


def _require_generated_namespace(
    ds: xr.Dataset,
    generated: tuple[str, ...],
    *,
    owner: str,
) -> None:
    for name in generated:
        if name not in ds.coords and name not in ds.dims:
            continue
        raise ValueError(
            f"{owner}: generated output name {name!r} conflicts with surviving source "
            "topology; rename the source dimension or coordinate explicitly."
        )


def _canonical_sources(
    selection: ExpandedFieldSelection,
    labels: tuple[str, ...],
) -> tuple[str, ...]:
    by_label = dict(selection.fields)
    return tuple(by_label[label] for label in labels)


def _component_specification(
    selection: ExpandedFieldSelection,
) -> tuple[tuple[str, ...], str, str]:
    if selection.slot == "position":
        return ("x", "y", "z"), "axis", "position"
    return ("x", "y", "z", "w"), "quat", "rotation"


def _preflight_components(
    ds: xr.Dataset,
    selections: tuple[ExpandedFieldSelection, ...],
    *,
    semantic_dims: tuple[str, ...],
    owner: str,
) -> None:
    for selection in selections:
        labels, core_dim, output_var = _component_specification(selection)
        names = _canonical_sources(selection, labels)
        _require_real_scalar_channels(ds, names, semantic_dims=semantic_dims, owner=owner)
        _require_generated_namespace(ds, (output_var, core_dim), owner=owner)


def _preflight_source_spatial_metadata(ds: xr.Dataset, *, owner: str) -> None:
    """Validate known source spatial declarations before target projection."""
    _validate_spatial_representation(ds, owner=owner)
    validate_spatial_roles(ds, owner=owner)
    get_position_intent(ds, owner=owner)
    get_kinematics_kind(ds, owner=owner)
    get_expressed_in(ds, owner=owner)
    get_instantaneous_inertial(ds, owner=owner)


def _prepare_field_build(
    raw: xr.Dataset,
    source_ao: AnalysisObject | None,
    names: tuple[str, ...],
    selections: tuple[ExpandedFieldSelection, ...],
    *,
    source_layout: AnalysisLayoutSpec | None,
    overrides: SpatialConstructionOverrides,
    owner: str,
) -> PreparedFieldBuild:
    schema_source, registry = _resolve_source_schema(
        raw, source_layout=source_layout, owner=owner
    )
    _preflight_source_spatial_metadata(schema_source, owner=owner)
    ingress = prepare_selected_ingress(
        schema_source,
        names,
        schema_plan=_scalar_projection_plan(schema_source),
        validated_registry=registry,
        owner=owner,
    )
    sequence_dim, batch_dims = _semantic_layout(ingress.target)
    semantic_dims = batch_dims + ((sequence_dim,) if sequence_dim else ())
    _preflight_components(
        ingress.target, selections, semantic_dims=semantic_dims, owner=owner
    )
    provisional = AnalysisObject._from_unvalidated(
        ingress.target, schema_prepared=True
    )
    association_source = source_ao if source_ao is not None else provisional
    construction = prepare_spatial_construction(
        (association_source,), overrides=overrides, owner=owner
    )
    return PreparedFieldBuild(
        ingress=ingress,
        selections=selections,
        source_ao=source_ao,
        raw_source=raw,
        construction=construction,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        param_name=read_param_coord_name(ingress.target),
        size_name=read_sequence_size_coord_name(ingress.target),
    )


def _without_consumed_field_metadata(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy(deep=False)
    for name in out.data_vars:
        out[name].attrs = {}
        out[name].encoding = {}
    return out


def _commit_field_source(plan: PreparedFieldBuild, *, owner: str) -> PreparedFieldSource:
    updated = commit_prepared_selected_ingress(plan.ingress)
    updated = _without_consumed_field_metadata(updated)
    if plan.source_ao is None:
        projected = isolate_external_dataset(updated)
        resource_source = None
    else:
        projected = metadata_isolated_dataset(updated, owner=owner)
        resource_source = plan.raw_source
    return PreparedFieldSource(
        selected=AnalysisObject._from_unvalidated(projected, schema_prepared=True),
        resource_source=resource_source,
        sequence_dim=plan.sequence_dim,
        batch_dims=plan.batch_dims,
        param_name=plan.param_name,
        size_name=plan.size_name,
    )


def _pack_component(
    source: PreparedFieldSource,
    selection: ExpandedFieldSelection,
    *,
    labels: tuple[str, ...],
    core_dim: str,
    output_var: str,
    owner: str,
) -> AnalysisObject:
    ds = analysis_object_dataset(source.selected)
    semantic_dims = source.batch_dims + ((source.sequence_dim,) if source.sequence_dim else ())
    names = _canonical_sources(selection, labels)
    parts = [ds[name].variable.transpose(*semantic_dims) for name in names]
    packed = xr.Variable.concat(parts, dim=core_dim).transpose(*semantic_dims, core_dim)
    packed.attrs = {}
    packed.encoding = {}
    base = ds.drop_vars(tuple(ds.data_vars)).assign_coords({core_dim: list(labels)})
    candidate = base.assign({output_var: packed})
    spec = CoreSchemaFinalizeSpec(
        sequence_dim=source.sequence_dim,
        batch_dims=source.batch_dims,
        core_dims=(core_dim,),
        param_name=source.param_name,
        size_name=source.size_name,
    )
    return finalize_with_schema(
        source.selected,
        candidate,
        spec=spec,
        validate=True,
        owner=owner,
    )


def _translate_typed_error(owner: str, error: TypeError | ValueError) -> Exception:
    if str(error).startswith(f"{owner}:"):
        return error
    return type(error)(f"{owner}: {error}")


def _project_target_spatial_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Remove source-type spatial declarations from a configuration target."""
    tal = ds.attrs.get("tal")
    ext = tal.get("ext") if isinstance(tal, Mapping) else None
    spatial = ext.get("spatial") if isinstance(ext, Mapping) else None
    if not isinstance(spatial, Mapping):
        return ds
    patch: dict[str, object] = {}
    if "roles" in spatial:
        patch["roles"] = None
    relation = spatial.get("relation")
    if isinstance(relation, Mapping) and "instantaneous_inertial" in relation:
        patch["relation"] = {"instantaneous_inertial": None}
    if not patch:
        return ds
    return merge_schema(ds, {"ext": {"spatial": patch}}, validate=False)


def _finalize_leaf(
    target: str,
    component: AnalysisObject,
    *,
    plan: SpatialConstructionPlan,
    owner: str,
):
    from ..position import Position
    from ..rotation import Rotation

    cls = Position if target == "position" else Rotation
    rep_owner = set_position_rep if target == "position" else set_rotation_rep
    rep = "cart" if target == "position" else "quat"
    target_owned = rep_owner(
        _project_target_spatial_metadata(analysis_object_dataset(component)),
        rep=rep,
        validate=False,
        owner=owner,
    )
    prepared = apply_spatial_construction(
        target_owned,
        plan=plan,
        owner=owner,
    )
    try:
        return finalize_spatial_as(cls, prepared, validate=True, association=plan.association)
    except SchemaError:
        raise
    except (TypeError, ValueError) as error:
        translated = _translate_typed_error(owner, error)
        if translated is error:
            raise
        raise translated from error


def _component_for_slot(
    source: PreparedFieldSource,
    selection: ExpandedFieldSelection,
    *,
    owner: str,
) -> AnalysisObject:
    if selection.slot == "position":
        return _pack_component(
            source,
            selection,
            labels=("x", "y", "z"),
            core_dim="axis",
            output_var="position",
            owner=owner,
        )
    return _pack_component(
        source,
        selection,
        labels=("x", "y", "z", "w"),
        core_dim="quat",
        output_var="rotation",
        owner=owner,
    )


def _finalize_pose(
    source: PreparedFieldSource,
    selections: tuple[ExpandedFieldSelection, ...],
    *,
    plan: SpatialConstructionPlan,
    owner: str,
):
    by_slot = {selection.slot: selection for selection in selections}
    position = _component_for_slot(source, by_slot["position"], owner=owner)
    rotation = _component_for_slot(source, by_slot["rotation"], owner=owner)
    try:
        combined = build_components_pose_dataset(
            analysis_object_dataset(rotation),
            analysis_object_dataset(position),
            owner=owner,
            policy=STRICT_NON_CORE_POLICY,
            metadata_source=analysis_object_dataset(source.selected),
        )
        return finalize_components_pose_output(
            combined,
            plan=plan,
            validate=True,
            owner=owner,
            metadata_isolated=True,
        )
    except SchemaError:
        raise
    except (TypeError, ValueError) as error:
        translated = _translate_typed_error(owner, error)
        if translated is error:
            raise
        raise translated from error


def _finish_resource(source: PreparedFieldSource, result: object):
    if source.resource_source is None:
        return result
    couple_dataset_resource(source.resource_source, analysis_object_dataset(result))
    return result


def _finalize_field_build(
    target: str,
    plan: PreparedFieldBuild,
    *,
    owner: str,
):
    prepared = _commit_field_source(plan, owner=owner)
    if target == "pose":
        result = _finalize_pose(
            prepared,
            plan.selections,
            plan=plan.construction,
            owner=owner,
        )
    else:
        component = _component_for_slot(prepared, plan.selections[0], owner=owner)
        result = _finalize_leaf(
            target,
            component,
            plan=plan.construction,
            owner=owner,
        )
    return _finish_resource(prepared, result)


def _execute_field_recipe(
    recipe: SpatialFieldRecipe[object],
    source: object,
    *,
    prefix: object,
    source_layout: object,
    parent: object,
    child: object,
    expressed_in: object,
    graph: object,
    owner: str,
):
    """Execute one field recipe through shared planning and finalization."""
    layout = _require_source_layout(source_layout, owner=owner)
    overrides: SpatialConstructionOverrides = preflight_spatial_construction(
        parent=parent,
        child=child,
        expressed_in=expressed_in,
        graph=graph,
        owner=owner,
    )
    selections = tuple(
        expand_field_selector(declaration, prefix=prefix, owner=owner)
        for declaration in recipe._selectors
    )
    names = tuple(dict.fromkeys(name for selection in selections for name in selection.source_names))
    raw, source_ao = _normalize_source(source, owner=owner)
    require_expanded_fields(raw, selections, owner=owner)
    prepared_plan = _prepare_field_build(
        raw,
        source_ao,
        names,
        selections,
        source_layout=layout,
        overrides=overrides,
        owner=owner,
    )
    return _finalize_field_build(recipe._target, prepared_plan, owner=owner)


def execute_field_recipe(
    recipe: SpatialFieldRecipe[object],
    source: object,
    *,
    prefix: object,
    source_layout: object,
    parent: object,
    child: object,
    expressed_in: object,
    graph: object,
    owner: str,
):
    """Execute one field recipe with a stable public error envelope."""
    try:
        return _execute_field_recipe(
            recipe,
            source,
            prefix=prefix,
            source_layout=source_layout,
            parent=parent,
            child=child,
            expressed_in=expressed_in,
            graph=graph,
            owner=owner,
        )
    except SchemaError as error:
        if error.hint.startswith(f"{owner}:"):
            raise
        context = f"{owner}: invalid source or target schema"
        raise _schema_error_with_context(error, context=context) from error


__all__ = ["execute_field_recipe"]
