from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.component_ops import ComponentRegistryOptions, define_components
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.alignment import align_exact_for_plan
from tal.core.orchestration.alignment_intent import select_topology_policy_with_intents
from tal.core.orchestration.runtime_checks import (
    resolve_single_numeric_var_single_core_dim,
)
from tal.core.orchestration.topology import (
    SEMANTIC_NON_CORE_POLICY,
    STRICT_NON_CORE_POLICY,
    TopologyPolicy,
    resolve_binary_topology,
    resolve_nary_topology,
)
from tal.core.schema_errors import SchemaError
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.utils.frame_schema import get_frames, set_frames
from tal.utils.topology_operation_families import (
    operation_intent_support_for_operation_family,
)

from ..association import (
    SpatialAssociationPlan,
    associated_graph,
    attach_spatial_association,
    finalize_spatial_as,
    preserve_spatial_basis,
    resolve_passive_association,
)
from ..conversion.finalize import allocate_dim_pair, dataset_dim_names
from ..kernels.pose_kernels import _matrix_to_components_prevalidated_kernel
from ..metadata import get_pose_rep, set_pose_rep, set_position_rep
from ..policies.frame import (
    resolve_components_shared_frames,
    resolve_compose_output_frames,
)
from ..position import Position
from ..rotation import Rotation
from . import pose_context
from .frame_owner_common import (
    frame_inverse_component_datasets,
    require_parent_basis_for_inverse,
)
from .pose_kernel_adapters import (
    apply_components_to_matrix_kernel,
    apply_pose_compose_translation_kernel,
    apply_pose_inverse_translation_kernel,
)
from .pose_matrix_validation import (
    prepare_pose_matrix_for_conversion,
    validate_pose_matrix_dataset,
)

if TYPE_CHECKING:
    from ..pose import Pose

_POSE_REPS = {"components", "matrix"}
_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")
_QUAT_LABELS: tuple[str, str, str, str] = ("x", "y", "z", "w")
_MATRIX_LABELS: tuple[str, str, str, str] = ("x", "y", "z", "w")
# Architecture lock sentinels preserved while kernel wrappers are delegated.
_POSE_KERNEL_WRAP_SENTINEL = "pose components->matrix conversion kernel failed."
_POSE_COMPOSE_WRAP_SENTINEL = "pose compose translation kernel failed."
_POSE_INVERSE_WRAP_SENTINEL = "pose inverse translation failed."
_POSE_KERNEL_PARTIAL_SENTINEL = "partial(_wrap_compose_translation_kernel, owner=owner)"
_POSE_KERNEL_PARTIAL_INVERSE_SENTINEL = "partial(_wrap_inverse_translation_kernel, owner=owner)"


def _pose_cls() -> type["Pose"]:
    from ..pose import Pose

    return Pose


def _wrap_pose_output(
    ds: xr.Dataset,
    *,
    validate: bool,
    owner: str,
    association: SpatialAssociationPlan,
) -> "Pose":
    cls = _pose_cls()
    if not validate:
        return finalize_spatial_as(
            cls,
            ds,
            validate=False,
            association=association,
        )
    if get_pose_rep(ds, owner=owner) == "matrix":
        ds = validate_pose_matrix_dataset(ds, owner=owner)
    result = cls._from_rigid_validated(ds, owner=owner)
    return attach_spatial_association(result, association)


def _coerce_pose_operand(value: object, *, owner: str) -> "Pose":
    cls = _pose_cls()
    if isinstance(value, cls):
        return value
    try:
        return cls(value)
    except SchemaError:
        raise
    except TypeError as exc:
        raise TypeError(f"{owner}: pose operand must be Pose, AnalysisObject, xr.Dataset, or xr.DataArray.") from exc
    except ValueError as exc:
        raise ValueError(f"{owner}: pose operand is not a valid Pose: {exc}") from exc


def _normalize_target_rep(rep: str, *, owner: str) -> str:
    if not isinstance(rep, str):
        raise TypeError(f"{owner}: rep must be a string.")
    if rep not in _POSE_REPS:
        raise ValueError(f"{owner}: unsupported pose representation {rep!r}.")
    return rep


def _canonical_components(pose: "Pose", *, owner: str) -> tuple[Position, Rotation]:
    try:
        position, rotation = pose.decompose(validate=False)
    except ValueError as exc:
        raise ValueError(f"{owner}: canonical decomposition failed: {exc}") from exc
    return position, rotation.as_quat(validate=False)


def _build_pose_matrix_output(
    matrix: xr.DataArray,
    *,
    row_dim: str,
    col_dim: str,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    position_ds: xr.Dataset,
    rotation_ds: xr.Dataset,
    policy: TopologyPolicy,
    owner: str,
) -> xr.Dataset:
    param, size = pose_context.resolve_series_optional_coord_names(
        position_ds,
        rotation_ds,
        owner=owner,
        allow_one_sided_inherit=policy.mode == "semantic_broadcast",
    )
    param_name = param if sequence_dim is not None else None
    size_name = size if sequence_dim is not None else None
    kwargs: dict[str, object] = {
        "batch_dims": batch_dims,
        "core_dims": (row_dim, col_dim),
        "param_coord": param_name,
        "sequence_size_coord": size_name,
        "validate": False,
    }
    if sequence_dim is not None:
        kwargs["sequence_dim"] = sequence_dim
    base = AnalysisObject.from_data(
        matrix.to_dataset(name="pose_matrix"),
        **kwargs,
    )
    cleared = define_components(base, opts=ComponentRegistryOptions(registry={}, replace=True), validate=False)
    output = set_pose_rep(analysis_object_dataset(cleared), rep="matrix", validate=False, owner=owner)
    parent, child = resolve_components_shared_frames(
        rotation_ds,
        position_ds,
        owner=owner,
        left_name="rotation",
        right_name="position",
    )
    return set_frames(output, parent=parent, child=child, validate=False)


def _matrix_to_components_arrays(
    source: xr.Dataset,
    *,
    row_dim: str,
    col_dim: str,
    axis_dim: str,
    quat_dim: str,
    owner: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    matrix = prepare_pose_matrix_for_conversion(source, owner=owner)
    try:
        position_da, rotation_da = xr.apply_ufunc(
            _matrix_to_components_prevalidated_kernel,
            matrix,
            input_core_dims=[[row_dim, col_dim]],
            output_core_dims=[[axis_dim], [quat_dim]],
            vectorize=False,
            dask="parallelized",
            output_dtypes=[np.float64, np.float64],
            dask_gufunc_kwargs={"output_sizes": {axis_dim: 3, quat_dim: 4}},
        )
    except ValueError as exc:
        raise ValueError(f"{owner}: pose matrix->components conversion failed: {exc}") from exc
    position_da = position_da.assign_coords({axis_dim: list(_XYZ_LABELS)})
    rotation_da = rotation_da.assign_coords({quat_dim: list(_QUAT_LABELS)})
    return position_da, rotation_da


def _wrap_components_from_arrays(
    position_da: xr.DataArray,
    rotation_da: xr.DataArray,
    *,
    seq_dim: str | None,
    batch_dims: tuple[str, ...],
    axis_dim: str,
    quat_dim: str,
    source: xr.Dataset,
    owner: str,
) -> xr.Dataset:
    param = read_param_coord_name(source)
    size = read_sequence_size_coord_name(source)
    param_name = param if seq_dim is not None else None
    size_name = size if seq_dim is not None else None
    base_kwargs: dict[str, object] = {
        "batch_dims": batch_dims,
        "param_coord": param_name,
        "sequence_size_coord": size_name,
        "validate": False,
    }
    if seq_dim is not None:
        base_kwargs["sequence_dim"] = seq_dim
    pos_ao = AnalysisObject.from_data(
        position_da.to_dataset(name="position"),
        core_dims=(axis_dim,),
        **base_kwargs,
    )
    rot_ao = AnalysisObject.from_data(
        rotation_da.to_dataset(name="rotation"),
        core_dims=(quat_dim,),
        **base_kwargs,
    )
    parent, child = get_frames(source)
    pos_ds = set_frames(set_position_rep(analysis_object_dataset(pos_ao), rep="cart", validate=False, owner=owner), parent=parent, child=child, validate=False)
    rot_ds = set_frames(analysis_object_dataset(rot_ao), parent=parent, child=child, validate=False)
    components = _pose_cls().from_components(Rotation._from_unvalidated(rot_ds), Position._from_unvalidated(pos_ds), validate=False)
    return set_pose_rep(analysis_object_dataset(components), rep="components", validate=False, owner=owner)


def _components_to_matrix_dataset(pose: "Pose", *, owner: str) -> xr.Dataset:
    selection = select_topology_policy_with_intents(
        (pose,),
        owner=owner,
        operation_family="spatial.pose.components",
        support=operation_intent_support_for_operation_family(
            "spatial.pose.components",
            owner=owner,
        ),
        strict_policy=STRICT_NON_CORE_POLICY,
        semantic_policy=SEMANTIC_NON_CORE_POLICY,
    )
    policy = selection.policy
    position, rotation, pos_dim, quat_dim, pos_da, rot_da, sequence_dim, batch_dims = _resolve_components_to_matrix_inputs(
        pose,
        owner=owner,
        policy=policy,
    )
    row_dim, col_dim = allocate_dim_pair(
        existing_dims=dataset_dim_names(analysis_object_dataset(pose)),
        first_candidates=("row", "pose_row"),
        first_base="pose_row",
        first_what="pose row dim",
        second_candidates=("col", "pose_col"),
        second_base="pose_col",
        second_what="pose col dim",
        owner=owner,
    )
    matrix = apply_components_to_matrix_kernel(
        pos_da,
        rot_da,
        pos_dim=pos_dim,
        quat_dim=quat_dim,
        row_dim=row_dim,
        col_dim=col_dim,
        owner=owner,
    )
    return _build_pose_matrix_output(
        matrix,
        row_dim=row_dim,
        col_dim=col_dim,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        position_ds=analysis_object_dataset(position),
        rotation_ds=analysis_object_dataset(rotation),
        policy=policy,
        owner=owner,
    )


def _resolve_components_to_matrix_inputs(
    pose: "Pose",
    *,
    owner: str,
    policy: TopologyPolicy,
) -> tuple[Position, Rotation, str, str, xr.DataArray, xr.DataArray, str | None, tuple[str, ...]]:
    position, rotation = _canonical_components(pose, owner=owner)
    position_ds = analysis_object_dataset(position)
    rotation_ds = analysis_object_dataset(rotation)
    pos_var, pos_dim = resolve_single_numeric_var_single_core_dim(position_ds, owner=owner, what="Pose position")
    rot_var, quat_dim = resolve_single_numeric_var_single_core_dim(rotation_ds, owner=owner, what="Pose rotation")
    plan = resolve_binary_topology(
        pose_context.topology_operand(
            position_ds,
            var_name=pos_var,
            core_dims=(pos_dim,),
            index=0,
            owner=owner,
            what="Pose position",
            policy=policy,
        ),
        pose_context.topology_operand(
            rotation_ds,
            var_name=rot_var,
            core_dims=(quat_dim,),
            index=1,
            owner=owner,
            what="Pose rotation",
            policy=policy,
        ),
        owner=owner,
        what="pose components->matrix conversion",
        policy=policy,
    )
    pos_da, rot_da = align_exact_for_plan(
        plan,
        owner=owner,
        what="pose components->matrix conversion",
    )
    return position, rotation, pos_dim, quat_dim, pos_da, rot_da, plan.sequence_dim, plan.batch_dims


def _matrix_to_components_dataset(pose: "Pose", *, owner: str) -> xr.Dataset:
    source = analysis_object_dataset(pose)
    declared, seq_dim, batch_dims, core_dims = read_roles(source)
    if not declared:
        raise ValueError(f"{owner}: Pose matrix layout requires declared roles.")
    if len(core_dims) != 2:
        raise ValueError(f"{owner}: Pose matrix layout requires exactly two core dims.")
    row_dim, col_dim = core_dims
    axis_dim, quat_dim = allocate_dim_pair(
        existing_dims=dataset_dim_names(source),
        first_candidates=("axis", "position_axis"),
        first_base="position_axis",
        first_what="position axis dim",
        second_candidates=("quat", "rotation_quat"),
        second_base="rotation_quat",
        second_what="rotation quat dim",
        owner=owner,
    )
    position_da, rotation_da = _matrix_to_components_arrays(
        source,
        row_dim=row_dim,
        col_dim=col_dim,
        axis_dim=axis_dim,
        quat_dim=quat_dim,
        owner=owner,
    )
    return _wrap_components_from_arrays(
        position_da,
        rotation_da,
        seq_dim=seq_dim,
        batch_dims=batch_dims,
        axis_dim=axis_dim,
        quat_dim=quat_dim,
        source=source,
        owner=owner,
    )


def _pose_to_rep_dataset(pose: "Pose", *, target_rep: str, owner: str) -> xr.Dataset:
    source = analysis_object_dataset(pose)
    current = get_pose_rep(source, owner=owner)
    if current == target_rep:
        return source
    if target_rep == "components":
        return _matrix_to_components_dataset(pose, owner=owner)
    return _components_to_matrix_dataset(pose, owner=owner)


def pose_to_rep(pose: "Pose", rep: str, *, validate: bool) -> "Pose":
    owner = "spatial.pose.to_rep"
    target = _normalize_target_rep(rep, owner=owner)
    pose._enforce_invariants(owner=owner)
    output = _pose_to_rep_dataset(pose, target_rep=target, owner=owner)
    output = preserve_spatial_basis(pose, output, owner=owner)
    association = SpatialAssociationPlan(associated_graph(pose))
    return _wrap_pose_output(
        output,
        validate=validate,
        owner=owner,
        association=association,
    )


def pose_as_components(pose: "Pose", *, validate: bool) -> "Pose":
    return pose_to_rep(pose, rep="components", validate=validate)


def pose_as_matrix(pose: "Pose", *, validate: bool) -> "Pose":
    return pose_to_rep(pose, rep="matrix", validate=validate)


def pose_compose(pose: "Pose", other: object, *, validate: bool) -> "Pose":
    owner = "spatial.pose.compose"
    right = _coerce_pose_operand(other, owner=owner)
    return _pose_compose_with_owner(pose, right, validate=validate, owner=owner)


def _prepare_pose_compose_inputs(
    left_pos: Position,
    left_rot: Rotation,
    right_pos: Position,
    right_rot: Rotation,
    *,
    owner: str,
    policy: TopologyPolicy,
) -> pose_context.PreparedPoseComposeInputs:
    specs = pose_context.resolve_pose_compose_specs(left_pos, right_pos, right_rot, owner=owner)
    operands = pose_context.pose_compose_topology_operands(
        specs,
        left_pos,
        right_pos,
        right_rot,
        owner=owner,
        policy=policy,
    )
    plan = resolve_nary_topology(
        operands,
        owner=owner,
        what="pose compose",
        policy=policy,
    )
    left_t, right_t, right_q = align_exact_for_plan(plan, owner=owner, what="pose compose")
    return pose_context.PreparedPoseComposeInputs(
        specs=specs,
        left_translation=left_t,
        right_translation=right_t,
        right_quaternion=right_q,
        sequence_dim=plan.sequence_dim,
        batch_dims=plan.batch_dims,
    )


def _pose_compose_policy(*, pose: "Pose", right: "Pose", owner: str) -> TopologyPolicy:
    selection = select_topology_policy_with_intents(
        (pose, right),
        owner=owner,
        operation_family="spatial.pose.compose",
        support=operation_intent_support_for_operation_family("spatial.pose.compose", owner=owner),
        strict_policy=STRICT_NON_CORE_POLICY,
        semantic_policy=SEMANTIC_NON_CORE_POLICY,
    )
    return selection.policy


def _compose_rotation(left: Rotation, right: Rotation, *, owner: str) -> Rotation:
    try:
        return left.compose(right, validate=False)
    except ValueError as exc:
        raise ValueError(f"{owner}: pose rotation compose failed: {exc}") from exc


def _inverse_rotation(rotation: Rotation, *, owner: str) -> Rotation:
    try:
        return rotation.inverse(validate=False)
    except ValueError as exc:
        raise ValueError(f"{owner}: pose inverse rotation failed: {exc}") from exc


def _pose_compose_with_owner(
    pose: "Pose",
    right: "Pose",
    *,
    validate: bool,
    owner: str,
) -> "Pose":
    association = resolve_passive_association((pose, right), owner=owner)
    pose._enforce_invariants(owner=owner)
    right._enforce_invariants(owner=owner)
    left_rep = get_pose_rep(source := analysis_object_dataset(pose), owner=owner)
    parent, child = resolve_compose_output_frames(source, analysis_object_dataset(right), owner=owner)
    (left_pos, left_rot), (right_pos, right_rot) = _canonical_components(pose, owner=owner), _canonical_components(right, owner=owner)
    policy = _pose_compose_policy(pose=pose, right=right, owner=owner)
    prepared = _prepare_pose_compose_inputs(
        left_pos,
        left_rot,
        right_pos,
        right_rot,
        owner=owner,
        policy=policy,
    )
    out_t = apply_pose_compose_translation_kernel(
        prepared.left_translation,
        prepared.right_translation,
        prepared.right_quaternion,
        left_dim=prepared.specs.left_dim,
        right_dim=prepared.specs.right_dim,
        right_quat_dim=prepared.specs.right_quat_dim,
        owner=owner,
    )
    out_rot = _compose_rotation(left_rot, right_rot, owner=owner)
    out_pos_ds = pose_context.build_composed_position_dataset(
        prepared,
        out_t,
        left_pos,
        right_pos,
        frames=(parent, child),
        policy=policy,
        owner=owner,
    )
    out_rot_ds = set_frames(analysis_object_dataset(out_rot), parent=parent, child=child, validate=False)
    components = _pose_cls().from_components(Rotation._from_unvalidated(out_rot_ds), Position._from_unvalidated(out_pos_ds), validate=False)
    return _wrap_pose_output(
        _pose_to_rep_dataset(components, target_rep=left_rep, owner=owner),
        validate=validate,
        owner=owner,
        association=association,
    )


def pose_inverse(pose: "Pose", *, validate: bool) -> "Pose":
    owner = "spatial.pose.inverse"
    return _pose_inverse_with_owner(pose, validate=validate, owner=owner)


def _pose_inverse_with_owner(
    pose: "Pose",
    *,
    validate: bool,
    owner: str,
) -> "Pose":
    association = SpatialAssociationPlan(associated_graph(pose))
    pose._enforce_invariants(owner=owner)
    require_parent_basis_for_inverse(pose, owner=owner)
    source_rep = get_pose_rep(source := analysis_object_dataset(pose), owner=owner)
    position, rotation = _canonical_components(pose, owner=owner)
    position_ds = analysis_object_dataset(position)
    rotation_ds = analysis_object_dataset(rotation)
    pos_var, pos_dim = resolve_single_numeric_var_single_core_dim(position_ds, owner=owner, what="Pose translation")
    quat_var, quat_dim = resolve_single_numeric_var_single_core_dim(rotation_ds, owner=owner, what="Pose rotation")
    out_t = apply_pose_inverse_translation_kernel(
        position_ds[pos_var],
        rotation_ds[quat_var],
        pos_dim=pos_dim,
        quat_dim=quat_dim,
        owner=owner,
    )
    out_rot = _inverse_rotation(rotation, owner=owner)
    declared, seq_dim, batch_dims, _ = read_roles(position_ds)
    if not declared:
        raise ValueError(f"{owner}: Pose inverse requires declared roles.")
    param = read_param_coord_name(position_ds)
    size = read_sequence_size_coord_name(position_ds)
    out_pos_ds = pose_context.build_position_dataset(out_t, var_name=pos_var, core_dim=pos_dim, sequence_dim=seq_dim, batch_dims=batch_dims, param_coord=param, sequence_size_coord=size, owner=owner)
    out_pos_ds, out_rot_ds = frame_inverse_component_datasets(
        source,
        (out_pos_ds, analysis_object_dataset(out_rot)),
        owner=owner,
    )
    components = _pose_cls().from_components(Rotation._from_unvalidated(out_rot_ds), Position._from_unvalidated(out_pos_ds), validate=False)
    return _wrap_pose_output(
        _pose_to_rep_dataset(components, target_rep=source_rep, owner=owner),
        validate=validate,
        owner=owner,
        association=association,
    )
__all__ = [
    "_pose_compose_with_owner",
    "_pose_inverse_with_owner",
    "pose_as_components",
    "pose_as_matrix",
    "pose_compose",
    "pose_inverse",
    "pose_to_rep",
]
