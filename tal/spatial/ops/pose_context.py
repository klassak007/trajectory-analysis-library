from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.orchestration.context import resolve_semantic_topology_from_dataset
from tal.core.orchestration.topology import TopologyOperand, TopologyPolicy
from tal.core.orchestration.runtime_checks import resolve_single_numeric_var_single_core_dim
from tal.core.schema_read import read_param_coord_name

from ..kinematics.paired_components import resolve_paired_optional_coord_names
from ..metadata import set_position_rep

if TYPE_CHECKING:
    from ..position import Position
    from ..rotation import Rotation


@dataclass(frozen=True)
class PoseComposeTopologySpecs:
    left_var: str
    left_dim: str
    right_var: str
    right_dim: str
    right_quat_var: str
    right_quat_dim: str


def resolve_series_optional_coord_names(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    owner: str,
    allow_one_sided_inherit: bool,
) -> tuple[str | None, str | None]:
    return resolve_paired_optional_coord_names(
        left_ds,
        right_ds,
        owner=owner,
        allow_one_sided_inherit=allow_one_sided_inherit,
    )


def build_position_dataset(
    translation: xr.DataArray,
    *,
    var_name: str,
    core_dim: str,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    param_coord: str | None,
    sequence_size_coord: str | None,
    owner: str,
) -> xr.Dataset:
    param_name = param_coord if sequence_dim is not None else None
    size_name = sequence_size_coord if sequence_dim is not None else None
    kwargs: dict[str, object] = {
        "batch_dims": batch_dims,
        "core_dims": (core_dim,),
        "param_coord": param_name,
        "sequence_size_coord": size_name,
        "validate": False,
    }
    if sequence_dim is not None:
        kwargs["sequence_dim"] = sequence_dim
    ao = AnalysisObject.from_data(
        translation.to_dataset(name=var_name),
        **kwargs,
    )
    return set_position_rep(ao.unsafe_data, rep="cart", validate=False, owner=owner)


def topology_operand(
    ds: xr.Dataset,
    *,
    var_name: str,
    core_dims: tuple[str, ...],
    index: int,
    owner: str,
    what: str,
    policy: TopologyPolicy,
) -> TopologyOperand:
    return TopologyOperand(
        index=index,
        data=ds[var_name],
        semantic=resolve_semantic_topology_from_dataset(
            ds,
            var_name=var_name,
            core_dims=core_dims,
            owner=owner,
            what=what,
            allow_missing_sequence_dim=policy.mode == "semantic_broadcast",
            allow_missing_batch_dims=policy.mode == "semantic_broadcast",
        ),
        param_coord=read_param_coord_name(ds),
    )


def resolve_pose_compose_specs(
    left_pos: Position,
    right_pos: Position,
    right_rot: Rotation,
    *,
    owner: str,
) -> PoseComposeTopologySpecs:
    left_var, left_dim = resolve_single_numeric_var_single_core_dim(
        left_pos.unsafe_data,
        owner=owner,
        what="left pose translation",
    )
    right_var, right_dim = resolve_single_numeric_var_single_core_dim(
        right_pos.unsafe_data,
        owner=owner,
        what="right pose translation",
    )
    right_quat_var, right_quat_dim = resolve_single_numeric_var_single_core_dim(
        right_rot.unsafe_data,
        owner=owner,
        what="right pose rotation",
    )
    return PoseComposeTopologySpecs(
        left_var=left_var,
        left_dim=left_dim,
        right_var=right_var,
        right_dim=right_dim,
        right_quat_var=right_quat_var,
        right_quat_dim=right_quat_dim,
    )


def pose_compose_topology_operands(
    specs: PoseComposeTopologySpecs,
    left_pos: Position,
    right_pos: Position,
    right_rot: Rotation,
    *,
    owner: str,
    policy: TopologyPolicy,
) -> tuple[TopologyOperand, TopologyOperand, TopologyOperand]:
    return (
        topology_operand(
            left_pos.unsafe_data,
            var_name=specs.left_var,
            core_dims=(specs.left_dim,),
            index=0,
            owner=owner,
            what="left pose translation",
            policy=policy,
        ),
        topology_operand(
            right_pos.unsafe_data,
            var_name=specs.right_var,
            core_dims=(specs.right_dim,),
            index=1,
            owner=owner,
            what="right pose translation",
            policy=policy,
        ),
        topology_operand(
            right_rot.unsafe_data,
            var_name=specs.right_quat_var,
            core_dims=(specs.right_quat_dim,),
            index=2,
            owner=owner,
            what="right pose rotation",
            policy=policy,
        ),
    )


__all__ = [
    "build_position_dataset",
    "PoseComposeTopologySpecs",
    "pose_compose_topology_operands",
    "resolve_series_optional_coord_names",
    "resolve_pose_compose_specs",
    "topology_operand",
]
