from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.core.param_ops import ParamAccessor

from .options import coerce_pose_temporal_options, coerce_rotation_temporal_options

if TYPE_CHECKING:
    from tal.spatial.pose import Pose
    from tal.spatial.rotation import Rotation


class RotationParamAccessor(ParamAccessor):
    """Typed param accessor for Rotation temporal defaults.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    def at(
        self,
        query: xr.DataArray | np.ndarray | Sequence[float] | float,
        *,
        on: str | None = None,
        opts: object | None = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "Rotation":
        from ..ops.rotation_temporal_ops import rotation_param_at

        normalized = coerce_rotation_temporal_options(opts, owner="spatial.rotation.param.at")
        return rotation_param_at(
            self._ao,
            query=query,
            on=on,
            opts=normalized,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.rotation.param.at",
        )

    def resample_to(
        self,
        grid: xr.DataArray | np.ndarray | Sequence[float] | float,
        *,
        on: str | None = None,
        opts: object | None = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "Rotation":
        from ..ops.rotation_temporal_ops import rotation_param_resample_to

        normalized = coerce_rotation_temporal_options(opts, owner="spatial.rotation.param.resample_to")
        return rotation_param_resample_to(
            self._ao,
            grid=grid,
            on=on,
            opts=normalized,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.rotation.param.resample_to",
        )


class PoseParamAccessor(ParamAccessor):
    """Typed param accessor for Pose temporal defaults.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    def at(
        self,
        query: xr.DataArray | np.ndarray | Sequence[float] | float,
        *,
        on: str | None = None,
        opts: object | None = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "Pose":
        from ..ops.pose_temporal_ops import pose_param_at

        normalized = coerce_pose_temporal_options(opts, owner="spatial.pose.param.at")
        return pose_param_at(
            self._ao,
            query=query,
            on=on,
            opts=normalized,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.pose.param.at",
        )

    def resample_to(
        self,
        grid: xr.DataArray | np.ndarray | Sequence[float] | float,
        *,
        on: str | None = None,
        opts: object | None = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "Pose":
        from ..ops.pose_temporal_ops import pose_param_resample_to

        normalized = coerce_pose_temporal_options(opts, owner="spatial.pose.param.resample_to")
        return pose_param_resample_to(
            self._ao,
            grid=grid,
            on=on,
            opts=normalized,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.pose.param.resample_to",
        )


__all__ = ["PoseParamAccessor", "RotationParamAccessor"]
