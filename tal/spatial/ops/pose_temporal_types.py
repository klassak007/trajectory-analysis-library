"""Immutable request declarations for Pose temporal orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from tal.core.param_engine.prepared import PreparedParamEvaluation

from ..temporal.options import PoseTemporalOptions

if TYPE_CHECKING:
    from ..pose import Pose


@dataclass(frozen=True)
class PreparedPoseEvaluation:
    """Prepared linear and rotation maps for one Pose request."""

    position: PreparedParamEvaluation
    rotation: PreparedParamEvaluation


@dataclass(frozen=True)
class PoseTemporalRequest:
    pose: Pose
    query: xr.DataArray | np.ndarray | Sequence[float] | float
    on: str | None
    opts: PoseTemporalOptions
    validate: bool
    sequence_dim: str | None
    batch_dims: Sequence[str] | None
    sequence_size_coord: str | None
    owner: str
    mode: Literal["at", "resample_to"]
    prepared: PreparedPoseEvaluation | None = None


__all__: list[str] = []
