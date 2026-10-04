from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.core.param_engine.prepared import PreparedParamEvaluation

from ..temporal.options import RotationTemporalOptions

if TYPE_CHECKING:
    from ..rotation import Rotation


@dataclass(frozen=True)
class RotationTemporalRequest:
    rotation: Rotation
    query: xr.DataArray | np.ndarray | Sequence[float] | float
    on: str | None
    opts: RotationTemporalOptions
    validate: bool
    sequence_dim: str | None
    batch_dims: Sequence[str] | None
    sequence_size_coord: str | None
    owner: str
    prepared: PreparedParamEvaluation | None = None
