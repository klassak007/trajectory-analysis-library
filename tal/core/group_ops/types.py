from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject

GroupingNaKeyPolicy = Literal["error", "drop", "group"]
ResolvedGroupingKeyKind = Literal["coord", "data_var", "external", "bin"]


@dataclass(frozen=True)
class GroupingBinSpec:
    """Explicit binning key specification for grouping foundation resolution.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    source: str | xr.DataArray
    bins: Sequence[float] | np.ndarray | xr.DataArray
    labels: Sequence[object] | None = None
    right: bool = True
    include_lowest: bool = False


GroupingSingleKey: TypeAlias = str | xr.DataArray | GroupingBinSpec
GroupingKeyInput: TypeAlias = GroupingSingleKey | tuple[GroupingSingleKey, ...] | list[GroupingSingleKey]


@dataclass(frozen=True)
class GroupingFoundationOptions:
    """Policy options for grouping foundation context resolution.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    na_key_policy: GroupingNaKeyPolicy = "error"
    na_group_label: object | None = None


@dataclass(frozen=True)
class ResolvedGroupingKey:
    """Normalized grouping key payload produced by grouping foundation owners.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    index: int
    kind: ResolvedGroupingKeyKind
    name: str
    data: xr.DataArray
    domain_order: tuple[object, ...] | None = None


@dataclass(frozen=True)
class GroupingFoundationContext:
    """Resolved grouping foundation context for grouped surface/reducer slices.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    ao: AnalysisObject
    ds: xr.Dataset
    sequence_dim: str
    batch_dims: tuple[str, ...]
    core_dims: tuple[str, ...]
    param_coord: str | None
    sequence_size_coord: str | None
    keys: tuple[ResolvedGroupingKey, ...]
    na_key_policy: GroupingNaKeyPolicy
    na_exclusion_mask: xr.DataArray | None
    na_group_label: object | None


__all__ = [
    "GroupingBinSpec",
    "GroupingFoundationContext",
    "GroupingFoundationOptions",
    "GroupingKeyInput",
    "GroupingNaKeyPolicy",
    "GroupingSingleKey",
    "ResolvedGroupingKey",
    "ResolvedGroupingKeyKind",
]
