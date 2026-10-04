from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from ..param_ops.types import ParamSyncOptions

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


@dataclass(frozen=True)
class BatchConcatOptions:
    """Options for batch concatenation.

    Notes
    -----
    ``batch_dim`` names the output batch axis when inputs do not already carry
    the same batch topology. Sequence labels are joined according to
    ``sequence_join``.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, BatchConcatOptions, concat_batch
    >>> left = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> right = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> out = concat_batch([left, right], opts=BatchConcatOptions(batch_dim="run"))
    >>> out.as_dataset().sizes["run"]
    2
    """

    batch_dim: str = "batch"
    batch_labels: tuple[object, ...] | None = None
    sequence_join: Literal["inner", "outer", "exact"] = "outer"
    fill_value: float | int | None = np.nan


@dataclass(frozen=True)
class SequenceConcatOptions:
    """Options for sequence concatenation.

    Notes
    -----
    ``overlap="error"`` rejects duplicate sequence labels. Use ``"sort"``
    only when overlapping labels should be accepted and sorted by coordinate.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, SequenceConcatOptions, concat_sequence
    >>> first = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> second = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [1]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> concat_sequence([first, second], opts=SequenceConcatOptions(overlap="error")).as_dataset().sizes["sample"]
    2
    """

    batch_join: Literal["inner", "outer", "exact"] = "outer"
    overlap: Literal["error", "sort"] = "error"
    fill_value: float | int | None = np.nan


@dataclass(frozen=True)
class ParamPrealignOptions:
    """Options for param pre-alignment before merge.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    grid: object | None = None
    sync_opts: ParamSyncOptions = field(default_factory=ParamSyncOptions)


@dataclass(frozen=True)
class MergeOptions:
    """Options for schema-aware AO merge.

    Notes
    -----
    Merge policy is xarray-label aware. ``sequence_join="exact"`` keeps the
    default conservative behavior for trajectory samples.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, MergeOptions, merge
    >>> left = AnalysisObject.from_data(xr.Dataset({"x": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> right = AnalysisObject.from_data(xr.Dataset({"y": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> sorted(merge([left, right], opts=MergeOptions()).as_dataset().data_vars)
    ['x', 'y']
    """

    batch_join: Literal["inner", "outer", "exact"] = "inner"
    sequence_join: Literal["inner", "outer", "exact"] = "exact"
    param_prealign: ParamPrealignOptions | None = None
    compat: Literal["identical", "equals", "broadcast_equals", "no_conflicts", "override"] = "no_conflicts"
    combine_attrs: Literal["drop", "identical", "no_conflicts", "drop_conflicts", "override"] = "override"
    outer_fill_value: float | int | None | dict[str, float | int | None] = np.nan


@dataclass(frozen=True)
class AlignOptions:
    """Options for label/topology alignment across AO inputs.

    Notes
    -----
    Alignment follows dimension names and coordinate labels. Outer joins fill
    absent labels with ``fill_value`` while preserving AO schema roles.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AlignOptions, AnalysisObject, align_pair
    >>> left = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> right = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [3.0, 4.0])}, coords={"sample": [1, 2]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> a, b = align_pair(left, right, opts=AlignOptions(sequence_join="outer"))
    >>> (a.as_dataset().sizes["sample"], b.as_dataset().sizes["sample"])
    (3, 3)
    """

    batch_join: Literal["inner", "outer", "exact"] = "inner"
    sequence_join: Literal["inner", "outer", "exact"] = "exact"
    fill_value: float | int | None = np.nan
    pad_invalid_outer: bool = True


@dataclass(frozen=True)
class CoreConcatOptions:
    """Options for concatenation over existing core dimensions.

    Notes
    -----
    ``core_dim`` must already be a core dimension on every operand.
    ``core_labels`` may override the concatenated coordinate labels.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, CoreConcatOptions, concat_core
    >>> x = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[1.0]])}, coords={"sample": [0], "axis": ["x"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
    >>> y = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[2.0]])}, coords={"sample": [0], "axis": ["y"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
    >>> concat_core([x, y], opts=CoreConcatOptions(core_dim="axis")).as_dataset().sizes["axis"]
    2
    """

    core_dim: str
    core_labels: tuple[object, ...] | None = None
    output_var: str | None = None


@dataclass(frozen=True)
class CoreDecomposeOptions:
    """Options for decomposition of selected core dimensions.

    Notes
    -----
    Decomposition returns one AO per selected core-coordinate key. Use
    ``key_mode="label"`` when output keys should contain coordinate labels.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, CoreDecomposeOptions, decompose_core
    >>> ao = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
    >>> sorted(decompose_core(ao, opts=CoreDecomposeOptions(core_dims=("axis",), key_mode="label")))
    [('x',), ('y',)]
    """

    core_dims: tuple[str, ...]
    key_mode: Literal["index", "label"] = "index"
    output_var: str | None = None


@dataclass(frozen=True)
class CoreOverlayOptions:
    """Options for label-based overlay on a core dimension.

    Notes
    -----
    Overlay patches selected core labels in a base AO. ``on_overlap="replace"``
    allows patches to intentionally replace existing labels.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, CoreOverlayOptions, overlay_core
    >>> base = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
    >>> patch = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[9.0]])}, coords={"sample": [0], "axis": ["y"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
    >>> overlay_core(base, patch, opts=CoreOverlayOptions(core_dim="axis", on_overlap="replace")).as_dataset()["v"].sel(axis="y").item()
    9.0
    """

    core_dim: str
    on_overlap: Literal["error", "replace"] = "error"
    output_var: str | None = None


@dataclass(frozen=True)
class CombineResolveOptions:
    """Internal options for AO combine-context resolution.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    require_sequence: bool = False
    owner: str = "combine ops"


@dataclass(frozen=True)
class CombineContext:
    """Resolved immutable combine context.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    ao: AnalysisObject
    ds: xr.Dataset
    roles_declared: bool
    sequence_dim: str | None
    batch_dims: tuple[str, ...]
    core_dims: tuple[str, ...]
    param_coord: str | None
    sequence_size_coord: str | None


@dataclass(frozen=True)
class BatchStackPlan:
    """Plan for flattening/restoring batch topology during combine.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    enabled: bool
    flat_dim: str
    batch_dims: tuple[str, ...]
