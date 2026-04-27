from __future__ import annotations

from typing import TYPE_CHECKING

from ..reducer_ops.types import DimLike, WeightInput
from .grouped_types import GroupMaterializeOptions
from .reducer_dispatch import grouped_reduce

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject
    from .accessor import GroupedView


def _dispatch(
    self: "GroupedView",
    *,
    op: str,
    dim: DimLike = None,
    skipna: bool = True,
    ddof: int = 0,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    return grouped_reduce(
        self,
        op=op,
        dim=dim,
        skipna=bool(skipna),
        ddof=int(ddof),
        weights=weights,
        opts=opts,
        validate=validate,
        owner=f"group.{op}",
    )


def _mean(
    self: "GroupedView",
    dim: DimLike = None,
    *,
    skipna: bool = True,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Grouped mean reducer.

    Parameters
    ----------
    dim : DimLike, optional
        Dimension(s) reduced by grouped reducer operations.
    skipna : bool, optional
        Whether to ignore missing values in reducer kernels.
    weights : WeightInput, optional
        Optional weight input for weighted grouped reduction.
    opts : GroupMaterializeOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``GroupMaterializeOptions`` key fields: ``layout`` (default 'padded'), ``group_dim`` (default None), ``member_dim`` (default None), ``sequence_index_coord`` (default None).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> ao.group.groupby("kind").mean(dim="sample").unsafe_data["value"].sel(group_key="sim").item()
    1.5
    """
    return _dispatch(self, op="mean", dim=dim, skipna=skipna, weights=weights, opts=opts, validate=validate)


def _sum(
    self: "GroupedView",
    dim: DimLike = None,
    *,
    skipna: bool = True,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Grouped sum reducer.

    Parameters
    ----------
    dim : DimLike, optional
        Dimension(s) reduced by grouped reducer operations.
    skipna : bool, optional
        Whether to ignore missing values in reducer kernels.
    weights : WeightInput, optional
        Optional weight input for weighted grouped reduction.
    opts : GroupMaterializeOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``GroupMaterializeOptions`` key fields: ``layout`` (default 'padded'), ``group_dim`` (default None), ``member_dim`` (default None), ``sequence_index_coord`` (default None).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> ao.group.groupby("kind").sum(dim="sample").unsafe_data["value"].sel(group_key="robot").item()
    7.0
    """
    return _dispatch(self, op="sum", dim=dim, skipna=skipna, weights=weights, opts=opts, validate=validate)


def _std(
    self: "GroupedView",
    dim: DimLike = None,
    *,
    skipna: bool = True,
    ddof: int = 0,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Grouped standard-deviation reducer.

    Parameters
    ----------
    dim : DimLike, optional
        Dimension(s) reduced by grouped reducer operations.
    skipna : bool, optional
        Whether to ignore missing values in reducer kernels.
    ddof : int, optional
        Delta degrees of freedom for variance/standard-deviation reducers.
    weights : WeightInput, optional
        Optional weight input for weighted grouped reduction.
    opts : GroupMaterializeOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``GroupMaterializeOptions`` key fields: ``layout`` (default 'padded'), ``group_dim`` (default None), ``member_dim`` (default None), ``sequence_index_coord`` (default None).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> ao.group.groupby("kind").std(dim="sample").unsafe_data["value"].sel(group_key="sim").item()
    0.5
    """
    return _dispatch(
        self,
        op="std",
        dim=dim,
        skipna=skipna,
        ddof=ddof,
        weights=weights,
        opts=opts,
        validate=validate,
    )


def _var(
    self: "GroupedView",
    dim: DimLike = None,
    *,
    skipna: bool = True,
    ddof: int = 0,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Grouped variance reducer.

    Parameters
    ----------
    dim : DimLike, optional
        Dimension(s) reduced by grouped reducer operations.
    skipna : bool, optional
        Whether to ignore missing values in reducer kernels.
    ddof : int, optional
        Delta degrees of freedom for variance/standard-deviation reducers.
    weights : WeightInput, optional
        Optional weight input for weighted grouped reduction.
    opts : GroupMaterializeOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``GroupMaterializeOptions`` key fields: ``layout`` (default 'padded'), ``group_dim`` (default None), ``member_dim`` (default None), ``sequence_index_coord`` (default None).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> ao.group.groupby("kind").var(dim="sample").unsafe_data["value"].sel(group_key="sim").item()
    0.25
    """
    return _dispatch(
        self,
        op="var",
        dim=dim,
        skipna=skipna,
        ddof=ddof,
        weights=weights,
        opts=opts,
        validate=validate,
    )


def _median(
    self: "GroupedView",
    dim: DimLike = None,
    *,
    skipna: bool = True,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Grouped median reducer.

    Parameters
    ----------
    dim : DimLike, optional
        Dimension(s) reduced by grouped reducer operations.
    skipna : bool, optional
        Whether to ignore missing values in reducer kernels.
    weights : WeightInput, optional
        Optional weight input for weighted grouped reduction.
    opts : GroupMaterializeOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``GroupMaterializeOptions`` key fields: ``layout`` (default 'padded'), ``group_dim`` (default None), ``member_dim`` (default None), ``sequence_index_coord`` (default None).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> ao.group.groupby("kind").median(dim="sample").unsafe_data["value"].sel(group_key="robot").item()
    3.5
    """
    return _dispatch(self, op="median", dim=dim, skipna=skipna, weights=weights, opts=opts, validate=validate)


def _min(
    self: "GroupedView",
    dim: DimLike = None,
    *,
    skipna: bool = True,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Grouped minimum reducer.

    Parameters
    ----------
    dim : DimLike, optional
        Dimension(s) reduced by grouped reducer operations.
    skipna : bool, optional
        Whether to ignore missing values in reducer kernels.
    weights : WeightInput, optional
        Optional weight input for weighted grouped reduction.
    opts : GroupMaterializeOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``GroupMaterializeOptions`` key fields: ``layout`` (default 'padded'), ``group_dim`` (default None), ``member_dim`` (default None), ``sequence_index_coord`` (default None).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> ao.group.groupby("kind").min(dim="sample").unsafe_data["value"].sel(group_key="sim").item()
    1.0
    """
    return _dispatch(self, op="min", dim=dim, skipna=skipna, weights=weights, opts=opts, validate=validate)


def _max(
    self: "GroupedView",
    dim: DimLike = None,
    *,
    skipna: bool = True,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Grouped maximum reducer.

    Parameters
    ----------
    dim : DimLike, optional
        Dimension(s) reduced by grouped reducer operations.
    skipna : bool, optional
        Whether to ignore missing values in reducer kernels.
    weights : WeightInput, optional
        Optional weight input for weighted grouped reduction.
    opts : GroupMaterializeOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``GroupMaterializeOptions`` key fields: ``layout`` (default 'padded'), ``group_dim`` (default None), ``member_dim`` (default None), ``sequence_index_coord`` (default None).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> ao.group.groupby("kind").max(dim="sample").unsafe_data["value"].sel(group_key="robot").item()
    4.0
    """
    return _dispatch(self, op="max", dim=dim, skipna=skipna, weights=weights, opts=opts, validate=validate)


def _count(
    self: "GroupedView",
    dim: DimLike = None,
    *,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Grouped count reducer.

    Parameters
    ----------
    dim : DimLike, optional
        Dimension(s) reduced by grouped reducer operations.
    weights : WeightInput, optional
        Optional weight input for weighted grouped reduction.
    opts : GroupMaterializeOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``GroupMaterializeOptions`` key fields: ``layout`` (default 'padded'), ``group_dim`` (default None), ``member_dim`` (default None), ``sequence_index_coord`` (default None).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> ao.group.groupby("kind").count(dim="sample").unsafe_data["value"].sel(group_key="sim").item()
    2
    """
    return _dispatch(self, op="count", dim=dim, weights=weights, opts=opts, validate=validate)


def _any(
    self: "GroupedView",
    dim: DimLike = None,
    *,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Grouped logical-any reducer.

    Parameters
    ----------
    dim : DimLike, optional
        Dimension(s) reduced by grouped reducer operations.
    weights : WeightInput, optional
        Optional weight input for weighted grouped reduction.
    opts : GroupMaterializeOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``GroupMaterializeOptions`` key fields: ``layout`` (default 'padded'), ``group_dim`` (default None), ``member_dim`` (default None), ``sequence_index_coord`` (default None).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"flag": (("run", "sample"), [[False, False], [False, True]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> bool(ao.group.groupby("kind").any(dim="sample").unsafe_data["flag"].sel(group_key="robot").item())
    True
    """
    return _dispatch(self, op="any", dim=dim, weights=weights, opts=opts, validate=validate)


def _all(
    self: "GroupedView",
    dim: DimLike = None,
    *,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Grouped logical-all reducer.

    Parameters
    ----------
    dim : DimLike, optional
        Dimension(s) reduced by grouped reducer operations.
    weights : WeightInput, optional
        Optional weight input for weighted grouped reduction.
    opts : GroupMaterializeOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``GroupMaterializeOptions`` key fields: ``layout`` (default 'padded'), ``group_dim`` (default None), ``member_dim`` (default None), ``sequence_index_coord`` (default None).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"flag": (("run", "sample"), [[True, False], [True, True]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> bool(ao.group.groupby("kind").all(dim="sample").unsafe_data["flag"].sel(group_key="robot").item())
    True
    """
    return _dispatch(self, op="all", dim=dim, weights=weights, opts=opts, validate=validate)


def install_grouped_view_reducers(cls: type) -> None:
    cls.mean = _mean
    cls.sum = _sum
    cls.std = _std
    cls.var = _var
    cls.median = _median
    cls.min = _min
    cls.max = _max
    cls.count = _count
    cls.any = _any
    cls.all = _all


__all__ = ["install_grouped_view_reducers"]
