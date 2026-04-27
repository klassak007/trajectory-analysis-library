from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .foundation import resolve_grouping_foundation_context
from .grouped_options import coerce_group_materialize_options, coerce_groupby_options
from .grouped_types import GroupByOptions, GroupMaterializeOptions, GroupedRuntimePlan
from .materialize import materialize_grouped_view
from .reducer_surface import install_grouped_view_reducers
from .runtime_plan import resolve_grouped_runtime_plan
from .types import GroupingBinSpec, GroupingFoundationContext, GroupingKeyInput


class GroupedView:
    """Immutable grouped wrapper that exposes grouped layout materialization only.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    def __init__(
        self,
        foundation: GroupingFoundationContext,
        *,
        opts: GroupByOptions,
    ) -> None:
        self._foundation = foundation
        self._opts = opts
        self._runtime_plan: GroupedRuntimePlan | None = None

    def _resolve_plan(self, *, owner: str) -> GroupedRuntimePlan:
        if self._runtime_plan is None:
            self._runtime_plan = resolve_grouped_runtime_plan(
                self._foundation,
                opts=self._opts,
                owner=owner,
            )
        return self._runtime_plan

    def materialize(
        self,
        *,
        opts: GroupMaterializeOptions | None = None,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Materialize grouped runtime plan into an AO layout.

        Parameters
        ----------
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
        >>> from tal.core import AnalysisObject, GroupMaterializeOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
        ...     sequence_dim="sample",
        ...     batch_dims=("run",),
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> out = ao.group.groupby("kind").materialize(opts=GroupMaterializeOptions(layout="padded"))
        >>> out.unsafe_data.sizes["group_key"]
        2
        """
        options = coerce_group_materialize_options(opts, owner="group.materialize")
        return materialize_grouped_view(
            self._resolve_plan(owner="group.materialize"),
            opts=options,
            validate=validate,
            owner="group.materialize",
        )

    def padded(
        self,
        *,
        opts: GroupMaterializeOptions | None = None,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Materialize grouped payload into padded layout.

        Parameters
        ----------
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
        >>> ao.group.groupby("kind").padded().unsafe_data.sizes["sample"]
        2
        """
        options = coerce_group_materialize_options(opts, owner="group.padded")
        materialize_opts = GroupMaterializeOptions(
            layout="padded",
            group_dim=options.group_dim,
            member_dim=options.member_dim,
            sequence_index_coord=options.sequence_index_coord,
            include_empty_groups=options.include_empty_groups,
        )
        return materialize_grouped_view(
            self._resolve_plan(owner="group.padded"),
            opts=materialize_opts,
            validate=validate,
            owner="group.padded",
        )

    def stacked(
        self,
        *,
        opts: GroupMaterializeOptions | None = None,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Materialize grouped payload into stacked row layout.

        Parameters
        ----------
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
        >>> ao.group.groupby("kind").stacked().unsafe_data.sizes["group_member"]
        4
        """
        options = coerce_group_materialize_options(opts, owner="group.stacked")
        materialize_opts = GroupMaterializeOptions(
            layout="stacked",
            group_dim=options.group_dim,
            member_dim=options.member_dim,
            sequence_index_coord=options.sequence_index_coord,
            include_empty_groups=options.include_empty_groups,
        )
        return materialize_grouped_view(
            self._resolve_plan(owner="group.stacked"),
            opts=materialize_opts,
            validate=validate,
            owner="group.stacked",
        )


class GroupAccessor:
    """Grouping accessor rooted at ``ao.group``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    def __init__(self, ao: "AnalysisObject") -> None:
        self._ao = ao

    def groupby(
        self,
        key: GroupingKeyInput,
        *,
        opts: GroupByOptions | None = None,
    ) -> GroupedView:
        """Construct a grouped view using grouping-key semantics.

        Parameters
        ----------
        key : GroupingKeyInput
            Grouping key used to derive grouped runtime partitions.
        opts : GroupByOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``GroupByOptions`` key fields: ``preserve_batch`` (default False), ``foundation_opts`` (default None), ``group_dim`` (default 'group_key'), ``member_dim`` (default 'group_member').

        Returns
        -------
        GroupedView
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
        >>> from tal.core import AnalysisObject, GroupByOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])}),
        ...     sequence_dim="sample",
        ...     batch_dims=("run",),
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> grouped = ao.group.groupby("kind", opts=GroupByOptions(preserve_batch=False))
        >>> grouped.mean(dim="sample").unsafe_data.sizes["group_key"]
        2
        """
        options = coerce_groupby_options(opts, owner="group.groupby")
        foundation = resolve_grouping_foundation_context(
            self._ao,
            key,
            opts=options.foundation_opts,
            owner="group.groupby",
        )
        return GroupedView(foundation, opts=options)

    def groupby_bins(
        self,
        source: str | xr.DataArray,
        bins: Sequence[float] | np.ndarray | xr.DataArray,
        *,
        labels: Sequence[object] | None = None,
        right: bool = True,
        include_lowest: bool = False,
        opts: GroupByOptions | None = None,
    ) -> GroupedView:
        """Construct grouped view by explicit binning specification.

        Parameters
        ----------
        source : str | xr.DataArray
            Input source value consumed by this operation.
        bins : Sequence[float] | np.ndarray | xr.DataArray
            Binning edges used to construct grouped buckets.
        labels : Sequence[object] | None, optional
            Optional labels assigned to grouped bins.
        right : bool, optional
            If ``True``, bins are right-closed; otherwise left-closed.
        include_lowest : bool, optional
            Include values equal to the first bin edge in the first bin.
        opts : GroupByOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``GroupByOptions`` key fields: ``preserve_batch`` (default False), ``foundation_opts`` (default None), ``group_dim`` (default 'group_key'), ``member_dim`` (default 'group_member').

        Returns
        -------
        GroupedView
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
        ...     xr.Dataset({"value": ("sample", [1.0, 2.0, 3.0])}, coords={"sample": [0, 1, 2], "time": ("sample", [0.0, 0.5, 1.5])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> out = ao.group.groupby_bins("time", bins=[0.0, 1.0, 2.0], include_lowest=True).padded()
        >>> out.unsafe_data.sizes["group_key"]
        2
        """
        spec = GroupingBinSpec(
            source=source,
            bins=bins,
            labels=labels,
            right=right,
            include_lowest=include_lowest,
        )
        return self.groupby(spec, opts=opts)


__all__ = ["GroupAccessor", "GroupedView"]

install_grouped_view_reducers(GroupedView)
