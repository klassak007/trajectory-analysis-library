from __future__ import annotations

import xarray as xr

from .around import evaluate_around_windows
from .at_boundaries import evaluate_at_boundaries_condition
from .boundary import extract_event_boundaries
from .evaluate import evaluate_mask
from .intervals import extract_intervals
from .options import (
    coerce_around_options,
    coerce_condition_eval_options,
    coerce_when_options,
    coerce_event_extract_options,
    coerce_interval_extract_options,
    coerce_at_boundaries_options,
)
from .pack import pack_event_table, pack_interval_table
from .resolve import resolve_event_eval_context
from .types import (
    AroundOptions,
    Condition,
    ConditionEvalOptions,
    WhenOptions,
    EventExtractOptions,
    IntervalExtractOptions,
    AtBoundariesOptions,
)
from .when import evaluate_when_condition


class EventsAccessor:
    """Accessor for condition evaluation and event operations.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    def __init__(self, ao: "AnalysisObject") -> None:
        self._ao = ao

    def mask(
        self,
        condition: Condition,
        *,
        opts: ConditionEvalOptions | None = None,
    ) -> "xr.DataArray":
        """Evaluate a condition and return the boolean mask.

        Parameters
        ----------
        condition : Condition
            Condition/expression input used for event evaluation.
        opts : ConditionEvalOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``ConditionEvalOptions`` key fields: ``on`` (default None), ``coord_name`` (default 'time'), ``sample_dim`` (default None), ``ao_interp`` (default 'linear').

        Returns
        -------
        xr.DataArray
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
        >>> from tal.core.event_ops import Condition
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [0.0, 2.0, 3.0, 1.0])}, coords={"sample": [0, 1, 2, 3], "time": ("sample", [0.0, 1.0, 2.0, 3.0])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> condition = Condition.compare(Condition.var("value"), "gt", 1.5)
        >>> ao.events.mask(condition).values.tolist()
        [False, True, True, False]
        """
        options = coerce_condition_eval_options(opts, owner="events.mask")
        context = resolve_event_eval_context(self._ao, opts=options, owner="events.mask")
        return evaluate_mask(condition, context=context, owner="events.mask")

    def events(
        self,
        condition: Condition,
        *,
        opts: EventExtractOptions | None = None,
    ) -> "xr.Dataset":
        """Extract event boundary table from a condition.

        Parameters
        ----------
        condition : Condition
            Condition/expression input used for event evaluation.
        opts : EventExtractOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``EventExtractOptions`` key fields: ``eval`` (default '<factory>'), ``include_initial`` (default False), ``truth_eval`` (default 'exact'), ``max_events`` (default None).

        Returns
        -------
        xr.Dataset
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
        >>> from tal.core.event_ops import Condition
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [0.0, 2.0, 3.0, 1.0])}, coords={"sample": [0, 1, 2, 3], "time": ("sample", [0.0, 1.0, 2.0, 3.0])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> table = ao.events.events(Condition.compare(Condition.var("value"), "gt", 1.5))
        >>> table["edge_code"].values.tolist()
        [1, 2]
        """
        options = coerce_event_extract_options(opts, owner="events.events")
        if options.truth_eval != "exact":
            raise NotImplementedError(
                f"events.events: truth_eval={options.truth_eval!r} is not supported."
            )
        context = resolve_event_eval_context(self._ao, opts=options.eval, owner="events.events")
        effective = evaluate_mask(condition, context=context, owner="events.events")
        payload = extract_event_boundaries(
            condition,
            effective_mask=effective,
            context=context,
            opts=options,
            owner="events.events",
        )
        return pack_event_table(payload, context=context, owner="events.events")

    def intervals(
        self,
        condition: Condition,
        *,
        opts: IntervalExtractOptions | None = None,
    ) -> "xr.Dataset":
        """Extract interval table from a condition.

        Parameters
        ----------
        condition : Condition
            Condition/expression input used for event evaluation.
        opts : IntervalExtractOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``IntervalExtractOptions`` key fields: ``eval`` (default '<factory>'), ``include_initial`` (default False), ``truth_eval`` (default 'exact'), ``max_segments`` (default None).

        Returns
        -------
        xr.Dataset
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
        >>> from tal.core.event_ops import Condition
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [0.0, 2.0, 3.0, 1.0])}, coords={"sample": [0, 1, 2, 3], "time": ("sample", [0.0, 1.0, 2.0, 3.0])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> intervals = ao.events.intervals(Condition.compare(Condition.var("value"), "gt", 1.5))
        >>> intervals.sizes["segment"]
        1
        """
        options = coerce_interval_extract_options(opts, owner="events.intervals")
        if options.truth_eval != "exact":
            raise NotImplementedError(
                f"events.intervals: truth_eval={options.truth_eval!r} is not supported."
            )
        context = resolve_event_eval_context(self._ao, opts=options.eval, owner="events.intervals")
        effective = evaluate_mask(condition, context=context, owner="events.intervals")
        payload = extract_intervals(
            condition,
            effective_mask=effective,
            context=context,
            opts=options,
            owner="events.intervals",
        )
        return pack_interval_table(payload, context=context, owner="events.intervals")

    def at_boundaries(
        self,
        condition: Condition,
        *,
        opts: AtBoundariesOptions | None = None,
    ) -> "AnalysisObject":
        """Sample AO values at event boundaries.

        Parameters
        ----------
        condition : Condition
            Condition/expression input used for event evaluation.
        opts : AtBoundariesOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``AtBoundariesOptions`` key fields: ``eval`` (default '<factory>'), ``edges`` (default 'all'), ``mode`` (default 'all'), ``max_events`` (default None).

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
        >>> from tal.core.event_ops import AtBoundariesOptions, Condition
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [0.0, 2.0, 3.0, 1.0])}, coords={"sample": [0, 1, 2, 3], "time": ("sample", [0.0, 1.0, 2.0, 3.0])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> out = ao.events.at_boundaries(Condition.compare(Condition.var("value"), "gt", 1.5), opts=AtBoundariesOptions(edges="enter"))
        >>> out.as_dataset()["value"].values.tolist()
        [2.0]
        """
        options = coerce_at_boundaries_options(opts, owner="events.at_boundaries")
        return evaluate_at_boundaries_condition(
            self._ao,
            condition,
            opts=options,
            validate=True,
            owner="events.at_boundaries",
        )

    def when(
        self,
        condition: Condition,
        *,
        opts: WhenOptions | None = None,
    ) -> "AnalysisObject":
        """Extract condition-selected windows from an AO.

        Parameters
        ----------
        condition : Condition
            Condition/expression input used for event evaluation.
        opts : WhenOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``WhenOptions`` key fields: ``eval`` (default '<factory>'), ``layout`` (default 'mask'), ``inside`` (default True), ``on_empty`` (default 'empty').

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
        >>> from tal.core.event_ops import Condition, WhenOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [0.0, 2.0, 3.0, 1.0])}, coords={"sample": [0, 1, 2, 3], "time": ("sample", [0.0, 1.0, 2.0, 3.0])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> out = ao.events.when(Condition.compare(Condition.var("value"), "gt", 1.5), opts=WhenOptions(layout="mask"))
        >>> out.as_dataset()["value"].isnull().values.tolist()
        [True, False, False, True]
        """
        options = coerce_when_options(opts, owner="events.when")
        return evaluate_when_condition(
            self._ao,
            condition,
            opts=options,
            validate=True,
            owner="events.when",
        )

    def around(
        self,
        events_or_condition: Condition | xr.DataArray | xr.Dataset,
        *,
        opts: AroundOptions | None = None,
    ) -> "AnalysisObject":
        """Extract around-event windows from explicit events or a condition.

        Parameters
        ----------
        events_or_condition : Condition | xr.DataArray | xr.Dataset
            Either a ``Condition`` or precomputed event payload used to define around-windows.
        opts : AroundOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``AroundOptions`` key fields: ``eval`` (default '<factory>'), ``edge`` (default 'enter'), ``pre`` (default 0.5), ``post`` (default 0.5).

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
        >>> from tal.core.event_ops import AroundOptions, Condition
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [0.0, 2.0, 3.0, 1.0])}, coords={"sample": [0, 1, 2, 3], "time": ("sample", [0.0, 1.0, 2.0, 3.0])}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> out = ao.events.around(Condition.compare(Condition.var("value"), "gt", 1.5), opts=AroundOptions(pre=0.0, post=0.0, dt=1.0))
        >>> out.as_dataset().sizes["event"]
        1
        """
        options = coerce_around_options(opts, owner="events.around")
        return evaluate_around_windows(
            self._ao,
            events_or_condition,
            opts=options,
            validate=True,
            owner="events.around",
        )


__all__ = ["EventsAccessor"]
