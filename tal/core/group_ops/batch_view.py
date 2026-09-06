from __future__ import annotations

from typing import cast

from .batch_runtime import resolve_batch_grouped_runtime_plan
from .grouped_types import BatchGroupedRuntimePlan, GroupByOptions
from .reducer_surface import install_grouped_view_reducers
from .types import BatchGroupingFoundationContext


class BatchGroupedView:
    """Represent batch-only grouping returned by ``ao.group.groupby(...)``.

    Notes
    -----
    This is a return-only result type. Direct construction is unsupported;
    ``ao.group.groupby(...)`` owns context and option validation.

    Unlike sequence grouping, batch grouping replaces its primary batch lane
    and therefore exposes no member-layout materialization methods.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, BatchGroupedView
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset(
    ...         {"value": ("trial", [1.0, 2.0, 3.0])},
    ...         coords={"trial": [0, 1, 2], "outcome": ("trial", ["a", "b", "a"])},
    ...     ),
    ...     batch_dims=("trial",),
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> grouped = ao.group.groupby("outcome")
    >>> isinstance(grouped, BatchGroupedView)
    True
    >>> grouped.mean().as_dataset()["value"].values.tolist()
    [2.0, 2.0]
    """

    def __init__(
        self,
        foundation: object,
        *,
        opts: GroupByOptions,
    ) -> None:
        self._foundation = cast(BatchGroupingFoundationContext, foundation)
        self._opts = opts
        self._runtime_plan: BatchGroupedRuntimePlan | None = None

    def _resolve_plan(self, *, owner: str) -> BatchGroupedRuntimePlan:
        if self._runtime_plan is None:
            self._runtime_plan = resolve_batch_grouped_runtime_plan(
                self._foundation,
                opts=self._opts,
                owner=owner,
            )
        return self._runtime_plan


install_grouped_view_reducers(BatchGroupedView)


__all__ = ["BatchGroupedView"]
