"""Reusable, complete core-layout declarations for external ingress."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import xarray as xr

from .analysis_object import AnalysisObject
from .schema import _SchemaUpdatePlan


def _optional_name(value: object, *, field: str) -> None:
    if value is not None and (type(value) is not str or not value):
        raise TypeError(f"AnalysisLayoutSpec: {field} must be None or a non-empty string.")


def _role_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if type(value) is not tuple or any(type(item) is not str or not item for item in value):
        raise TypeError(f"AnalysisLayoutSpec: {field} must be a tuple of non-empty strings.")
    if len(set(value)) != len(value):
        raise ValueError(f"AnalysisLayoutSpec: {field} contains duplicate dimensions.")
    return value


@dataclass(frozen=True)
class AnalysisLayoutSpec:
    """Declare a complete, reusable TAL core layout.

    Unlike ``AnalysisObject.from_data``, absent fields clear existing
    declarations rather than inheriting them from the source Dataset.
    """

    sequence_dim: str | None = None
    batch_dims: tuple[str, ...] = ()
    core_dims: tuple[str, ...] = ()
    param_coord: str | None = None
    sequence_size_coord: str | None = None

    def __post_init__(self) -> None:
        for field in ("sequence_dim", "param_coord", "sequence_size_coord"):
            _optional_name(getattr(self, field), field=field)
        batch = _role_tuple(self.batch_dims, field="batch_dims")
        core = _role_tuple(self.core_dims, field="core_dims")
        roles = ((self.sequence_dim,) if self.sequence_dim is not None else ()) + batch + core
        if len(set(roles)) != len(roles):
            raise ValueError("AnalysisLayoutSpec: sequence, batch, and core dimensions must be disjoint.")
        if self.sequence_dim is None and (self.param_coord is not None or self.sequence_size_coord is not None):
            raise ValueError("AnalysisLayoutSpec: parameter and size coordinates require sequence_dim.")
        if self.param_coord is not None and self.param_coord == self.sequence_size_coord:
            raise ValueError("AnalysisLayoutSpec: parameter and size coordinates must be distinct.")

    def wrap(
        self,
        data: xr.Dataset | xr.DataArray,
        *,
        data_vars: str | Sequence[str] | None = None,
        validate: bool = True,
    ) -> AnalysisObject:
        """Wrap external xarray data using this complete layout declaration.

        Parameters
        ----------
        data
            External Dataset or DataArray.
        data_vars
            Ordered Dataset data-variable selection, or ``None`` for all.
        validate
            Whether to perform the optional final whole-schema validation.

        Returns
        -------
        AnalysisObject
            An owning, metadata-isolated result.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisLayoutSpec
        >>> layout = AnalysisLayoutSpec(sequence_dim="sample")
        >>> ds = xr.Dataset({"x": ("sample", [1.0, 2.0]), "y": ("sample", [3.0, 4.0])})
        >>> list(layout.wrap(ds, data_vars="x").as_dataset().data_vars)
        ['x']
        """
        from .layout_ingress import complete_ingress, selected_external_ingress

        plan = _SchemaUpdatePlan(
            sequence_dim=self.sequence_dim,
            batch_dims=self.batch_dims,
            core_dims=self.core_dims,
            param_coord=self.param_coord,
            sequence_size_coord=self.sequence_size_coord,
            complete_target=True,
        )
        if data_vars is not None:
            return selected_external_ingress(AnalysisObject, data, names=data_vars, plan=plan, validate=validate)
        return complete_ingress(AnalysisObject, data, plan=plan, validate=validate, owner="AnalysisLayoutSpec.wrap")
