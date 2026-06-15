from __future__ import annotations

"""Auto-grid synthesis orchestration for param synchronization."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr

from .sync_autogrid_backend import join_datetime_rows_batched, join_rows_batched
from .types import ParamRuntimeContext


@dataclass(frozen=True)
class AutoGridJoinInputs:
    """Materialized row matrices for auto-grid synthesis joins.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    sequence_dim: str
    spec_name: str
    batch_dim: str | None
    batch_labels: xr.DataArray | None
    param_rows: tuple[np.ndarray, ...]
    valid_rows: tuple[np.ndarray, ...]


def _materialize_data_rows(
    da: xr.DataArray,
    *,
    sequence_dim: str,
    batch_dim: str | None,
    batch_size: int,
    dtype: np.dtype,
) -> np.ndarray:
    if batch_dim is None:
        row = np.asarray(da.transpose(sequence_dim).data, dtype=dtype)
        return row[np.newaxis, :]
    if batch_dim in da.dims:
        return np.asarray(da.transpose(batch_dim, sequence_dim).data, dtype=dtype)
    row = np.asarray(da.transpose(sequence_dim).data, dtype=dtype)
    return np.broadcast_to(row, (batch_size, row.size))


def _materialize_param_rows(
    da: xr.DataArray,
    *,
    sequence_dim: str,
    batch_dim: str | None,
    batch_size: int,
    param_kind: str,
) -> np.ndarray:
    if param_kind == "datetime64":
        rows = _materialize_data_rows(
            da.astype("datetime64[ns]"),
            sequence_dim=sequence_dim,
            batch_dim=batch_dim,
            batch_size=batch_size,
            dtype=np.dtype("datetime64[ns]"),
        )
        return rows.view("int64")
    return _materialize_data_rows(
        da,
        sequence_dim=sequence_dim,
        batch_dim=batch_dim,
        batch_size=batch_size,
        dtype=np.dtype("float64"),
    )


def _materialize_join_inputs(
    contexts: Sequence[ParamRuntimeContext],
    *,
    param_kind: str,
) -> AutoGridJoinInputs:
    batch_dim = contexts[0].batch_dims[0] if contexts[0].batch_dims else None
    batch_labels = contexts[0].batch_coords[batch_dim] if batch_dim is not None else None
    batch_size = int(batch_labels.size) if batch_labels is not None else 1
    param_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    for context in contexts:
        param_rows.append(
            _materialize_param_rows(
                context.spec.coord,
                sequence_dim=context.sequence_dim,
                batch_dim=batch_dim,
                batch_size=batch_size,
                param_kind=param_kind,
            )
        )
        valid_rows.append(
            _materialize_data_rows(
                context.valid_mask,
                sequence_dim=context.sequence_dim,
                batch_dim=batch_dim,
                batch_size=batch_size,
                dtype=np.dtype("bool"),
            )
        )
    return AutoGridJoinInputs(
        sequence_dim=contexts[0].sequence_dim,
        spec_name=contexts[0].spec.name,
        batch_dim=batch_dim,
        batch_labels=batch_labels,
        param_rows=tuple(param_rows),
        valid_rows=tuple(valid_rows),
    )


def build_auto_grid_from_join(
    contexts: Sequence[ParamRuntimeContext],
    *,
    join: Literal["outer", "inner", "domain", "exact"],
    tol: float | int,
    owner: str,
    param_kind: str = "numeric",
) -> xr.DataArray:
    """Build a synthesized target grid from per-row param-domain joins.

    Parameters
    ----------
    contexts : Sequence[ParamRuntimeContext]
        Resolved runtime context/payload used by this orchestration boundary.
    join : Literal['outer', 'inner', 'domain', 'exact'], optional
        Policy selector controlling alignment/join behavior.
    tol : float | int, optional
        Numeric tolerance used for matching/alignment logic.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.
    param_kind : {'numeric', 'datetime64'}, optional
        Parameter coordinate kind used to materialize/join rows.

    Returns
    -------
    xr.DataArray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    inputs = _materialize_join_inputs(contexts, param_kind=param_kind)
    if param_kind == "datetime64":
        joined = join_datetime_rows_batched(
            inputs.param_rows,
            inputs.valid_rows,
            join=join,
            tol=int(tol),
            owner=owner,
        ).view("datetime64[ns]")
    else:
        joined = join_rows_batched(
            inputs.param_rows,
            inputs.valid_rows,
            join=join,
            tol=float(tol),
            owner=owner,
        )
    if inputs.batch_dim is None:
        return xr.DataArray(joined[0], dims=[inputs.sequence_dim], name=inputs.spec_name)
    width = joined.shape[1]
    return xr.DataArray(
        joined,
        dims=[inputs.batch_dim, inputs.sequence_dim],
        coords={
            inputs.batch_dim: inputs.batch_labels,
            inputs.sequence_dim: np.arange(width, dtype="int64"),
        },
    )


__all__ = ["AutoGridJoinInputs", "build_auto_grid_from_join"]
