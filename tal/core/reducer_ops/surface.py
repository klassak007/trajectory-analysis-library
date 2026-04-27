from __future__ import annotations

from typing import TYPE_CHECKING

from ..schema_read import read_roles
from .api import reduce_analysis_object

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject
    from .types import DimLike, WeightInput


def _default_required_component_dims(self: "AnalysisObject") -> tuple[str, ...]:
    if self.__class__.__name__ == "AnalysisObject":
        return ()
    try:
        declared, _, _, core_dims = read_roles(self.unsafe_data)
    except Exception:
        return ()
    if not declared:
        return ()
    return tuple(core_dims)


def _dispatch(
    self: "AnalysisObject",
    *,
    op: str,
    dim: "DimLike" = None,
    skipna: bool = True,
    ddof: int = 0,
    weights: "WeightInput" = None,
    validate: bool = True,
) -> "AnalysisObject":
    return reduce_analysis_object(
        self,
        op=op,
        dim=dim,
        skipna=bool(skipna),
        ddof=int(ddof),
        weights=weights,
        validate=validate,
        owner=f"{self.__class__.__name__}.{op}",
    )


def _mean(self: "AnalysisObject", dim: "DimLike" = None, *, skipna: bool = True, weights: "WeightInput" = None, validate: bool = True):
    """Reduce with arithmetic mean."""
    return _dispatch(self, op="mean", dim=dim, skipna=skipna, weights=weights, validate=validate)


def _sum(self: "AnalysisObject", dim: "DimLike" = None, *, skipna: bool = True, weights: "WeightInput" = None, validate: bool = True):
    """Reduce with arithmetic sum."""
    return _dispatch(self, op="sum", dim=dim, skipna=skipna, weights=weights, validate=validate)


def _std(self: "AnalysisObject", dim: "DimLike" = None, *, skipna: bool = True, ddof: int = 0, weights: "WeightInput" = None, validate: bool = True):
    """Reduce with sample standard deviation."""
    return _dispatch(self, op="std", dim=dim, skipna=skipna, ddof=ddof, weights=weights, validate=validate)


def _var(self: "AnalysisObject", dim: "DimLike" = None, *, skipna: bool = True, ddof: int = 0, weights: "WeightInput" = None, validate: bool = True):
    """Reduce with sample variance."""
    return _dispatch(self, op="var", dim=dim, skipna=skipna, ddof=ddof, weights=weights, validate=validate)


def _median(self: "AnalysisObject", dim: "DimLike" = None, *, skipna: bool = True, weights: "WeightInput" = None, validate: bool = True):
    """Reduce with median."""
    return _dispatch(self, op="median", dim=dim, skipna=skipna, weights=weights, validate=validate)


def _min(self: "AnalysisObject", dim: "DimLike" = None, *, skipna: bool = True, weights: "WeightInput" = None, validate: bool = True):
    """Reduce with minimum."""
    return _dispatch(self, op="min", dim=dim, skipna=skipna, weights=weights, validate=validate)


def _max(self: "AnalysisObject", dim: "DimLike" = None, *, skipna: bool = True, weights: "WeightInput" = None, validate: bool = True):
    """Reduce with maximum."""
    return _dispatch(self, op="max", dim=dim, skipna=skipna, weights=weights, validate=validate)


def _count(self: "AnalysisObject", dim: "DimLike" = None, *, weights: "WeightInput" = None, validate: bool = True):
    """Reduce with non-null count."""
    return _dispatch(self, op="count", dim=dim, weights=weights, validate=validate)


def _any(self: "AnalysisObject", dim: "DimLike" = None, *, weights: "WeightInput" = None, validate: bool = True):
    """Reduce with logical any."""
    return _dispatch(self, op="any", dim=dim, weights=weights, validate=validate)


def _all(self: "AnalysisObject", dim: "DimLike" = None, *, weights: "WeightInput" = None, validate: bool = True):
    """Reduce with logical all."""
    return _dispatch(self, op="all", dim=dim, weights=weights, validate=validate)


_mean.__doc__ = """Reduce values with the arithmetic mean.

Parameters
----------
dim : str or iterable of str or None, optional
    Dimension or dimensions to reduce. ``None`` reduces all reducible
    dimensions.
skipna : bool, optional
    Whether missing values should be skipped.
weights : object, optional
    Optional weights passed to TAL reducer orchestration.
validate : bool, optional
    When ``True``, validate output schema/layout invariants before returning.

Returns
-------
AnalysisObject
    AO containing mean values with TAL metadata finalized for the reduced
    dimensions.

Notes
-----
Reducers follow xarray dimension-name semantics. Reducing the declared sequence
dimension removes sequence semantics; reducing batch or core dimensions updates
the corresponding role metadata.

Examples
--------
>>> import xarray as xr
>>> from tal.core import AnalysisObject
>>> ao = AnalysisObject.from_data(
...     xr.Dataset({"value": ("sample", [1.0, 3.0])}, coords={"sample": [0, 1]}),
...     sequence_dim="sample",
...     core_dims=(),
...     validate=True,
... )
>>> ao.mean(dim="sample").unsafe_data["value"].item()
2.0
"""

_sum.__doc__ = """Reduce values with arithmetic summation.

Parameters
----------
dim : str or iterable of str or None, optional
    Dimension or dimensions to reduce. ``None`` reduces all reducible
    dimensions.
skipna : bool, optional
    Whether missing values should be skipped.
weights : object, optional
    Optional weights passed to TAL reducer orchestration.
validate : bool, optional
    When ``True``, validate output schema/layout invariants before returning.

Returns
-------
AnalysisObject
    AO containing summed values with TAL metadata finalized for the reduced
    dimensions.

Notes
-----
Reducers follow xarray dimension-name semantics and repair role metadata after
the reduced dimensions are removed.

Examples
--------
>>> import xarray as xr
>>> from tal.core import AnalysisObject
>>> ao = AnalysisObject.from_data(
...     xr.Dataset({"value": ("sample", [1.0, 3.0])}, coords={"sample": [0, 1]}),
...     sequence_dim="sample",
...     core_dims=(),
...     validate=True,
... )
>>> ao.sum(dim="sample").unsafe_data["value"].item()
4.0
"""

_std.__doc__ = """Reduce values with standard deviation.

Parameters
----------
dim : str or iterable of str or None, optional
    Dimension or dimensions to reduce.
skipna : bool, optional
    Whether missing values should be skipped.
ddof : int, optional
    Delta degrees of freedom passed to the variance kernel.
weights : object, optional
    Optional weights passed to TAL reducer orchestration.
validate : bool, optional
    When ``True``, validate output schema/layout invariants before returning.

Returns
-------
AnalysisObject
    AO containing standard deviations with TAL metadata finalized for the
    reduced dimensions.

Notes
-----
The default ``ddof=0`` matches NumPy/xarray population standard deviation.

Examples
--------
>>> import xarray as xr
>>> from tal.core import AnalysisObject
>>> ao = AnalysisObject.from_data(
...     xr.Dataset({"value": ("sample", [1.0, 3.0])}, coords={"sample": [0, 1]}),
...     sequence_dim="sample",
...     core_dims=(),
...     validate=True,
... )
>>> ao.std(dim="sample").unsafe_data["value"].item()
1.0
"""

_var.__doc__ = """Reduce values with variance.

Parameters
----------
dim : str or iterable of str or None, optional
    Dimension or dimensions to reduce.
skipna : bool, optional
    Whether missing values should be skipped.
ddof : int, optional
    Delta degrees of freedom passed to the variance kernel.
weights : object, optional
    Optional weights passed to TAL reducer orchestration.
validate : bool, optional
    When ``True``, validate output schema/layout invariants before returning.

Returns
-------
AnalysisObject
    AO containing variances with TAL metadata finalized for the reduced
    dimensions.

Notes
-----
The default ``ddof=0`` matches NumPy/xarray population variance.

Examples
--------
>>> import xarray as xr
>>> from tal.core import AnalysisObject
>>> ao = AnalysisObject.from_data(
...     xr.Dataset({"value": ("sample", [1.0, 3.0])}, coords={"sample": [0, 1]}),
...     sequence_dim="sample",
...     core_dims=(),
...     validate=True,
... )
>>> ao.var(dim="sample").unsafe_data["value"].item()
1.0
"""

_median.__doc__ = """Reduce values with the median.

Parameters
----------
dim : str or iterable of str or None, optional
    Dimension or dimensions to reduce.
skipna : bool, optional
    Whether missing values should be skipped.
weights : object, optional
    Optional weights passed to TAL reducer orchestration.
validate : bool, optional
    When ``True``, validate output schema/layout invariants before returning.

Returns
-------
AnalysisObject
    AO containing median values with TAL metadata finalized for the reduced
    dimensions.

Notes
-----
Reducers follow xarray dimension-name semantics and preserve remaining TAL
roles by name.

Examples
--------
>>> import xarray as xr
>>> from tal.core import AnalysisObject
>>> ao = AnalysisObject.from_data(
...     xr.Dataset({"value": ("sample", [1.0, 3.0, 5.0])}, coords={"sample": [0, 1, 2]}),
...     sequence_dim="sample",
...     core_dims=(),
...     validate=True,
... )
>>> ao.median(dim="sample").unsafe_data["value"].item()
3.0
"""

_min.__doc__ = """Reduce values with the minimum.

Parameters
----------
dim : str or iterable of str or None, optional
    Dimension or dimensions to reduce.
skipna : bool, optional
    Whether missing values should be skipped.
weights : object, optional
    Optional weights passed to TAL reducer orchestration.
validate : bool, optional
    When ``True``, validate output schema/layout invariants before returning.

Returns
-------
AnalysisObject
    AO containing minima with TAL metadata finalized for the reduced
    dimensions.

Notes
-----
Reducers follow xarray dimension-name semantics and preserve remaining TAL
roles by name.

Examples
--------
>>> import xarray as xr
>>> from tal.core import AnalysisObject
>>> ao = AnalysisObject.from_data(
...     xr.Dataset({"value": ("sample", [1.0, 3.0])}, coords={"sample": [0, 1]}),
...     sequence_dim="sample",
...     core_dims=(),
...     validate=True,
... )
>>> ao.min(dim="sample").unsafe_data["value"].item()
1.0
"""

_max.__doc__ = """Reduce values with the maximum.

Parameters
----------
dim : str or iterable of str or None, optional
    Dimension or dimensions to reduce.
skipna : bool, optional
    Whether missing values should be skipped.
weights : object, optional
    Optional weights passed to TAL reducer orchestration.
validate : bool, optional
    When ``True``, validate output schema/layout invariants before returning.

Returns
-------
AnalysisObject
    AO containing maxima with TAL metadata finalized for the reduced
    dimensions.

Notes
-----
Reducers follow xarray dimension-name semantics and preserve remaining TAL
roles by name.

Examples
--------
>>> import xarray as xr
>>> from tal.core import AnalysisObject
>>> ao = AnalysisObject.from_data(
...     xr.Dataset({"value": ("sample", [1.0, 3.0])}, coords={"sample": [0, 1]}),
...     sequence_dim="sample",
...     core_dims=(),
...     validate=True,
... )
>>> ao.max(dim="sample").unsafe_data["value"].item()
3.0
"""

_count.__doc__ = """Reduce values with a non-null count.

Parameters
----------
dim : str or iterable of str or None, optional
    Dimension or dimensions to reduce.
weights : object, optional
    Optional weights passed to TAL reducer orchestration.
validate : bool, optional
    When ``True``, validate output schema/layout invariants before returning.

Returns
-------
AnalysisObject
    AO containing non-null counts with TAL metadata finalized for the reduced
    dimensions.

Notes
-----
``count`` follows xarray missing-value semantics for each variable.

Examples
--------
>>> import xarray as xr
>>> from tal.core import AnalysisObject
>>> ao = AnalysisObject.from_data(
...     xr.Dataset({"value": ("sample", [1.0, None])}, coords={"sample": [0, 1]}),
...     sequence_dim="sample",
...     core_dims=(),
...     validate=True,
... )
>>> ao.count(dim="sample").unsafe_data["value"].item()
1
"""

_any.__doc__ = """Reduce boolean values with logical any.

Parameters
----------
dim : str or iterable of str or None, optional
    Dimension or dimensions to reduce.
weights : object, optional
    Optional weights passed to TAL reducer orchestration.
validate : bool, optional
    When ``True``, validate output schema/layout invariants before returning.

Returns
-------
AnalysisObject
    AO containing logical-any results with TAL metadata finalized for the
    reduced dimensions.

Notes
-----
Use this reducer for boolean variables or masks; non-boolean variables follow
xarray truth-reduction behavior.

Examples
--------
>>> import xarray as xr
>>> from tal.core import AnalysisObject
>>> ao = AnalysisObject.from_data(
...     xr.Dataset({"flag": ("sample", [False, True])}, coords={"sample": [0, 1]}),
...     sequence_dim="sample",
...     core_dims=(),
...     validate=True,
... )
>>> bool(ao.any(dim="sample").unsafe_data["flag"].item())
True
"""

_all.__doc__ = """Reduce boolean values with logical all.

Parameters
----------
dim : str or iterable of str or None, optional
    Dimension or dimensions to reduce.
weights : object, optional
    Optional weights passed to TAL reducer orchestration.
validate : bool, optional
    When ``True``, validate output schema/layout invariants before returning.

Returns
-------
AnalysisObject
    AO containing logical-all results with TAL metadata finalized for the
    reduced dimensions.

Notes
-----
Use this reducer for boolean variables or masks; non-boolean variables follow
xarray truth-reduction behavior.

Examples
--------
>>> import xarray as xr
>>> from tal.core import AnalysisObject
>>> ao = AnalysisObject.from_data(
...     xr.Dataset({"flag": ("sample", [True, True])}, coords={"sample": [0, 1]}),
...     sequence_dim="sample",
...     core_dims=(),
...     validate=True,
... )
>>> bool(ao.all(dim="sample").unsafe_data["flag"].item())
True
"""


def install_analysis_object_reducers(cls: type) -> None:
    cls._required_component_dims_for_reduce = _default_required_component_dims
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


__all__ = ["install_analysis_object_reducers"]
