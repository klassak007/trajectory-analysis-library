from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from tal.core.param_ops import ParamAccessor

from .options import GeodeticInterpolationOptions, coerce_geodetic_interpolation_options

if TYPE_CHECKING:
    from .geodetic import GeodeticPosition


class GeodeticParamAccessor(ParamAccessor):
    """Typed param accessor for geodetic interpolation defaults.

    Notes
    -----
    Public TAL class surface. Methods delegate to ``tal.geo.interpolation`` so
    query mapping and finalization remain core-owned.
    """

    def at(
        self,
        query: xr.DataArray | np.ndarray | Sequence[float] | float,
        *,
        on: str | None = None,
        opts: GeodeticInterpolationOptions | None = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "GeodeticPosition":
        """Evaluate geodetic LLA values at param queries.

        Parameters
        ----------
        query : xarray.DataArray | numpy.ndarray | Sequence[float] | float
            Query coordinate/grid for param-aware interpolation.
        on : str | None, optional
            Coordinate/dimension name used as the operation domain.
        opts : GeodeticInterpolationOptions | None, optional
            Geodetic interpolation options. Defaults use
            ``method='geodesic_linear'`` and ``longitude_wrap='shortest'``.
        validate : bool, optional
            Whether to validate the output before returning.
        sequence_dim : str | None, optional
            Optional sequence dimension override.
        batch_dims : Sequence[str] | None, optional
            Optional batch dimension override.
        sequence_size_coord : str | None, optional
            Optional sequence-size coordinate for ragged validity handling.

        Returns
        -------
        GeodeticPosition
            Interpolated geodetic position.

        Raises
        ------
        ImportError
            If the selected method requires ``pyproj`` from ``tal[geo]``.
        TypeError
            If ``opts`` or query inputs have unsupported types.
        ValueError
            If param topology, duplicate labels for linear methods, local
            origins, longitude wrapping, or geodetic metadata are invalid.

        Notes
        -----
        ``nearest`` follows core nearest-selection behavior. Linear geodetic
        methods reuse core param mapping, including its eager query-map
        boundary, while payload math remains xarray/Dask-aware where supported.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import GeodeticInterpolationOptions, GeodeticPosition
        >>> ds = xr.Dataset(
        ...     {"lla": (("sample", "lla_axis"), [[0.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "lla_axis": ["lat", "lon", "alt"], "time_s": ("sample", [0.0])},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla_axis",), param_coord="time_s", validate=True)
        >>> opts = GeodeticInterpolationOptions(method="nearest")
        >>> out = GeodeticPosition.from_lla(ao).param.at([0.0], on="time_s", opts=opts)
        >>> isinstance(out, GeodeticPosition)
        True
        """
        from .interpolation import geodetic_param_at

        normalized = coerce_geodetic_interpolation_options(opts, owner="geo.GeodeticPosition.param.at")
        return geodetic_param_at(
            self._ao,
            query=query,
            on=on,
            opts=normalized,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="geo.GeodeticPosition.param.at",
        )

    def resample_to(
        self,
        grid: xr.DataArray | np.ndarray | Sequence[float] | float,
        *,
        on: str | None = None,
        opts: GeodeticInterpolationOptions | None = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "GeodeticPosition":
        """Resample geodetic LLA values to a target param grid.

        Parameters
        ----------
        grid : xarray.DataArray | numpy.ndarray | Sequence[float] | float
            Target query grid for geodetic interpolation.
        on : str | None, optional
            Coordinate/dimension name used as the operation domain.
        opts : GeodeticInterpolationOptions | None, optional
            Geodetic interpolation options including ``method``,
            ``longitude_wrap``, ``duplicate_policy``, and ``query_dim``.
        validate : bool, optional
            Whether to validate the output before returning.
        sequence_dim : str | None, optional
            Optional sequence dimension override.
        batch_dims : Sequence[str] | None, optional
            Optional batch dimension override.
        sequence_size_coord : str | None, optional
            Optional sequence-size coordinate for ragged validity handling.

        Returns
        -------
        GeodeticPosition
            Resampled geodetic position.

        Raises
        ------
        ImportError
            If the selected method requires ``pyproj`` from ``tal[geo]``.
        TypeError
            If ``opts`` or query inputs have unsupported types.
        ValueError
            If param topology, duplicate labels for linear methods, local
            origins, longitude wrapping, or geodetic metadata are invalid.

        Notes
        -----
        This method is equivalent to :meth:`at` with resampling naming and the
        same geodetic interpolation semantics.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import GeodeticInterpolationOptions, GeodeticPosition
        >>> ds = xr.Dataset(
        ...     {"lla": (("sample", "lla_axis"), [[0.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "lla_axis": ["lat", "lon", "alt"], "time_s": ("sample", [0.0])},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla_axis",), param_coord="time_s", validate=True)
        >>> opts = GeodeticInterpolationOptions(method="nearest", query_dim="target")
        >>> out = GeodeticPosition.from_lla(ao).param.resample_to([0.0], on="time_s", opts=opts)
        >>> isinstance(out, GeodeticPosition)
        True
        """
        from .interpolation import geodetic_param_resample_to

        normalized = coerce_geodetic_interpolation_options(opts, owner="geo.GeodeticPosition.param.resample_to")
        return geodetic_param_resample_to(
            self._ao,
            grid=grid,
            on=on,
            opts=normalized,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="geo.GeodeticPosition.param.resample_to",
        )

    def interp_like(
        self,
        other: object,
        *,
        on: str | None = None,
        opts: GeodeticInterpolationOptions | None = None,
        batch_join: Literal["inner", "left"] = "inner",
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] | None = None,
        sequence_size_coord: str | None = None,
    ) -> "GeodeticPosition":
        """Interpolate geodetic LLA values onto another object's param grid.

        Parameters
        ----------
        other : object
            AO-like object carrying the target param coordinate named by
            ``on`` or by the source's declared param coordinate.
        on : str | None, optional
            Coordinate/dimension name used as the operation domain.
        opts : GeodeticInterpolationOptions | None, optional
            Geodetic interpolation options including ``method``,
            ``longitude_wrap``, ``duplicate_policy``, and ``query_dim``.
        batch_join : {'inner', 'left'}, optional
            Batch join policy accepted for API parity with core param
            interpolation. Batched ``'inner'`` joins fail closed; use
            ``'left'`` or pass an explicit aligned query to :meth:`resample_to`.
        validate : bool, optional
            Whether to validate the output before returning.
        sequence_dim : str | None, optional
            Optional sequence dimension override.
        batch_dims : Sequence[str] | None, optional
            Optional batch dimension override.
        sequence_size_coord : str | None, optional
            Optional sequence-size coordinate for ragged validity handling.

        Returns
        -------
        GeodeticPosition
            Geodetic position interpolated onto ``other``'s param grid.

        Raises
        ------
        ImportError
            If the selected method requires ``pyproj`` from ``tal[geo]``.
        TypeError
            If ``opts``, ``other``, or local origins have unsupported types.
        ValueError
            If param topology, duplicate labels for linear methods, local
            origins, pole interpolation, longitude wrapping, or geodetic
            metadata are invalid.

        Notes
        -----
        This override intentionally bypasses generic component-wise LLA
        interpolation. It extracts the target param coordinate from ``other``
        and delegates to TAL's geodetic interpolation owner.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import GeodeticInterpolationOptions, GeodeticPosition
        >>> src = xr.Dataset(
        ...     {"lla": (("sample", "lla_axis"), [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])},
        ...     coords={"sample": [0, 1], "lla_axis": ["lat", "lon", "alt"], "time_s": ("sample", [0.0, 10.0])},
        ... )
        >>> target = xr.Dataset({"value": ("query", [1.0])}, coords={"query": [0], "time_s": ("query", [5.0])})
        >>> ao = AnalysisObject.from_data(src, sequence_dim="sample", core_dims=("lla_axis",), param_coord="time_s", validate=True)
        >>> opts = GeodeticInterpolationOptions(method="nearest", query_dim="target")
        >>> out = GeodeticPosition.from_lla(ao).param.interp_like(target, on="time_s", opts=opts)
        >>> isinstance(out, GeodeticPosition)
        True
        """
        from .interpolation import geodetic_param_interp_like

        normalized = coerce_geodetic_interpolation_options(opts, owner="geo.GeodeticPosition.param.interp_like")
        return geodetic_param_interp_like(
            self._ao,
            other=other,
            on=on,
            opts=normalized,
            batch_join=batch_join,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="geo.GeodeticPosition.param.interp_like",
        )


__all__ = ["GeodeticParamAccessor"]
