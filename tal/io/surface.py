from __future__ import annotations

from .csv_io import read_analysis_object_csv, write_analysis_object_csv
from .options import AOCsvReadOptions, AOCsvWriteOptions, AOZarrReadOptions, AOZarrWriteOptions
from .zarr_io import read_analysis_object_zarr, write_analysis_object_zarr


class AnalysisObjectIOAccessor:
    """AO-direct I/O accessor rooted at ``ao.io``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    def __init__(self, ao: "AnalysisObject") -> None:
        self._ao = ao

    def to_zarr(self, store: str, *, opts: AOZarrWriteOptions | None = None):
        """Persist this AnalysisObject to a Zarr store.

        Parameters
        ----------
        store : str
            Zarr store location used by this IO operation.
        opts : AOZarrWriteOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``AOZarrWriteOptions`` key fields: ``mode`` (default None), ``consolidated`` (default None).

        Returns
        -------
        object
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
        >>> import tempfile
        >>> import xarray as xr
        >>> from pathlib import Path
        >>> from tal.core import AnalysisObject
        >>> from tal.io import AOZarrWriteOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     store = Path(tmpdir) / "trajectory.zarr"
        ...     _ = ao.io.to_zarr(str(store), opts=AOZarrWriteOptions(mode="w"))
        ...     loaded = AnalysisObject.from_zarr(str(store))
        >>> loaded.unsafe_data["value"].values.tolist()
        [1.0, 2.0]
        """
        return write_analysis_object_zarr(self._ao, store, opts=opts, owner="AnalysisObject.io.to_zarr")

    def to_csv(self, path: str, *, opts: AOCsvWriteOptions | None = None) -> str:
        """Write this AnalysisObject to CSV files rooted at ``path``.

        Parameters
        ----------
        path : str
            Filesystem path used by this IO operation.
        opts : AOCsvWriteOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``AOCsvWriteOptions`` key fields: ``metadata_channel`` (default 'sidecar_json'), ``metadata_path`` (default None), ``float_format`` (default None).

        Returns
        -------
        str
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
        >>> import tempfile
        >>> import xarray as xr
        >>> from pathlib import Path
        >>> from tal.core import AnalysisObject
        >>> from tal.io import AOCsvWriteOptions
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = ao.io.to_csv(str(Path(tmpdir) / "trajectory"), opts=AOCsvWriteOptions())
        ...     loaded = AnalysisObject.from_csv(path)
        >>> loaded.unsafe_data["value"].values.tolist()
        [1.0, 2.0]
        """
        return write_analysis_object_csv(self._ao, path, opts=opts, owner="AnalysisObject.io.to_csv")


def _from_zarr(
    cls: type["AnalysisObject"],
    store: str,
    *,
    opts: AOZarrReadOptions | None = None,
    validate: bool = True,
):
    """Load an AnalysisObject from a Zarr store via ``AnalysisObject.from_zarr``.

    Parameters
    ----------
    store : str
        Zarr store location used by this IO operation.
    opts : AOZarrReadOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``AOZarrReadOptions`` key fields: ``consolidated`` (default None), ``chunks`` (default None).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    object
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
    >>> import tempfile
    >>> import xarray as xr
    >>> from pathlib import Path
    >>> from tal.core import AnalysisObject
    >>> from tal.io import AOZarrReadOptions, AOZarrWriteOptions
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     store = Path(tmpdir) / "trajectory.zarr"
    ...     _ = ao.io.to_zarr(str(store), opts=AOZarrWriteOptions(mode="w"))
    ...     loaded = AnalysisObject.from_zarr(str(store), opts=AOZarrReadOptions())
    >>> loaded.unsafe_data.sizes["sample"]
    2
    """
    return read_analysis_object_zarr(
        cls,
        store,
        opts=opts,
        validate=validate,
        owner=f"{cls.__name__}.from_zarr",
    )


def _from_csv(
    cls: type["AnalysisObject"],
    path: str,
    *,
    opts: AOCsvReadOptions | None = None,
    validate: bool = True,
):
    """Load an AnalysisObject from a CSV path via ``AnalysisObject.from_csv``.

    Parameters
    ----------
    path : str
        Filesystem path used by this IO operation.
    opts : AOCsvReadOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``AOCsvReadOptions`` key fields: ``metadata_channel`` (default 'sidecar_json'), ``metadata_path`` (default None).
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    object
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
    >>> import tempfile
    >>> import xarray as xr
    >>> from pathlib import Path
    >>> from tal.core import AnalysisObject
    >>> from tal.io import AOCsvReadOptions, AOCsvWriteOptions
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = ao.io.to_csv(str(Path(tmpdir) / "trajectory"), opts=AOCsvWriteOptions())
    ...     loaded = AnalysisObject.from_csv(path, opts=AOCsvReadOptions())
    >>> loaded.unsafe_data.sizes["sample"]
    2
    """
    return read_analysis_object_csv(
        cls,
        path,
        opts=opts,
        validate=validate,
        owner=f"{cls.__name__}.from_csv",
    )


def _install_io_property(cls: type) -> None:
    existing = getattr(cls, "io", None)
    if isinstance(existing, property):
        fget = existing.fget
        if fget is not None and getattr(fget, "__module__", "") == __name__:
            return
        raise ValueError(
            "install_analysis_object_io_surface: AnalysisObject.io is already owned by another accessor."
        )
    if existing is not None:
        raise ValueError(
            "install_analysis_object_io_surface: AnalysisObject.io already exists and is not a property."
        )

    def _io_accessor(self: "AnalysisObject") -> AnalysisObjectIOAccessor:
        """Return the ``ao.io`` accessor bound to this AnalysisObject."""
        return AnalysisObjectIOAccessor(self)

    cls.io = property(_io_accessor)  # type: ignore[assignment]


def _install_classmethod(cls: type, *, name: str, func) -> None:
    existing = getattr(cls, name, None)
    if existing is None:
        setattr(cls, name, classmethod(func))
        return
    if getattr(existing, "__module__", "") == __name__:
        return
    raise ValueError(
        f"install_analysis_object_io_surface: AnalysisObject.{name} already exists and is not owned by tal.io.surface."
    )


def install_analysis_object_io_surface() -> None:
    """Install ``ao.io`` and ``AnalysisObject.from_*`` I/O bindings.

    Parameters
    ----------
    None
        This callable does not accept user-facing parameters.

    Returns
    -------
    None
        Returns ``None``; side effects are applied through owned state/metadata updates.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    from tal.core.analysis_object import AnalysisObject

    _install_io_property(AnalysisObject)
    _install_classmethod(AnalysisObject, name="from_zarr", func=_from_zarr)
    _install_classmethod(AnalysisObject, name="from_csv", func=_from_csv)


__all__ = ["AnalysisObjectIOAccessor", "install_analysis_object_io_surface"]
