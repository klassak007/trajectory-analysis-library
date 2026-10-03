from __future__ import annotations

from collections.abc import Callable


import dask.array as da

import numpy as np

import pytest

import xarray as xr

from dask.callbacks import Callback

from tal_extensions.astro import TopocentricDirection

from tal.core import AnalysisObject


from tal_extensions.geo import GeodeticPosition, ProjectedPosition


TypedConstructor = Callable[[object], AnalysisObject]

TypedSourceFactory = Callable[[], AnalysisObject]


class _TaskCounter(Callback):
    def __init__(self) -> None:
        self.keys: list[object] = []
        super().__init__(pretask=self._record)

    def _record(self, key: object, _dsk: object, _state: object) -> None:
        self.keys.append(key)


def _analysis_object(
    values: object,
    *,
    dims: tuple[str, ...],
    coords: dict[str, object],
    var_name: str,
    core_dims: tuple[str, ...],
) -> AnalysisObject:
    ds = xr.Dataset({var_name: (dims, values)}, coords=coords)
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        core_dims=core_dims,
        validate=True,
    )


def _vector_source(
    *,
    var_name: str = "value",
    core_dim: str = "axis",
    labels: tuple[str, ...] = ("x", "y", "z"),
    lazy: bool = False,
) -> AnalysisObject:
    eager = np.arange(2 * len(labels), dtype=float).reshape(2, len(labels))
    values: object = da.from_array(eager, chunks=(1, len(labels))) if lazy else eager
    return _analysis_object(
        values,
        dims=("sample", core_dim),
        coords={"sample": [0, 1], core_dim: list(labels)},
        var_name=var_name,
        core_dims=(core_dim,),
    )


def _generic_source(value: AnalysisObject) -> AnalysisObject:
    return AnalysisObject._from_validated(value.as_dataset(copy="deep"))


def _projected_source() -> AnalysisObject:
    source = _vector_source(
        var_name="position",
        core_dim="projected",
        labels=("easting", "northing"),
    )
    return _generic_source(ProjectedPosition.from_projected(source, crs="EPSG:32611"))


_PROMOTION_CASES = {
    "geodetic": (
        GeodeticPosition,
        lambda: _vector_source(
            var_name="position", core_dim="lla", labels=("lat", "lon", "alt")
        ),
    ),
    "topocentric": (
        TopocentricDirection,
        lambda: _vector_source(
            var_name="direction", core_dim="enu", labels=("east", "north", "up")
        ),
    ),
    "projected": (ProjectedPosition, _projected_source),
}


def _promotion_case(name: str) -> tuple[TypedConstructor, AnalysisObject]:
    constructor, source_factory = _PROMOTION_CASES[name]
    return constructor, source_factory()


def _decorate_owned_metadata(ao: AnalysisObject) -> None:
    ds = ao.as_dataset(copy="none")
    ds.attrs["nested"] = {"items": ["dataset"]}
    ds.encoding["nested"] = {"items": ["dataset-encoding"]}
    for name in ds.variables:
        ds[name].attrs["nested"] = {"items": [f"{name}-attrs"]}
        ds[name].encoding["nested"] = {"items": [f"{name}-encoding"]}


@pytest.mark.parametrize("family", ("geodetic", "projected", "topocentric"))
def test_typed_ownership_001_all_constructor_families_share_owned_payloads(
    family: str,
) -> None:
    """ID: EXT_TYPED_OWNERSHIP_001_all_constructor_families_share_owned_payloads."""
    constructor, source = _promotion_case(family)
    _decorate_owned_metadata(source)
    source_ds = source.as_dataset(copy="none")
    before = source.as_dataset(copy="deep")

    promoted = constructor(source)
    promoted_ds = promoted.as_dataset(copy="none")

    xr.testing.assert_identical(source_ds, before)
    assert promoted_ds is not source_ds
    for name in source_ds.data_vars:
        assert promoted_ds.variables[name] is not source_ds.variables[name]
        assert np.shares_memory(promoted_ds[name].data, source_ds[name].data)
    for name in source_ds.coords:
        assert promoted_ds.variables[name] is not source_ds.variables[name]
        source_data = source_ds.coords[name].data
        if np.shares_memory(source_data, source_ds.coords[name].data):
            assert np.shares_memory(promoted_ds.coords[name].data, source_data)
    for name, source_index in source_ds.xindexes.items():
        assert promoted_ds.xindexes[name] is not source_index
        assert type(promoted_ds.xindexes[name]) is type(source_index)
        assert promoted_ds.xindexes[name].equals(source_index)
    assert promoted_ds.attrs["nested"] is not source_ds.attrs["nested"]
    promoted_ds.attrs["nested"]["items"].append("promoted")
    first_var = next(iter(source_ds.data_vars))
    promoted_ds[first_var].encoding["nested"]["items"].append("promoted")
    first_coord = next(iter(source_ds.coords))
    promoted_ds[first_coord].attrs["nested"]["items"].append("promoted")
    assert source_ds.attrs["nested"]["items"] == ["dataset"]
    assert source_ds[first_var].encoding["nested"]["items"] == [f"{first_var}-encoding"]
    assert source_ds[first_coord].attrs["nested"]["items"] == [f"{first_coord}-attrs"]
    if family == "topocentric":
        assert set(promoted_ds.data_vars) - set(source_ds.data_vars) == {
            "altitude_deg",
            "azimuth_deg",
        }


def test_typed_ownership_009_normalizing_dask_promotion_adds_only_derived_graphs() -> (
    None
):
    """ID: EXT_TYPED_OWNERSHIP_009_normalizing_dask_promotion_adds_only_derived_graphs."""
    source = _vector_source(
        var_name="direction",
        core_dim="enu",
        labels=("east", "north", "up"),
        lazy=True,
    )
    source_data = source.as_dataset(copy="none")["direction"].data
    source_keys = set(source_data.__dask_graph__())
    counter = _TaskCounter()

    with counter:
        promoted = TopocentricDirection(source)

    promoted_ds = promoted.as_dataset(copy="none")
    assert counter.keys == []
    assert promoted_ds["direction"].data is source_data
    assert set(promoted_ds.data_vars) == {
        "direction",
        "altitude_deg",
        "azimuth_deg",
    }
    for name in ("altitude_deg", "azimuth_deg"):
        derived = promoted_ds[name].data
        assert isinstance(derived, da.Array)
        assert source_keys <= set(derived.__dask_graph__())
