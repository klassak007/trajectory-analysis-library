"""Public coordinate restoration guarantees from Contracts 009 and 012."""

import warnings
from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject


class _UnevaluatedTopologyTransform(xr.indexes.CoordinateTransform):
    def __init__(self, size: int, calls: list[str], *, dim: str) -> None:
        self.calls = calls
        super().__init__((dim,), {dim: size})

    def forward(self, dim_positions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("forward")
        dim = self.coord_names[0]
        return {dim: dim_positions[dim] + 0.25}

    def reverse(self, coord_labels: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("reverse")
        dim = self.coord_names[0]
        return {dim: coord_labels[dim] - 0.25}

    def equals(self, other: object, **kwargs: object) -> bool:
        _ = kwargs
        return (
            isinstance(other, _UnevaluatedTopologyTransform)
            and self.coord_names == other.coord_names
            and self.dim_size == other.dim_size
        )


class _TaggedTopologyTransform(_UnevaluatedTopologyTransform):
    def __init__(self, size: int, calls: list[str], *, dim: str, tag: str) -> None:
        self.tag = tag
        super().__init__(size, calls, dim=dim)

    def equals(self, other: object, **kwargs: object) -> bool:
        return super().equals(other, **kwargs) and isinstance(other, _TaggedTopologyTransform) and self.tag == other.tag


def _source(*, lazy: bool) -> AnalysisObject:
    ds = xr.Dataset(
        {"value": ("sample", [0.0, 10.0, 20.0])},
        coords={
            "time": ("sample", [0.0, 1.0, 2.0]),
            "tag": ("sample", [4, 5, 6]),
            "group_size": 3,
        },
    )
    ds.coords["tag"].attrs["description"] = "source labels"
    if lazy:
        pytest.importorskip("dask.array")
        ds = ds.chunk({"sample": 3})
    return AnalysisObject.from_data(
        ds, sequence_dim="sample", core_dims=(), param_coord="time",
        sequence_size_coord="group_size",
    )


def _task_counter(tasks: list[object], *, lazy: bool):
    if not lazy:
        return nullcontext()
    from dask.callbacks import Callback

    return Callback(pretask=lambda key, *_: tasks.append(key))


@pytest.mark.parametrize("lazy", (False, True))
def test_param_query_coordinates_001_nonempty_selection_restores_topology(lazy: bool) -> None:
    """ID: PARAM_QUERY_COORDINATES_001_nonempty_selection_restores_topology."""
    source = _source(lazy=lazy)
    query = xr.DataArray(
        [[0.25, 0.75], [1.25, 1.75]],
        dims=("row", "col"),
        coords={"row": ["b", "a"], "col": [4, 3], "note": ("row", [8, 9])},
    )
    if lazy:
        query = query.chunk({"row": 1, "col": 2})
    before = source.as_dataset(copy="deep")
    query_before = query.copy(deep=True)
    tasks: list[object] = []
    with warnings.catch_warnings(), _task_counter(tasks, lazy=lazy):
        warnings.simplefilter("error")
        actual = source.param.sel(query).as_dataset(copy="none")
    assert tasks == []
    assert actual["value"].dims == ("row", "col")
    computed = actual.compute(scheduler="synchronous")
    np.testing.assert_array_equal(computed["value"], [[0.0, 10.0], [10.0, 20.0]])
    np.testing.assert_array_equal(computed.coords["sample_index"], [[0, 1], [1, 2]])
    np.testing.assert_array_equal(computed.coords["time"], [[0.0, 1.0], [1.0, 2.0]])
    np.testing.assert_array_equal(computed.coords["tag"], [[4, 5], [5, 6]])
    assert bool(computed.coords["valid"].all())
    for dim in query.dims:
        assert computed.xindexes[dim].equals(query.xindexes[dim])
    np.testing.assert_array_equal(
        computed.coords["note"].broadcast_like(computed["value"]), [[8, 8], [9, 9]],
    )
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)
    xr.testing.assert_identical(query, query_before)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("operation", ("at", "sel", "resample_to"))
@pytest.mark.parametrize("shape", ((2, 0), (0, 2), (2, 0, 3)))
def test_param_query_coordinates_002_empty_results_retain_generated_coordinates(
    lazy: bool, operation: str, shape: tuple[int, ...],
) -> None:
    """ID: PARAM_QUERY_COORDINATES_002_empty_results_retain_generated_coordinates."""
    source = _source(lazy=lazy)
    dims = tuple(f"axis{i}" for i in range(len(shape)))
    query = xr.DataArray(np.empty(shape), dims=dims)
    if lazy:
        query = query.chunk({dim: max(size, 1) for dim, size in zip(dims, shape, strict=True)})
    before = source.as_dataset(copy="deep")
    tasks: list[object] = []
    with warnings.catch_warnings(), _task_counter(tasks, lazy=lazy):
        warnings.simplefilter("error")
        actual = getattr(source.param, operation)(query).as_dataset(copy="none")
    assert tasks == []
    names = ("valid", "sample_index", "time", "tag") if operation == "sel" else ("valid",)
    for name in names:
        assert actual.coords[name].dims == dims
        assert actual.coords[name].shape == shape
    assert actual.coords["valid"].dtype == np.dtype(bool)
    assert actual.coords["valid"].attrs  # TAL reserved-coordinate ownership survives.
    if operation == "sel":
        assert actual.coords["sample_index"].attrs
        assert actual.coords["tag"].attrs == before.coords["tag"].attrs
    if lazy:
        assert actual.coords["valid"].chunks is not None
    computed = actual.compute(scheduler="synchronous")
    assert dict(computed.sizes) == dict(zip(dims, shape, strict=True))
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("operation", ("at", "sel", "resample_to"))
@pytest.mark.parametrize("valid_domain", (False, True))
def test_param_query_coordinates_003_empty_static_payload_retains_domain_validation(
    operation: str, valid_domain: bool,
) -> None:
    """ID: PARAM_QUERY_COORDINATES_003_empty_static_payload_retains_domain_validation."""
    da = pytest.importorskip("dask.array")
    from dask import delayed

    param = da.from_delayed(
        delayed(np.array)([0.0, 1.0] if valid_domain else [1.0, 0.0]),
        shape=(2,), dtype=float,
    )
    source = AnalysisObject.from_data(
        xr.Dataset({"constant": xr.DataArray(3.0)}, coords={"time": ("sample", param)}),
        sequence_dim="sample", core_dims=(), param_coord="time",
    )
    query = xr.DataArray(np.empty((2, 0)), dims=("row", "col"))
    before = source.as_dataset(copy="deep")
    tasks: list[object] = []
    with _task_counter(tasks, lazy=True):
        actual = getattr(source.param, operation)(query).as_dataset(copy="none")
    assert tasks == []
    assert dict(actual.sizes) == {"row": 2, "col": 0}
    assert actual.coords["valid"].chunks is not None
    if valid_domain:
        assert actual.compute(scheduler="synchronous")["constant"].item() == 3.0
    else:
        with pytest.raises(ValueError, match="parameter coordinate must be monotonic"):
            actual.compute(scheduler="synchronous")
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("operation", ("at", "sel", "resample_to"))
def test_param_query_coordinates_005_generated_validity_wins_during_empty_restore(
    operation: str,
) -> None:
    """ID: PARAM_QUERY_COORDINATES_005_generated_validity_wins_during_empty_restore."""
    da = pytest.importorskip("dask.array")
    from dask import delayed

    prior = _source(lazy=False).param.at([0.25, 0.75]).as_dataset(copy="none")
    query = (
        prior.coords["time"]
        .rename({"sample": "row"})
        .expand_dims({"col": 0})
        .transpose("row", "col")
    )
    assert query.coords["valid"].dims == ("row",)

    param = da.from_delayed(
        delayed(np.array)([1.0, 0.0]),
        shape=(2,),
        dtype=float,
    )
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"constant": xr.DataArray(3.0)},
            coords={"time": ("sample", param)},
        ),
        sequence_dim="sample",
        core_dims=(),
        param_coord="time",
    )
    before = source.as_dataset(copy="deep")
    tasks: list[object] = []
    with _task_counter(tasks, lazy=True):
        actual = getattr(source.param, operation)(query).as_dataset(copy="none")

    assert tasks == []
    valid = actual.coords["valid"]
    assert valid.dims == ("row", "col")
    assert valid.shape == (2, 0)
    assert valid.attrs["tal_reserved_owner"] == "param_ops"
    assert valid.attrs["tal_reserved_name"] == "valid"
    assert valid.attrs["tal_reserved_token"] == "tal:param_ops:reserved:v1"
    assert valid.chunks is not None
    with pytest.raises(ValueError, match="parameter coordinate must be monotonic"):
        actual.compute(scheduler="synchronous")
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("operation", ("at", "sel", "resample_to"))
def test_param_query_coordinates_008_generated_validity_wins_nonempty_restore(
    operation: str,
) -> None:
    """ID: PARAM_QUERY_TOPOLOGY_RUNTIME_COORD_001_generated_validity_wins."""
    da = pytest.importorskip("dask.array")
    from dask import delayed

    prior = _source(lazy=False).param.at([0.25, 0.75]).as_dataset(copy="none")
    query = (
        prior.coords["time"]
        .rename({"sample": "row"})
        .expand_dims({"col": 1})
        .transpose("row", "col")
    )
    param = da.from_delayed(delayed(np.array)([1.0, 0.0]), shape=(2,), dtype=float)
    source = AnalysisObject.from_data(
        xr.Dataset({"constant": xr.DataArray(3.0)}, coords={"time": ("sample", param)}),
        sequence_dim="sample",
        core_dims=(),
        param_coord="time",
    )

    actual = getattr(source.param, operation)(query).as_dataset(copy="none")

    assert actual.coords["valid"].dims == ("row", "col")
    assert actual.coords["valid"].attrs["tal_reserved_owner"] == "param_ops"
    with pytest.raises(ValueError, match="parameter coordinate must be monotonic"):
        actual.compute(scheduler="synchronous")


@pytest.mark.parametrize("operation", ("at", "sel", "resample_to"))
@pytest.mark.parametrize("shape", ((1, 2), (2, 0)))
@pytest.mark.parametrize("lazy", (False, True))
def test_param_query_coordinates_009_caller_cannot_reclassify_output_variable(
    operation: str, shape: tuple[int, int], lazy: bool,
) -> None:
    """ID: PARAM_HARD_QUERY_NAMESPACE_001_output_variables_are_protected."""
    source = _source(lazy=lazy)
    query = xr.DataArray(
        np.full(shape, 0.25), dims=("row", "col"),
        coords={"value": (("row", "col"), np.full(shape, 99.0))},
    )
    before = source.as_dataset(copy="deep")
    query_before = query.copy(deep=True)
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy), pytest.raises(
        ValueError, match="query name 'value' collides with an output data variable"
    ):
        getattr(source.param, operation)(query)
    assert tasks == []
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)
    xr.testing.assert_identical(query, query_before)


@pytest.mark.parametrize("operation", ("at", "sel", "resample_to"))
def test_param_query_coordinates_010_caller_cannot_replace_core_index(operation: str) -> None:
    """ID: PARAM_HARD_QUERY_NAMESPACE_002_core_index_is_protected."""
    source = AnalysisObject.from_data(
        xr.DataArray(
            np.arange(6.0).reshape(3, 2), dims=("sample", "axis"),
            coords={"axis": ["x", "y"], "time": ("sample", [0.0, 1.0, 2.0])},
            name="value",
        ),
        sequence_dim="sample", core_dims=("axis",), param_coord="time",
    )
    query = xr.DataArray(
        [[0.25, 0.75]], dims=("row", "col"),
        coords={"axis": (("row", "col"), [["wrong", "labels"]])},
    )
    before = source.as_dataset(copy="deep")
    with pytest.raises(ValueError, match="query name 'axis' collides with a surviving core dimension"):
        getattr(source.param, operation)(query)
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("shape", ((1, 2), (2, 0)))
def test_param_query_coordinates_011_generated_sampled_coord_wins(
    shape: tuple[int, int],
) -> None:
    """ID: PARAM_CORE_QUERY_COORD_PRECEDENCE_001_sampled_coordinate_wins."""
    source = _source(lazy=False)
    query = xr.DataArray(
        np.full(shape, 0.25), dims=("row", "col"),
        coords={"time": (("row", "col"), np.full(shape, 99.0))},
    )
    query.coords["time"].attrs["owner"] = "caller"
    actual = source.param.sel(query).as_dataset(copy="none")
    assert actual.coords["time"].dims == ("row", "col")
    assert actual.coords["time"].shape == shape
    assert "owner" not in actual.coords["time"].attrs
    if shape[1]:
        np.testing.assert_array_equal(actual.coords["time"], [[0.0, 0.0]])
    assert "value" in actual.data_vars


@pytest.mark.parametrize("operation", ("at", "resample_to"))
@pytest.mark.parametrize("shape", ((1, 2), (2, 0)))
def test_param_query_values_own_sampled_coordinate_on_stacked_grid(
    operation: str, shape: tuple[int, int],
) -> None:
    """ID: PARAM_CORE_QUERY_OUTPUT_001_current_sampled_coordinate_wins."""
    source = _source(lazy=False)
    query = xr.DataArray(
        np.full(shape, 0.25), dims=("row", "col"),
        coords={"time": (("row", "col"), np.full(shape, 99.0))},
    )
    actual = getattr(source.param, operation)(query).as_dataset(copy="none")
    assert actual.coords["time"].dims == ("row", "col")
    assert actual.coords["time"].shape == shape
    np.testing.assert_array_equal(actual.coords["time"], np.full(shape, 0.25))


@pytest.mark.parametrize("operation", ("at", "sel", "resample_to"))
@pytest.mark.parametrize("matching", (False, True))
def test_param_query_coordinates_012_shared_coordinate_requires_identity(
    operation: str, matching: bool,
) -> None:
    """ID: PARAM_HARD_QUERY_NAMESPACE_003_shared_coordinates_are_exact."""
    base = _source(lazy=False).as_dataset(copy="none").assign_coords(station=7)
    source = AnalysisObject.from_data(
        base, sequence_dim="sample", core_dims=(), param_coord="time",
        sequence_size_coord="group_size",
    )
    query = xr.DataArray(
        [[0.25, 0.75]], dims=("row", "col"),
        coords={"station": 7 if matching else 8, "note": ("row", ["kept"])},
    )
    before = source.as_dataset(copy="deep")
    if matching:
        actual = getattr(source.param, operation)(query).as_dataset(copy="none")
        assert actual.coords["station"].item() == 7
        assert actual.coords["note"].item() == "kept"
        assert "value" in actual.data_vars
    else:
        with pytest.raises(ValueError, match="query coordinate 'station' conflicts"):
            getattr(source.param, operation)(query)
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


def test_param_query_coordinates_013_index_retains_caller_only_temp_name() -> None:
    """ID: PARAM_CORE_QUERY_COORD_PRECEDENCE_002_index_preserves_caller_only_names."""
    source = _source(lazy=False)
    query = xr.DataArray(
        [[0.25, 0.75]], dims=("row", "col"),
        coords={"__tal_query_value": ("row", ["caller"])},
    )
    actual = source.param.index(query)
    assert actual.dims == ("row", "col")
    assert actual.coords["__tal_query_value"].item() == "caller"
    np.testing.assert_array_equal(actual, [[0, 1]])


def test_param_query_coordinates_014_query_dim_cannot_reclassify_source_scalar() -> None:
    """ID: PARAM_HARD_QUERY_NAMESPACE_004_source_scalar_name_is_protected."""
    base = _source(lazy=False).as_dataset(copy="none").assign_coords(station=7)
    source = AnalysisObject.from_data(
        base, sequence_dim="sample", core_dims=(), param_coord="time",
        sequence_size_coord="group_size",
    )
    query = xr.DataArray(np.full((1, 2), 0.25), dims=("station", "col"))
    with pytest.raises(ValueError, match="query dimension 'station' collides"):
        source.param.at(query)


def test_param_query_coordinates_015_multidimensional_batch_labels_reindex() -> None:
    """ID: PARAM_CORE_QUERY_COORD_PRECEDENCE_003_batch_alignment_remains_label_based."""
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), [[0.0, 10.0], [100.0, 110.0]])},
            coords={
                "trial": ["a", "b"],
                "time": (("trial", "sample"), [[0.0, 1.0], [0.0, 1.0]]),
            },
        ),
        sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="time",
    )
    query = xr.DataArray(
        np.full((2, 1, 2), 0.5), dims=("trial", "row", "col"),
        coords={"trial": ["b", "a"], "row": [7]},
    )
    actual = source.param.at(query).as_dataset(copy="none")
    assert actual["value"].dims == ("trial", "row", "col")
    np.testing.assert_array_equal(actual.coords["trial"], ["a", "b"])
    np.testing.assert_array_equal(actual["value"], [[[5.0, 5.0]], [[105.0, 105.0]]])


def _batched_source(*, lazy: bool) -> AnalysisObject:
    ds = xr.Dataset(
        {"value": (("trial", "sample"), [[0.0, 10.0], [100.0, 110.0]])},
        coords={
            "trial": ["a", "b"],
            "time": (("trial", "sample"), [[0.0, 1.0], [0.0, 1.0]]),
        },
    )
    if lazy:
        pytest.importorskip("dask.array")
        ds = ds.chunk({"trial": 1, "sample": 2})
    return AnalysisObject.from_data(
        ds, sequence_dim="sample", batch_dims=("trial",),
        core_dims=(), param_coord="time",
    )


@pytest.mark.parametrize("operation", ("at", "sel", "resample_to"))
@pytest.mark.parametrize("query_kind", ("scalar", "one_dim", "empty", "per_batch", "batch_lane"))
@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("validate", (False, True))
def test_param_query_coordinates_016_all_labeled_shapes_protect_payload(
    operation: str, query_kind: str, lazy: bool, validate: bool,
) -> None:
    """ID: PARAM_HARD_QUERY_NAMESPACE_005_all_labeled_shapes_protect_payload."""
    source = _batched_source(lazy=lazy) if query_kind in {"per_batch", "batch_lane"} else _source(lazy=lazy)
    if query_kind == "scalar":
        query = xr.DataArray(0.5, coords={"value": 99.0})
    elif query_kind in {"one_dim", "empty"}:
        size = 0 if query_kind == "empty" else 1
        query = xr.DataArray(
            np.full(size, 0.5), dims="when",
            coords={"value": ("when", np.full(size, 99.0))},
        )
    elif query_kind == "per_batch":
        query = xr.DataArray(
            [0.5, 0.5], dims="trial",
            coords={"trial": ["a", "b"], "value": ("trial", [99.0, 99.0])},
        )
    else:
        query = xr.DataArray(
            [[0.5], [0.5]], dims=("trial", "when"),
            coords={
                "trial": ["a", "b"],
                "value": (("trial", "when"), [[99.0], [99.0]]),
            },
        )
    before = source.as_dataset(copy="deep")
    query_before = query.copy(deep=True)
    owner = "param sel" if operation == "sel" else "param at/resample"
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy), pytest.raises(
        ValueError, match=rf"^{owner}: query name 'value' collides with an output data variable",
    ):
        getattr(source.param, operation)(query, validate=validate)
    assert tasks == []
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)
    xr.testing.assert_identical(query, query_before)


@pytest.mark.parametrize("operation", ("at", "sel", "resample_to"))
@pytest.mark.parametrize("as_dimension", (False, True))
def test_param_query_coordinates_017_one_dimensional_core_index_is_protected(
    operation: str, as_dimension: bool,
) -> None:
    """ID: PARAM_HARD_QUERY_NAMESPACE_006_core_index_is_protected."""
    axis = xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(2, dim="axis"))
    ds = xr.Dataset(
        {"value": (("sample", "axis"), [[0.0, 1.0], [10.0, 11.0]])},
        coords={"time": ("sample", [0.0, 1.0])},
    ).assign_coords(axis)
    source = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), param_coord="time")
    query = (
        xr.DataArray([0.5], dims="axis") if as_dimension
        else xr.DataArray([0.5], dims="when", coords={"axis": ("when", [99])})
    )
    owner = "param sel" if operation == "sel" else "param at/resample"
    with pytest.raises(ValueError, match=rf"^{owner}: query name 'axis' collides with a surviving core dimension"):
        getattr(source.param, operation)(query)
    assert isinstance(source.as_dataset(copy="none").xindexes["axis"], xr.indexes.RangeIndex)


@pytest.mark.parametrize("operation", ("index", "at", "sel", "resample_to"))
@pytest.mark.parametrize("matching", (False, True))
def test_param_query_coordinates_018_one_dimensional_shared_coordinates_have_public_owner(
    operation: str, matching: bool,
) -> None:
    """ID: PARAM_HARD_QUERY_NAMESPACE_007_shared_coordinate_owner."""
    base = _source(lazy=False).as_dataset(copy="none").assign_coords(station=7)
    source = AnalysisObject.from_data(
        base, sequence_dim="sample", core_dims=(), param_coord="time",
        sequence_size_coord="group_size",
    )
    query = xr.DataArray(
        [0.5], dims="when",
        coords={"station": 7 if matching else 8, "note": ("when", ["kept"])},
    )
    owner = "param index" if operation == "index" else "param sel" if operation == "sel" else "param at/resample"
    if not matching:
        with pytest.raises(ValueError, match=rf"^{owner}: query coordinate 'station' conflicts"):
            getattr(source.param, operation)(query)
        return
    result = getattr(source.param, operation)(query)
    actual = result if operation == "index" else result.as_dataset(copy="none")
    assert actual.coords["station"].item() == 7
    assert actual.coords["note"].item() == "kept"
    if operation != "index":
        assert "value" in actual.data_vars


@pytest.mark.parametrize("operation", ("at", "sel", "resample_to"))
def test_param_query_coordinates_019_one_dimensional_generated_validity_wins(operation: str) -> None:
    """ID: PARAM_CORE_QUERY_COORD_PRECEDENCE_004_generated_validity_wins."""
    prior_operation = "sel" if operation == "sel" else "at"
    prior_query = [np.nan] if operation == "sel" else [-1.0]
    prior = getattr(_source(lazy=False).param, prior_operation)(prior_query).as_dataset(copy="none")
    query = prior.coords["time"].rename({"sample": "when"}).copy(data=np.array([0.5]))
    assert not bool(query.coords["valid"].all())
    if operation == "sel":
        assert query.coords["sample_index"].item() == -1
    actual = getattr(_source(lazy=False).param, operation)(query).as_dataset(copy="none")
    assert bool(actual.coords["valid"].all())
    assert actual.coords["valid"].attrs["tal_reserved_owner"] == "param_ops"
    if operation == "sel":
        assert actual.coords["sample_index"].item() == 0


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("operation", ("at", "sel", "resample_to"))
@pytest.mark.parametrize("axis_coordinate", ("absent", "independent", "multiindex"))
def test_param_query_coordinates_004_trajectory_preserves_indexed_labels(
    lazy: bool, operation: str, axis_coordinate: str,
) -> None:
    """ID: PARAM_QUERY_COORDINATES_004_trajectory_preserves_indexed_labels."""
    source = _source(lazy=lazy)
    query = xr.DataArray(
        [0.25, 0.75], dims="when",
        coords={"label": ("when", ["b", "a"]), "quality": ("when", [4.0, 5.0])},
    ).set_xindex("label")
    if axis_coordinate == "independent":
        query = query.assign_coords(when=[20, 10])
    elif axis_coordinate == "multiindex":
        query = query.drop_indexes("label").set_index(when=["label", "quality"])
    query.coords["label"].attrs["description"] = "caller labels"
    query.coords["label"].encoding["source"] = "caller"
    if lazy:
        query = query.chunk({"when": 2})
    before = source.as_dataset(copy="deep")
    query_before = query.copy(deep=True)
    tasks: list[object] = []
    with warnings.catch_warnings(), _task_counter(tasks, lazy=lazy):
        warnings.simplefilter("error")
        actual = getattr(source.param, operation)(query).as_dataset(copy="none")
    assert tasks == []
    computed = actual.compute(scheduler="synchronous")
    np.testing.assert_array_equal(computed.coords["sample"], [0, 1])
    np.testing.assert_array_equal(computed["value"], [0.0, 10.0] if operation == "sel" else [2.5, 7.5])
    for name in ("label", "quality"):
        expected = query.coords[name].rename({"when": "sample"}).variable
        xr.testing.assert_identical(
            computed.coords[name].variable.to_base_variable(), expected.compute().to_base_variable(),
        )
        assert computed.coords[name].encoding == query.coords[name].encoding
    if axis_coordinate == "multiindex":
        assert "label" not in computed.xindexes
    else:
        assert type(computed.xindexes["label"]) is type(query.xindexes["label"])
    if not lazy:
        assert computed.coords["group_size"].item() == 2
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)
    xr.testing.assert_identical(query, query_before)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("validate", (False, True))
@pytest.mark.parametrize("layout", ("trajectory", "grid", "empty_grid"))
@pytest.mark.parametrize("caller_values", ((40, 50), ("caller-a", "caller-b")))
def test_point_selection_sampled_coordinate_owns_caller_auxiliary_name(
    lazy: bool, validate: bool, layout: str, caller_values: tuple[object, object],
) -> None:
    """ID: PARAM_CORE_QUERY_OUTPUT_010_sampled_coordinate_precedes_caller_auxiliary."""
    source = _source(lazy=lazy)
    if layout == "trajectory":
        query = xr.DataArray(
            [0.25, 0.75], dims="when",
            coords={"tag": ("when", list(caller_values)), "note": ("when", [8, 9])},
        )
        expected = [4.0, 5.0]
        expected_dims = ("sample",)
    else:
        width = 0 if layout == "empty_grid" else 2
        query = xr.DataArray(
            np.full((2, width), 0.5), dims=("row", "col"),
            coords={"row": ["a", "b"], "tag": ("row", list(caller_values)),
                    "note": ("row", [8, 9])},
        )
        expected = np.full((2, width), 4.0)
        expected_dims = ("row", "col")
    if lazy:
        query = query.chunk({dim: max(size, 1) for dim, size in query.sizes.items()})
    source_before = source.as_dataset(copy="deep")
    query_before = query.copy(deep=True)
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy):
        actual = source.param.sel(query, validate=validate).as_dataset(copy="none")
    assert tasks == []
    assert actual.coords["tag"].dims == expected_dims
    assert actual.coords["tag"].attrs == source_before.coords["tag"].attrs
    np.testing.assert_array_equal(actual.compute(scheduler="synchronous").coords["tag"], expected)
    assert "note" in actual.coords
    if layout != "trajectory":
        assert actual.xindexes["row"].equals(query.xindexes["row"])
    xr.testing.assert_identical(source.as_dataset(copy="none"), source_before)
    xr.testing.assert_identical(query, query_before)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("layout", ("trajectory", "grid", "empty_grid", "aux_index"))
def test_point_selection_rejects_sampled_coordinate_axis_before_mapping(
    lazy: bool, layout: str,
) -> None:
    """ID: PARAM_HARD_QUERY_OUTPUT_006_sampled_coordinate_axis_preflight."""
    source = _source(lazy=lazy)
    if layout == "trajectory":
        query = xr.DataArray([0.25, 0.75], dims="tag", coords={"tag": ["a", "b"]})
    elif layout == "aux_index":
        query = xr.DataArray(
            [0.25, 0.75], dims="when", coords={"tag": ("when", ["a", "b"])}
        ).set_xindex("tag")
    else:
        width = 0 if layout == "empty_grid" else 2
        query = xr.DataArray(np.full((2, width), 0.5), dims=("tag", "col"))
    if lazy:
        query = query.chunk({dim: max(size, 1) for dim, size in query.sizes.items()})
    source_before = source.as_dataset(copy="deep")
    query_before = query.copy(deep=True)
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy), pytest.raises(
        ValueError,
        match="^param sel: query axis or index 'tag' conflicts with generated output metadata; rename",
    ):
        source.param.sel(query)
    assert tasks == []
    xr.testing.assert_identical(source.as_dataset(copy="none"), source_before)
    xr.testing.assert_identical(query, query_before)


@pytest.mark.parametrize("lazy", (False, True))
def test_point_selection_empty_source_keeps_sampled_coordinate_owner(lazy: bool) -> None:
    """ID: PARAM_CORE_QUERY_OUTPUT_011_empty_source_preserves_sampled_ownership."""
    ds = xr.Dataset(
        {"value": ("sample", np.empty(0))},
        coords={"time": ("sample", np.empty(0)), "tag": ("sample", np.empty(0))},
    )
    if lazy:
        ds = ds.chunk({"sample": 1})
    source = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=(), param_coord="time")
    query = xr.DataArray([0.5], dims="when", coords={"tag": ("when", ["caller"])})
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy):
        actual = source.param.sel(query).as_dataset(copy="none")
    assert tasks == []
    computed = actual.compute(scheduler="synchronous")
    assert np.isnan(computed.coords["tag"]).all()
    assert not bool(computed.coords["valid"].any())
    np.testing.assert_array_equal(computed.coords["sample_index"], [-1])


@pytest.mark.parametrize("operation", ("at", "resample_to", "index"))
def test_nonselecting_query_can_reuse_consumed_sampled_coordinate_name(operation: str) -> None:
    """ID: PARAM_CORE_QUERY_OUTPUT_013_nonselecting_dimension_reuse."""
    source = _source(lazy=False)
    query = xr.DataArray([0.25, 0.75], dims="tag", coords={"tag": ["a", "b"]})
    result = getattr(source.param, operation)(query)
    actual = result if operation == "index" else result.as_dataset(copy="none")["value"]
    expected = [0, 1] if operation == "index" else [2.5, 7.5]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("lazy", (False, True))
def test_point_selection_batch_lanes_keep_sampled_coordinate_values(lazy: bool) -> None:
    """ID: PARAM_CORE_QUERY_OUTPUT_012_batched_sampled_coordinate_precedence."""
    ds = _batched_source(lazy=lazy).as_dataset(copy="none").assign_coords(
        tag=(("trial", "sample"), [[4, 5], [6, 7]])
    )
    source = AnalysisObject.from_data(
        ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="time",
    )
    query = xr.DataArray(
        [[0.25], [0.25]], dims=("trial", "when"),
        coords={"trial": ["a", "b"], "tag": (("trial", "when"), [[40], [60]])},
    )
    if lazy:
        query = query.chunk({"trial": 1, "when": 1})
    before = source.as_dataset(copy="deep")
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy):
        actual = source.param.sel(query).as_dataset(copy="none")
    assert tasks == []
    computed = actual.compute(scheduler="synchronous")
    np.testing.assert_array_equal(computed.coords["tag"], [[4], [6]])
    assert computed.xindexes["trial"].equals(source.as_dataset(copy="none").xindexes["trial"])
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


def test_shared_dataset_gather_keeps_sampled_coordinate_owner() -> None:
    """Focused unit check for shared gather coordinate precedence."""
    from tal.core.param_engine.map_apply import gather_dataset_along_sequence

    source = _source(lazy=False).as_dataset(copy="none")
    indexer = xr.DataArray(
        [0, 1], dims="query", coords={"tag": ("query", [40, 50])},
    )
    actual = gather_dataset_along_sequence(
        source, indexer, sequence_dim="sample", query_dim="query", owner="gather probe",
    )
    np.testing.assert_array_equal(actual.coords["tag"], [4, 5])
    np.testing.assert_array_equal(actual["value"], [0.0, 10.0])


@pytest.mark.parametrize("bound_side", ("start", "stop"))
@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("validate", (False, True))
def test_slice_bound_cannot_reclassify_payload(
    bound_side: str, lazy: bool, validate: bool,
) -> None:
    """ID: PARAM_HARD_SLICE_BOUND_OUTPUT_001_protected_payload_preflight."""
    source = _source(lazy=lazy)
    bound = xr.DataArray(0.0 if bound_side == "start" else 2.0, coords={"value": 999.0})
    query = slice(bound, 2.0) if bound_side == "start" else slice(0.0, bound)
    source_before = source.as_dataset(copy="deep")
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy), pytest.raises(
        ValueError, match="^param sel: query name 'value' collides with an output data variable",
    ):
        source.param.sel(query, validate=validate)
    assert tasks == []
    xr.testing.assert_identical(source.as_dataset(copy="none"), source_before)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("validate", (False, True))
def test_slice_bound_metadata_uses_output_ownership(lazy: bool, validate: bool) -> None:
    """ID: PARAM_CORE_SLICE_BOUND_OUTPUT_001_generated_and_caller_coordinates."""
    source = _source(lazy=lazy)
    start = xr.DataArray(0.0, coords={"tag": 900, "valid": False, "note": 7})
    stop = xr.DataArray(2.0, coords={"tag": 900, "valid": False, "note": 7})
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy):
        actual = source.param.sel(slice(start, stop), validate=validate).as_dataset(copy="none")
    assert tasks == []
    assert "value" in actual.data_vars and "value" not in actual.coords
    computed = actual.compute(scheduler="synchronous")
    np.testing.assert_array_equal(computed.coords["tag"], [4, 5, 6])
    assert bool(computed.coords["valid"].all())
    assert int(computed.coords["note"]) == 7
    assert int(computed.coords["group_size"]) == 3


@pytest.mark.parametrize("conflict", ("source", "bounds"))
def test_slice_bound_shared_coordinate_conflicts_are_owned(conflict: str) -> None:
    """ID: PARAM_HARD_SLICE_BOUND_OUTPUT_002_shared_coordinate_compatibility."""
    ds = _source(lazy=False).as_dataset(copy="none").assign_coords(station=7)
    source = AnalysisObject.from_data(
        ds, sequence_dim="sample", core_dims=(), param_coord="time",
        sequence_size_coord="group_size",
    )
    start = xr.DataArray(0.0, coords={"station": 8 if conflict == "source" else 7, "note": 1})
    stop = xr.DataArray(2.0, coords={"station": 7, "note": 2 if conflict == "bounds" else 1})
    with pytest.raises(ValueError, match="^param sel: .*coordinate '(station|note)'.*conflict"):
        source.param.sel(slice(start, stop))


@pytest.mark.parametrize("operation", ("at", "sel", "resample_to"))
@pytest.mark.parametrize("lazy", (False, True))
def test_query_lane_named_like_source_sequence_is_consumed(operation: str, lazy: bool) -> None:
    """ID: PARAM_CORE_QUERY_LANE_001_source_sequence_name_is_valid_query_lane."""
    source = _source(lazy=lazy)
    query = source.as_dataset(copy="none").coords["time"].assign_coords(sample=[10, 20, 30])
    before = source.as_dataset(copy="deep")
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy):
        actual = getattr(source.param, operation)(query).as_dataset(copy="none")
    if lazy:
        assert actual["value"].chunks is not None
    else:
        assert tasks == []
    assert actual["value"].dims == ("sample",)
    np.testing.assert_array_equal(actual.coords["sample"], [0, 1, 2])
    computed = actual.compute(scheduler="synchronous")
    np.testing.assert_array_equal(computed["value"], [0.0, 10.0, 20.0])
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


def test_query_auxiliary_cannot_claim_output_sequence_axis() -> None:
    """ID: PARAM_HARD_QUERY_LANE_001_nonaxis_sequence_coordinate_rejected."""
    query = xr.DataArray(
        [0.25, 0.75], dims="when", coords={"sample": ("when", [10, 20])},
    )
    with pytest.raises(ValueError, match="^param sel: query coordinate 'sample' collides with the positional output sequence"):
        _source(lazy=False).param.sel(query)


@pytest.mark.parametrize("operation", ("index", "at", "sel", "resample_to"))
@pytest.mark.parametrize("shape", ((2, 2), (2, 0)))
@pytest.mark.parametrize("lazy", (False, True))
def test_param_query_coordinates_006_public_query_dim_name_survives_stacking(
    operation: str,
    shape: tuple[int, int],
    lazy: bool,
) -> None:
    """ID: PARAM_QUERY_TOPOLOGY_PUBLIC_DIM_001_public_name_survives_stacking."""
    source = _source(lazy=lazy)
    query = xr.DataArray(
        np.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape),
        dims=("query", "col"),
        coords=xr.Coordinates.from_xindex(
            xr.indexes.RangeIndex.arange(shape[0], dim="query")
        ),
    ).assign_coords(
        xr.Coordinates.from_xindex(
            xr.indexes.RangeIndex.arange(shape[1], dim="col")
        )
    )
    if lazy:
        query = query.chunk({"query": 2, "col": max(shape[1], 1)})
    before = query.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = getattr(source.param, operation)(query)
        actual = result if operation == "index" else result.as_dataset(copy="none")

    assert tuple(actual.dims) == ("query", "col")
    assert tuple(actual.sizes.values()) == shape
    for dim in query.dims:
        assert type(actual.xindexes[dim]) is type(query.xindexes[dim])
        assert actual.xindexes[dim].equals(query.xindexes[dim])
    xr.testing.assert_identical(query, before)


@pytest.mark.parametrize("operation", ("index", "at", "sel", "resample_to"))
@pytest.mark.parametrize("shape", ((2, 2), (2, 0)))
def test_param_query_coordinates_007_native_indexes_are_not_materialized(
    operation: str,
    shape: tuple[int, int],
) -> None:
    """ID: PARAM_QUERY_TOPOLOGY_NATIVE_INDEX_001_native_indexes_stay_lazy."""
    calls: list[str] = []
    transform = xr.indexes.CoordinateTransformIndex(
        _UnevaluatedTopologyTransform(2, calls, dim="row")
    )
    query = xr.DataArray(
        np.linspace(0.25, 1.75, int(np.prod(shape))).reshape(shape),
        dims=("row", "col"),
        coords=xr.Coordinates.from_xindex(transform),
    ).assign_coords(
        xr.Coordinates.from_xindex(
            xr.indexes.RangeIndex.arange(shape[1], dim="col")
        )
    )
    source = _source(lazy=True)
    before = source.as_dataset(copy="deep")

    result = getattr(source.param, operation)(query)
    actual = result if operation == "index" else result.as_dataset(copy="none")

    assert calls == []
    assert type(actual.xindexes["row"]) is type(query.xindexes["row"])
    assert actual.xindexes["row"].equals(query.xindexes["row"])
    assert isinstance(actual.xindexes["col"], xr.indexes.RangeIndex)
    computed = actual.compute(scheduler="synchronous")
    assert dict(computed.sizes) == {"row": 2, "col": shape[1]}
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


def test_param_index_consumes_prior_tal_runtime_coordinates() -> None:
    """ID: PARAM_CORE_QUERY_OUTPUT_002_index_consumes_inherited_runtime_metadata."""
    prior = _source(lazy=False)
    previous = prior.param.at([3.0]).as_dataset(copy="none")
    query = previous.coords["time"]
    assert not bool(query.coords["valid"].item())
    assert int(query.coords["group_size"].item()) == 0
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("sample", [0.0, 1.0, 2.0, 3.0])},
            coords={"time": ("sample", [0.0, 1.0, 2.0, 3.0]), "group_size": 4},
        ),
        sequence_dim="sample", core_dims=(), param_coord="time",
        sequence_size_coord="group_size",
    )
    actual = source.param.index(query)
    np.testing.assert_array_equal(actual, [3])
    assert "valid" not in actual.coords
    assert "sample_index" not in actual.coords
    assert "group_size" not in actual.coords
    np.testing.assert_array_equal(actual.coords["time"], [3.0])


def test_param_index_empty_chained_query_retains_current_validation() -> None:
    """ID: PARAM_HARD_QUERY_OUTPUT_001_empty_index_retains_deferred_validation."""
    da = pytest.importorskip("dask.array")
    from dask import delayed

    param = da.from_delayed(delayed(np.array)([1.0, 0.0]), shape=(2,), dtype=float)
    source = AnalysisObject.from_data(
        xr.Dataset({"constant": xr.DataArray(3.0)}, coords={"time": ("sample", param)}),
        sequence_dim="sample", core_dims=(), param_coord="time",
    )
    prior = _source(lazy=False)
    previous = prior.param.at(xr.DataArray(np.empty((2, 0)), dims=("row", "col")))
    query = previous.as_dataset(copy="none").coords["time"]
    tasks: list[object] = []
    with _task_counter(tasks, lazy=True):
        actual = source.param.index(query)
    assert tasks == []
    assert dict(actual.sizes) == {"row": 2, "col": 0}
    assert "valid" not in actual.coords
    with pytest.raises(ValueError, match="parameter coordinate must be monotonic"):
        actual.compute(scheduler="synchronous")


def test_param_index_rejects_unowned_reserved_query_coordinate() -> None:
    """ID: PARAM_HARD_QUERY_OUTPUT_002_unowned_reserved_name_is_rejected."""
    query = xr.DataArray([0.5], dims="when", coords={"valid": ("when", [False])})
    with pytest.raises(ValueError, match="^param index: query coordinate 'valid' is reserved"):
        _source(lazy=False).param.index(query)


@pytest.mark.parametrize("operation,name", (
    ("at", "valid"), ("at", "time"),
    ("resample_to", "valid"), ("resample_to", "time"),
    ("sel", "valid"), ("sel", "time"), ("sel", "sample_index"),
))
@pytest.mark.parametrize("width", (0, 2))
@pytest.mark.parametrize("lazy", (False, True))
def test_query_axis_cannot_erase_generated_coordinate(
    operation: str, name: str, width: int, lazy: bool,
) -> None:
    """ID: PARAM_HARD_QUERY_OUTPUT_003_generated_axis_conflict_preflight."""
    source = _source(lazy=lazy)
    query = xr.DataArray(np.full((2, width), 0.5), dims=(name, "col"))
    before = source.as_dataset(copy="none").copy(deep=True)
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy), pytest.raises(
        ValueError, match=rf"^param .*: query axis or index '{name}'",
    ):
        getattr(source.param, operation)(query)
    assert tasks == []
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("operation", ("at", "resample_to"))
def test_consumed_source_only_dimension_does_not_reserve_query_name(operation: str) -> None:
    """ID: PARAM_CORE_QUERY_OUTPUT_003_projected_source_dimension_ownership."""
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("sample", [0.0, 1.0, 2.0])},
            coords={
                "time": ("sample", [0.0, 1.0, 2.0]),
                "aux": (("sample", "ghost"), np.ones((3, 2))),
            },
        ),
        sequence_dim="sample", core_dims=(), param_coord="time",
    )
    query = xr.DataArray([0.5, 1.5], dims=("ghost",), coords={"ghost": ["first", "second"]})
    actual = getattr(source.param, operation)(query).as_dataset(copy="none")
    assert actual.sizes["sample"] == 2
    assert "aux" not in actual.coords


def test_selected_source_aux_dimension_remains_protected() -> None:
    """ID: PARAM_HARD_QUERY_OUTPUT_004_selection_protects_sampled_aux_dims."""
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("sample", [0.0, 1.0])},
            coords={
                "time": ("sample", [0.0, 1.0]),
                "aux": (("sample", "ghost"), np.ones((2, 2))),
            },
        ), sequence_dim="sample", core_dims=(), param_coord="time",
    )
    query = xr.DataArray([0.25, 0.75], dims="ghost")
    with pytest.raises(ValueError, match="^param sel: query name 'ghost' collides"):
        source.param.sel(query)


def test_index_retains_caller_axis_named_like_consumed_size_metadata() -> None:
    """ID: PARAM_CORE_QUERY_OUTPUT_004_same_name_caller_index_survives."""
    source = _source(lazy=False)
    query = xr.DataArray(
        [0.25, 1.25], dims=("group_size",),
        coords={"group_size": ["first", "second"]},
    )
    actual = source.param.index(query)
    np.testing.assert_array_equal(actual, [0, 1])
    assert actual.dims == ("query",)
    np.testing.assert_array_equal(actual.coords["query"], ["first", "second"])
    assert type(actual.xindexes["query"]) is type(query.xindexes["group_size"])


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("shape", ((2,), (2, 0)))
def test_index_retains_caller_auxiliary_named_like_source_size(
    lazy: bool, shape: tuple[int, ...],
) -> None:
    """ID: PARAM_CORE_QUERY_OUTPUT_005_caller_size_name_auxiliary_survives."""
    source = _source(lazy=lazy)
    dims = ("when",) if len(shape) == 1 else ("row", "col")
    query = xr.DataArray(
        np.full(shape, 0.5), dims=dims,
        coords={"group_size": (dims[0], [7, 8])},
    )
    if lazy:
        query = query.chunk({dim: max(size, 1) for dim, size in zip(dims, shape, strict=True)})
    before = source.as_dataset(copy="deep")
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy):
        actual = source.param.index(query)
    assert tasks == []
    expected_dim = "query" if len(shape) == 1 else "row"
    assert actual.coords["group_size"].dims == (expected_dim,)
    np.testing.assert_array_equal(actual.coords["group_size"], [7, 8])
    assert actual.shape == shape
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("operation", ("index", "at", "sel", "resample_to"))
@pytest.mark.parametrize("lazy", (False, True))
def test_shared_batch_index_type_conflict_has_public_owner(
    operation: str, lazy: bool,
) -> None:
    """ID: PARAM_HARD_QUERY_OUTPUT_005_shared_native_index_type_preflight."""
    axis = xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(2, dim="trial"))
    ds = xr.Dataset(
        {"value": (("trial", "sample"), [[0.0, 1.0], [2.0, 3.0]])},
        coords={"time": ("sample", [0.0, 1.0])},
    ).assign_coords(axis)
    if lazy:
        pytest.importorskip("dask.array")
        ds = ds.chunk({"trial": 1, "sample": 2})
    source = AnalysisObject.from_data(
        ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="time",
    )
    query = xr.DataArray(
        [[0.5], [0.5]], dims=("trial", "when"), coords={"trial": [0, 1]},
    )
    before = source.as_dataset(copy="deep")
    owner = "param index" if operation == "index" else "param sel" if operation == "sel" else "param at/resample"
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy), pytest.raises(
        ValueError, match=rf"^{owner}: shared batch index along 'trial' has incompatible xarray index topology",
    ):
        getattr(source.param, operation)(query)
    assert tasks == []
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("operation", ("index", "at", "sel", "resample_to"))
def test_matching_native_batch_index_remains_supported(operation: str) -> None:
    """ID: PARAM_CORE_QUERY_OUTPUT_006_matching_native_batch_index_survives."""
    axis = xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(2, dim="trial"))
    ds = xr.Dataset(
        {"value": (("trial", "sample"), [[0.0, 1.0], [2.0, 3.0]])},
        coords={"time": ("sample", [0.0, 1.0])},
    ).assign_coords(axis)
    source = AnalysisObject.from_data(
        ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="time",
    )
    query = xr.DataArray([[0.5], [0.5]], dims=("trial", "when")).assign_coords(axis)
    result = getattr(source.param, operation)(query)
    actual = result if operation == "index" else result.as_dataset(copy="none")
    assert isinstance(actual.xindexes["trial"], xr.indexes.RangeIndex)
    assert actual.xindexes["trial"].equals(source.as_dataset(copy="none").xindexes["trial"])


@pytest.mark.parametrize("operation", ("index", "at", "sel", "resample_to"))
@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("query_lane", ("when", "query"))
def test_reordered_native_batch_labels_are_aligned_before_evaluation(
    operation: str, lazy: bool, query_lane: str,
) -> None:
    """ID: PARAM_CORE_NATIVE_BATCH_REINDEX_001_label_order_and_index_survive."""
    source_axis = xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(2, dim="trial"))
    query_axis = xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(1, -1, -1, dim="trial"))
    ds = xr.Dataset(
        {"value": (("trial", "sample"), [[0.0, 1.0], [2.0, 3.0]])},
        coords={"time": ("sample", [0.0, 1.0])},
    ).assign_coords(source_axis)
    if lazy:
        pytest.importorskip("dask.array")
        ds = ds.chunk({"trial": 1, "sample": 2})
    source = AnalysisObject.from_data(
        ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="time",
    )
    query = xr.DataArray([[1.0], [0.0]], dims=("trial", query_lane), coords=query_axis)
    before = source.as_dataset(copy="deep")
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy):
        result = getattr(source.param, operation)(query)
    assert tasks == []
    actual = result if operation == "index" else result.as_dataset(copy="none")
    assert isinstance(actual.xindexes["trial"], xr.indexes.RangeIndex)
    assert actual.xindexes["trial"].equals(source.as_dataset(copy="none").xindexes["trial"])
    eager = actual.compute(scheduler="synchronous")
    values = eager if operation == "index" else eager["value"]
    np.testing.assert_array_equal(values, [[0], [1]] if operation == "index" else [[0.0], [3.0]])
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("operation", ("index", "at", "sel", "resample_to"))
def test_unequal_transform_batch_indexes_fail_under_public_owner(operation: str) -> None:
    """ID: PARAM_HARD_NATIVE_BATCH_REINDEX_001_unequal_transform_preflight."""
    calls: list[str] = []
    source_axis = xr.Coordinates.from_xindex(xr.indexes.CoordinateTransformIndex(
        _TaggedTopologyTransform(2, calls, dim="trial", tag="source"),
    ))
    query_axis = xr.Coordinates.from_xindex(xr.indexes.CoordinateTransformIndex(
        _TaggedTopologyTransform(2, calls, dim="trial", tag="query"),
    ))
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), [[0.0, 1.0], [2.0, 3.0]])},
            coords={"time": ("sample", [0.0, 1.0])},
        ).assign_coords(source_axis),
        sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="time",
    )
    query = xr.DataArray([[0.5], [0.5]], dims=("trial", "when"), coords=query_axis)
    owner = "param index" if operation == "index" else "param sel" if operation == "sel" else "param at/resample"
    with pytest.raises(ValueError, match=rf"^{owner}: shared batch index along 'trial' has incompatible"):
        getattr(source.param, operation)(query)
    assert calls == []


@pytest.mark.parametrize("operation", ("index", "at", "sel", "resample_to"))
def test_partial_native_batch_overlap_preserves_target_labels(operation: str) -> None:
    """ID: PARAM_CORE_NATIVE_BATCH_REINDEX_002_missing_query_label_is_invalid."""
    source_axis = xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(2, dim="trial"))
    query_axis = xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(0, 4, 2, dim="trial"))
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), [[0.0, 1.0], [2.0, 3.0]])},
            coords={"time": ("sample", [0.0, 1.0])},
        ).assign_coords(source_axis),
        sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="time",
    )
    query = xr.DataArray([[0.0], [1.0]], dims=("trial", "when"), coords=query_axis)
    result = getattr(source.param, operation)(query)
    actual = result if operation == "index" else result.as_dataset(copy="none")
    assert isinstance(actual.xindexes["trial"], xr.indexes.RangeIndex)
    assert actual.xindexes["trial"].equals(source.as_dataset(copy="none").xindexes["trial"])
    values = actual if operation == "index" else actual["value"]
    np.testing.assert_array_equal(values.isel(trial=0), [0.0])
    if operation == "index":
        np.testing.assert_array_equal(values.isel(trial=1), [-1])
    else:
        assert bool(values.isel(trial=1).isnull().all())


@pytest.mark.parametrize("lazy", (False, True))
def test_index_consumes_prior_generated_size_with_different_batch_topology(lazy: bool) -> None:
    """ID: PARAM_CORE_GENERATED_SIZE_PROVENANCE_001_cross_source_query."""
    prior = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), [[0.0, 1.0], [0.0, 1.0]])},
            coords={"time": ("sample", [0.0, 1.0]), "group_size": ("trial", [2, 2])},
        ),
        sequence_dim="sample", batch_dims=("trial",), core_dims=(),
        param_coord="time", sequence_size_coord="group_size",
    )
    query = prior.param.at([0.25, 0.75]).as_dataset(copy="none")["value"]
    assert query.coords["group_size"].dims == ("trial",)
    source = _source(lazy=lazy)
    before = source.as_dataset(copy="deep")
    actual = source.param.index(query)
    assert "group_size" not in actual.coords
    assert "valid" not in actual.coords
    assert actual.sizes == {"trial": 2, "sample": 2}
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("query_size", (0, 2))
def test_index_consumes_generated_size_from_differently_named_source(
    lazy: bool, query_size: int,
) -> None:
    """ID: PARAM_CORE_GENERATED_SIZE_PROVENANCE_002_cross_name_query."""
    prior_ds = xr.Dataset(
        {"value": (("trial", "sample"), [[0.0, 1.0], [0.0, 1.0]])},
        coords={"time": ("sample", [0.0, 1.0]), "prior_count": ("trial", [2, 2])},
    )
    if lazy:
        prior_ds["value"] = prior_ds["value"].chunk({"trial": 1, "sample": 2})
    prior = AnalysisObject.from_data(
        prior_ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(),
        param_coord="time", sequence_size_coord="prior_count",
    )
    query = prior.param.at(np.linspace(0.25, 0.75, query_size)).as_dataset(copy="none")["value"]
    assert query.coords["prior_count"].attrs["tal_reserved_owner"] == "param_ops"
    source = _source(lazy=lazy)
    before = source.as_dataset(copy="deep")
    query_before = query.copy(deep=True)
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy):
        actual = source.param.index(query)
    assert tasks == []
    assert actual.sizes == {"trial": 2, "sample": query_size}
    assert "prior_count" not in actual.coords
    assert "valid" not in actual.coords
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)
    xr.testing.assert_identical(query, query_before)


def test_index_preserves_unowned_auxiliary_from_differently_named_source() -> None:
    """ID: PARAM_CORE_GENERATED_SIZE_PROVENANCE_003_unowned_auxiliary_survives."""
    source = _source(lazy=False)
    query = xr.DataArray(
        [[0.25, 0.75], [0.25, 0.75]], dims=("trial", "when"),
        coords={"prior_count": ("trial", [2, 2])},
    )
    actual = source.param.index(query)
    xr.testing.assert_identical(actual.coords["prior_count"], query.coords["prior_count"])


def test_cross_source_empty_lazy_index_retains_current_domain_validation() -> None:
    """ID: PARAM_HARD_GENERATED_SIZE_PROVENANCE_001_empty_query_validation."""
    da = pytest.importorskip("dask.array")
    prior = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), [[0.0, 1.0], [0.0, 1.0]])},
            coords={"time": ("sample", [0.0, 1.0]), "prior_count": ("trial", [2, 2])},
        ),
        sequence_dim="sample", batch_dims=("trial",), core_dims=(),
        param_coord="time", sequence_size_coord="prior_count",
    )
    query = prior.param.at(np.empty(0)).as_dataset(copy="none")["value"].chunk({"trial": 1, "sample": 1})
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("sample", [0.0, 1.0, 2.0])},
            coords={"time": ("sample", da.from_array([0.0, 2.0, 1.0], chunks=3))},
        ),
        sequence_dim="sample", core_dims=(), param_coord="time",
    )
    tasks: list[object] = []
    with _task_counter(tasks, lazy=True):
        actual = source.param.index(query)
    assert tasks == []
    assert actual.sizes == {"trial": 2, "sample": 0}
    assert "prior_count" not in actual.coords
    with pytest.raises(ValueError, match="monotonic non-decreasing"):
        actual.compute(scheduler="synchronous")


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("index_kind", ("range", "transform"))
def test_index_preserves_renamed_native_query_axis(lazy: bool, index_kind: str) -> None:
    """ID: PARAM_CORE_QUERY_INDEX_001_native_query_axis_survives_normalization."""
    source = _source(lazy=lazy)
    calls: list[str] = []
    native_index = (
        xr.indexes.RangeIndex.arange(10, 14, 2, dim="when")
        if index_kind == "range"
        else xr.indexes.CoordinateTransformIndex(_UnevaluatedTopologyTransform(2, calls, dim="when"))
    )
    query = xr.DataArray(
        [0.25, 0.75], dims="when",
        coords=xr.Coordinates.from_xindex(native_index),
    )
    before = source.as_dataset(copy="deep")
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy):
        actual = source.param.index(query)
    assert tasks == []
    assert actual.dims == ("query",)
    assert type(actual.xindexes["query"]) is type(native_index)
    assert actual.xindexes["query"].equals(query.rename({"when": "query"}).xindexes["query"])
    assert calls == []
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("query", (0.5, [0.5], xr.DataArray([[0.5, 1.0]], dims=("row", "col"))))
@pytest.mark.parametrize("datetime", (False, True))
def test_point_selection_zero_sample_source_is_all_invalid(query: object, lazy: bool, datetime: bool) -> None:
    """ID: PARAM_CORE_ZERO_SAMPLE_POINT_001_no_fabricated_source_row."""
    values = np.empty((0,), dtype="float64")
    domain = values.astype("datetime64[ns]") if datetime else values
    ds = xr.Dataset({"value": ("sample", values)}, coords={"time": ("sample", domain)})
    if lazy:
        pytest.importorskip("dask.array")
        ds = ds.chunk({"sample": 1})
    source = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=(), param_coord="time")
    if datetime:
        convert = lambda value: np.datetime64("2024-01-01") if np.isscalar(value) else value
        if isinstance(query, xr.DataArray):
            query = xr.full_like(query, np.datetime64("2024-01-01"), dtype="datetime64[ns]")
        elif isinstance(query, list):
            query = [convert(value) for value in query]
        else:
            query = convert(query)
    before = source.as_dataset(copy="deep")
    tasks: list[object] = []
    with _task_counter(tasks, lazy=lazy):
        actual = source.param.sel(query).as_dataset(copy="none")
    assert tasks == []
    eager = actual.compute(scheduler="synchronous")
    if not np.isscalar(query):
        assert not bool(eager.coords["valid"].any())
    assert bool(eager["value"].isnull().all())
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


def test_zero_sample_point_selection_does_not_execute_source_graph() -> None:
    """ID: PARAM_HARD_ZERO_SAMPLE_POINT_001_source_tasks_are_unreachable."""
    da = pytest.importorskip("dask.array")
    from dask import delayed

    def fail_if_called() -> np.ndarray:
        raise AssertionError("zero-sample source task must not execute")

    payload = da.from_delayed(delayed(fail_if_called)(), shape=(0,), dtype=float)
    domain = da.from_delayed(delayed(fail_if_called)(), shape=(0,), dtype=float)
    source = AnalysisObject.from_data(
        xr.Dataset({"value": ("sample", payload)}, coords={"time": ("sample", domain)}),
        sequence_dim="sample", core_dims=(), param_coord="time",
    )
    result = source.param.sel([0.5]).as_dataset(copy="none")
    assert bool(result.compute(scheduler="synchronous")["value"].isnull().all())
