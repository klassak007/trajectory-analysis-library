from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from tal.core import (
    AnalysisObject,
    BatchGroupedView,
    BatchGroupReduceOptions,
    GroupByOptions,
    GroupMaterializeOptions,
)
from tal.core.group_ops import GroupingFoundationOptions


def _batch_ao() -> AnalysisObject:
    values = np.arange(16, dtype=float).reshape(4, 2, 2)
    ds = xr.Dataset(
        {
            "value": (("trial", "lane", "axis"), values),
            "static": (("lane", "axis"), [[10.0, 20.0], [30.0, 40.0]]),
        },
        coords={
            "trial": ["t0", "t1", "t2", "t3"],
            "lane": ["left", "right"],
            "axis": ["x", "y"],
            "outcome": ("trial", ["win", "loss", "win", "loss"]),
            "phase": ("trial", [1, 1, 2, 2]),
            "lane_key": ("lane", ["a", "b"]),
            "axis_key": ("axis", ["x", "y"]),
        },
    )
    return AnalysisObject.from_data(
        ds,
        batch_dims=("trial", "lane"),
        core_dims=("axis",),
        validate=True,
    )


def _batch_roles(ao: AnalysisObject) -> dict[str, object]:
    return ao.as_dataset(copy="none").attrs["tal"]["core"]["roles"]


def _roleless_grouping_ao() -> AnalysisObject:
    return AnalysisObject.from_data(
        xr.Dataset({"value": ("axis", [1.0, 2.0])}, coords={"axis": ["x", "y"]}),
        core_dims=("axis",),
        validate=True,
    )


def test_batch_grouping_after_sequence_reduction_uses_public_entrypoint() -> None:
    """ID: BATCH_GROUP_001_post_sequence_reduction_uses_public_entrypoint."""
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"distance": (("trial", "sample"), [[4.0, 2.0], [3.0, 1.0], [8.0, 6.0]])},
            coords={
                "trial": ["a", "b", "c"],
                "sample": [0, 1],
                "outcome": ("trial", ["win", "loss", "win"]),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )

    reduced = source.min(dim="sample")
    grouped = reduced.group.groupby("outcome")
    out = grouped.mean(dim="trial")

    assert isinstance(grouped, BatchGroupedView)
    xr.testing.assert_allclose(
        out.as_dataset(copy="none")["distance"],
        xr.DataArray([4.0, 1.0], dims=("group_key",), coords={"group_key": ["win", "loss"]}),
    )
    assert _batch_roles(out)["batch_dims"] == ["group_key"]


def test_batch_grouping_preserves_supplemental_core_and_static_topology() -> None:
    """ID: BATCH_GROUP_002_preserves_supplemental_and_static_topology."""
    source = _batch_ao()
    out = source.group.groupby("outcome").mean(dim="trial")
    actual = out.as_dataset(copy="none")
    source_value = source.as_dataset(copy="none")["value"]
    expected = xr.concat(
        (
            source_value.isel(trial=[0, 2]).mean("trial"),
            source_value.isel(trial=[1, 3]).mean("trial"),
        ),
        dim="group_key",
    ).assign_coords(group_key=["win", "loss"])

    np.testing.assert_allclose(actual["value"].data, expected.data)
    xr.testing.assert_identical(actual["static"], source.as_dataset(copy="none")["static"])
    assert tuple(actual["value"].dims) == ("group_key", "lane", "axis")
    assert _batch_roles(out) == {
        "batch_dims": ["group_key", "lane"],
        "core_dims": ["axis"],
    }


def test_batch_grouping_retains_group_when_payload_is_primary_independent() -> None:
    """ID: BATCH_GROUP_003_primary_independent_payload_retains_group."""
    source = _batch_ao().drop_vars("value")
    out = source.group.groupby("outcome").mean(dim="trial")
    actual = out.as_dataset(copy="none")

    assert actual.coords["group_key"].values.tolist() == ["win", "loss"]
    assert tuple(actual["static"].dims) == ("lane", "axis")
    assert _batch_roles(out)["batch_dims"] == ["group_key", "lane"]


def test_batch_grouping_requires_exact_index_or_two_unindexed_lanes() -> None:
    """ID: BATCH_GROUP_004_key_alignment_is_exact_or_positional_unindexed."""
    source = _batch_ao()
    equal = xr.DataArray(
        ["a", "b", "a", "b"],
        dims=("trial",),
        coords={"trial": ["t0", "t1", "t2", "t3"]},
    )
    assert source.group.groupby(equal).mean().as_dataset(copy="none").sizes["group_key"] == 2

    unequal = equal.assign_coords(trial=["t1", "t0", "t2", "t3"])
    with pytest.raises(ValueError, match="index must exactly match"):
        source.group.groupby(unequal)
    with pytest.raises(ValueError, match="cannot mix indexed and unindexed"):
        source.group.groupby(xr.DataArray(["a", "b", "a", "b"], dims=("trial",)))

    unindexed_ds = source.as_dataset().drop_indexes("trial").drop_vars("trial")
    unindexed = AnalysisObject.from_data(
        unindexed_ds,
        batch_dims=("trial", "lane"),
        core_dims=("axis",),
        validate=True,
    )
    out = unindexed.group.groupby(
        xr.DataArray(["a", "b", "a", "b"], dims=("trial",))
    ).sum()
    assert out.as_dataset(copy="none").sizes["group_key"] == 2


@pytest.mark.parametrize("key", ["lane_key", "axis_key"])
def test_batch_grouping_rejects_supplemental_and_core_keys(key: str) -> None:
    """ID: BATCH_GROUP_005_rejects_supplemental_and_core_keys."""
    with pytest.raises(ValueError, match="must vary over exactly"):
        _batch_ao().group.groupby(key)


def test_batch_grouping_rejects_scalar_key() -> None:
    """ID: BATCH_GROUP_006_rejects_scalar_keys."""
    with pytest.raises(ValueError, match="must vary over exactly"):
        _batch_ao().group.groupby(xr.DataArray("same"))


def test_batch_grouping_tuple_keys_preserve_first_appearance_order() -> None:
    """ID: BATCH_GROUP_007_tuple_keys_preserve_first_appearance_order."""
    out = _batch_ao().group.groupby(("outcome", "phase")).sum(dim="trial")
    labels = out.as_dataset(copy="none").coords["group_key"].values.tolist()
    assert labels == [("win", 1), ("loss", 1), ("win", 2), ("loss", 2)]


def test_batch_grouping_bins_and_empty_group_options() -> None:
    """ID: BATCH_GROUP_008_bins_and_empty_group_domains."""
    source = _batch_ao()
    grouped = source.group.groupby_bins(
        xr.DataArray(
            [0.2, 0.4, 1.2, 1.4],
            dims=("trial",),
            coords={"trial": ["t0", "t1", "t2", "t3"]},
        ),
        bins=[0.0, 1.0, 2.0, 3.0],
        labels=["low", "mid", "high"],
    )

    present = grouped.mean().as_dataset(copy="none")
    assert present.coords["group_key"].values.tolist() == ["low", "mid"]
    complete = grouped.mean(
        opts=BatchGroupReduceOptions(include_empty_groups=True)
    ).as_dataset(copy="none")
    assert complete.coords["group_key"].values.tolist() == ["low", "mid", "high"]
    assert bool(complete["value"].sel(group_key="high").isnull().all())
    counted = grouped.count(
        opts=BatchGroupReduceOptions(include_empty_groups=True)
    ).as_dataset(copy="none")
    assert bool((counted["value"].sel(group_key="high") == 0).all())


@pytest.mark.parametrize("policy", ["error", "drop", "group"])
def test_batch_grouping_na_policies(policy: str) -> None:
    """ID: BATCH_GROUP_009_na_policies_are_topology_neutral."""
    source = _batch_ao()
    key = xr.DataArray(
        ["a", None, "a", "b"],
        dims=("trial",),
        coords={"trial": ["t0", "t1", "t2", "t3"]},
    )
    grouped = source.group.groupby(
        key,
        opts=GroupByOptions(
            foundation_opts=GroupingFoundationOptions(na_key_policy=policy),
        ),
    )
    if policy == "error":
        with pytest.raises(ValueError, match="na_key_policy='error'"):
            grouped.mean()
        return
    labels = grouped.mean().as_dataset(copy="none").coords["group_key"].values.tolist()
    expected = ["a", "b"] if policy == "drop" else ["a", "__tal_na_group__", "b"]
    assert labels == expected
    if policy == "group":
        tuple_label = ("missing", 0)
        tuple_grouped = source.group.groupby(
            key,
            opts=GroupByOptions(
                foundation_opts=GroupingFoundationOptions(
                    na_key_policy="group",
                    na_group_label=tuple_label,
                ),
            ),
        ).mean()
        assert tuple_label in tuple_grouped.as_dataset(copy="none").coords[
            "group_key"
        ].values.tolist()


def test_batch_grouping_reducer_dims_weights_and_boolean_semantics() -> None:
    """ID: BATCH_GROUP_010_reducers_reuse_ordinary_semantics."""
    source = _batch_ao()
    weighted = source.group.groupby("outcome").mean(
        dim="trial",
        weights=xr.DataArray(
            [1.0, 1.0, 3.0, 1.0],
            dims=("trial",),
            coords={"trial": ["t0", "t1", "t2", "t3"]},
        ),
    )
    expected_win = (source.as_dataset(copy="none")["value"].isel(trial=0) + 3 * source.as_dataset(copy="none")["value"].isel(trial=2)) / 4
    xr.testing.assert_allclose(
        weighted.as_dataset(copy="none")["value"].sel(group_key="win").drop_vars("group_key"),
        expected_win.drop_vars("outcome"),
    )

    over_axis = source.group.groupby("outcome").sum(dim=("trial", "axis"))
    assert tuple(over_axis.as_dataset(copy="none")["value"].dims) == ("group_key", "lane")
    with pytest.raises(ValueError, match="must include primary batch dimension"):
        source.group.groupby("outcome").mean(dim="axis")

    flags = AnalysisObject.from_data(
        xr.Dataset(
            {"flag": ("trial", [False, True, False, False])},
            coords={"trial": [0, 1, 2, 3], "outcome": ("trial", ["a", "a", "b", "b"])},
        ),
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )
    assert flags.group.groupby("outcome").any().as_dataset(copy="none")["flag"].values.tolist() == [True, False]


def test_batch_grouping_skipna_only_observes_group_members() -> None:
    """ID: BATCH_GROUP_011_skipna_observes_only_group_members."""
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("trial", [1.0, np.nan, 3.0, 4.0])},
            coords={"trial": [0, 1, 2, 3], "outcome": ("trial", ["a", "b", "a", "b"])},
        ),
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )
    actual = source.group.groupby("outcome").mean(skipna=False).as_dataset(copy="none")["value"]
    assert actual.sel(group_key="a").item() == 2.0
    assert np.isnan(actual.sel(group_key="b").item())


def test_batch_grouping_options_surface_and_preflight() -> None:
    """ID: BATCH_GROUP_012_options_surface_and_preflight_are_topology_specific."""
    source = _batch_ao()
    grouped = source.group.groupby("outcome", preserve_batch=False)
    assert isinstance(grouped, BatchGroupedView)
    assert not hasattr(grouped, "materialize")
    assert not hasattr(grouped, "padded")
    assert not hasattr(grouped, "stacked")

    renamed = grouped.mean(
        opts=BatchGroupReduceOptions(group_dim="result_group")
    ).as_dataset(copy="none")
    assert "result_group" in renamed.dims
    with pytest.raises(TypeError, match="cannot be combined"):
        source.group.groupby("outcome", opts=GroupByOptions(), preserve_batch=False)
    with pytest.raises(ValueError, match="preserve_batch=True"):
        source.group.groupby("outcome", preserve_batch=True)
    with pytest.raises(ValueError, match="member_dim applies only"):
        source.group.groupby("outcome", opts=GroupByOptions(member_dim="member"))
    with pytest.raises(ValueError, match="sequence_index_coord applies only"):
        source.group.groupby(
            "outcome",
            opts=GroupByOptions(sequence_index_coord="source_row"),
        )
    with pytest.raises(TypeError, match="BatchGroupReduceOptions"):
        grouped.mean(opts=GroupMaterializeOptions())
    with pytest.raises(ValueError, match="collides with source namespace"):
        grouped.mean(opts=BatchGroupReduceOptions(group_dim="axis"))


def test_batch_grouping_is_lazy_except_for_declared_key_planning() -> None:
    """ID: BATCH_GROUP_013_only_declared_key_planning_is_eager."""
    calls: list[str] = []

    def record(name: str, values: np.ndarray) -> np.ndarray:
        calls.append(name)
        return values

    key = da.from_delayed(
        delayed(record)("key", np.asarray(["a", "b", "a", "b"], dtype=object)),
        shape=(4,),
        dtype=object,
    )
    payload = da.from_delayed(
        delayed(record)("payload", np.asarray([1.0, 2.0, 3.0, 4.0])),
        shape=(4,),
        dtype=float,
    )
    source = AnalysisObject.from_data(
        xr.Dataset({"value": ("trial", payload), "key": ("trial", key)}),
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )

    grouped = source.group.groupby("key")
    assert calls == []
    out = grouped.mean()
    assert calls == ["key"]
    assert isinstance(out.as_dataset(copy="none")["value"].data, da.Array)
    assert "payload" not in calls
    second = grouped.sum()
    assert calls == ["key"]
    assert isinstance(second.as_dataset(copy="none")["value"].data, da.Array)
    out.as_dataset(copy="none")["value"].compute()
    assert calls == ["key", "payload"]


def test_batch_grouping_wrong_options_fail_before_lazy_key_realization() -> None:
    """ID: BATCH_GROUP_014_wrong_options_fail_before_key_realization."""
    calls: list[str] = []

    @delayed
    def key_values() -> np.ndarray:
        calls.append("key")
        return np.asarray(["a", "b"], dtype=object)

    source = AnalysisObject.from_data(
        xr.Dataset(
            {
                "value": ("trial", [1.0, 2.0]),
                "key": ("trial", da.from_delayed(key_values(), shape=(2,), dtype=object)),
            }
        ),
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )
    grouped = source.group.groupby("key")
    with pytest.raises(TypeError, match="BatchGroupReduceOptions"):
        grouped.mean(opts=GroupMaterializeOptions())
    assert calls == []


def test_sequence_grouping_sugar_and_options_remain_identical() -> None:
    """ID: BATCH_GROUP_015_sequence_grouping_remains_unchanged."""
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), [[1.0, 2.0], [3.0, 4.0]])},
            coords={
                "trial": ["a", "b"],
                "sample": [0, 1],
                "outcome": ("trial", ["x", "y"]),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )
    sugar = source.group.groupby("outcome", preserve_batch=True).mean(dim="trial")
    options = source.group.groupby(
        "outcome",
        opts=GroupByOptions(preserve_batch=True),
    ).mean(dim="trial")
    xr.testing.assert_identical(
        sugar.as_dataset(copy="none"),
        options.as_dataset(copy="none"),
    )
    with pytest.raises(TypeError, match="GroupMaterializeOptions"):
        source.group.groupby("outcome").mean(opts=BatchGroupReduceOptions())


def test_grouping_requires_sequence_or_batch_semantics() -> None:
    """ID: BATCH_GROUP_016_requires_sequence_or_batch_semantics."""
    source = _roleless_grouping_ao()
    with pytest.raises(ValueError, match="sequence dimension or at least one batch"):
        source.group.groupby(xr.DataArray(["a", "b"], dims=("axis",)))


def test_missing_topology_precedes_topology_specific_options_and_key_work() -> None:
    """ID: BATCH_GROUP_045_missing_topology_precedes_configuration."""
    source = _roleless_grouping_ao()
    invalid_options = GroupByOptions(group_dim="duplicate", member_dim="duplicate")

    with pytest.raises(ValueError, match="sequence dimension or at least one batch"):
        source.group.groupby("axis", opts=invalid_options)
    with pytest.raises(ValueError, match="sequence dimension or at least one batch"):
        source.group.groupby(object())


@pytest.mark.parametrize(
    ("opts", "message"),
    [
        (GroupByOptions(group_dim=""), "opts.group_dim must be a non-empty string"),
        (GroupByOptions(member_dim=""), "opts.member_dim must be a non-empty string"),
        (
            GroupByOptions(sequence_index_coord=""),
            "opts.sequence_index_coord must be a non-empty string",
        ),
    ],
)
def test_basic_groupby_option_fields_precede_topology(
    opts: GroupByOptions,
    message: str,
) -> None:
    """ID: BATCH_GROUP_046_basic_option_fields_precede_topology."""
    source = _roleless_grouping_ao()

    with pytest.raises(ValueError, match=message):
        source.group.groupby(object(), opts=opts)
