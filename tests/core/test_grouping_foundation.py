from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr

import tal.core.group_ops.key_resolve as key_resolve_module
from tal.core.analysis_object import AnalysisObject
from tal.core.group_ops import GroupingBinSpec, GroupingFoundationOptions, resolve_grouping_foundation_context
from tal.core.group_ops.options import coerce_grouping_foundation_options


def _grouping_ao() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={
            "signal": (("trial", "sample"), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)),
            "bucket": (("trial", "sample"), np.array([[0, 1, 0], [1, 0, 1]], dtype=int)),
        },
        coords={
            "trial": np.array(["t0", "t1"], dtype=object),
            "sample": np.array([0, 1, 2], dtype=int),
            "outcome": ("trial", np.array(["ok", "fail"], dtype=object)),
            "time_s": (
                ("trial", "sample"),
                np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=float),
            ),
            "phase": (
                ("trial", "sample"),
                np.array([["A", "B", "B"], ["A", "A", "B"]], dtype=object),
            ),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s",
    )


def _windowed_grouping_ao() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={
            "signal": (
                ("trial", "event", "tau"),
                np.asarray(
                    [
                        [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],
                        [[4.0, 5.0, 6.0], [4.1, 5.1, 6.1]],
                    ],
                    dtype=float,
                ),
            ),
        },
        coords={
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "event": np.asarray([0, 1], dtype=np.int64),
            "tau": np.asarray([-1.0, 0.0, 1.0], dtype=float),
            "window_tau": ("tau", np.asarray([-1.0, 0.0, 1.0], dtype=float)),
            "outcome": ("trial", np.asarray(["ok", "fail"], dtype=object)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="tau",
        batch_dims=("trial", "event"),
        core_dims=(),
        param_coord="window_tau",
    )


def _external_key(ao: AnalysisObject, values: np.ndarray, *, name: str = "ext_key") -> xr.DataArray:
    ds = ao.unsafe_data
    return xr.DataArray(
        values,
        dims=("trial", "sample"),
        coords={"trial": ds.coords["trial"], "sample": ds.coords["sample"]},
        name=name,
    )


def _sequence_only_external_key(ao: AnalysisObject, values: np.ndarray, *, name: str = "seq_key") -> xr.DataArray:
    ds = ao.unsafe_data
    return xr.DataArray(
        values,
        dims=("sample",),
        coords={"sample": ds.coords["sample"]},
        name=name,
    )


def test_group_core_p9a_001_grouping_options_are_frozen_and_validated() -> None:
    """ID: GROUP_CORE_P9A_001_grouping_options_are_frozen_and_validated."""
    default_opts = coerce_grouping_foundation_options(None, owner="grouping.test")
    assert default_opts.na_key_policy == "error"
    with pytest.raises(FrozenInstanceError):
        default_opts.na_key_policy = "drop"
    with pytest.raises(ValueError, match="na_key_policy"):
        coerce_grouping_foundation_options(
            GroupingFoundationOptions(na_key_policy="invalid"),  # type: ignore[arg-type]
            owner="grouping.test",
        )
    with pytest.raises(ValueError, match="na_group_label"):
        coerce_grouping_foundation_options(
            GroupingFoundationOptions(na_key_policy="drop", na_group_label="NA"),
            owner="grouping.test",
        )


def test_group_core_p9a_002_grouping_key_resolution_is_deterministic_for_supported_key_kinds() -> None:
    """ID: GROUP_CORE_P9A_002_grouping_key_resolution_is_deterministic_for_supported_key_kinds."""
    ao = _grouping_ao()
    by_coord = resolve_grouping_foundation_context(ao, "time_s", owner="grouping.test")
    by_data_var = resolve_grouping_foundation_context(ao, "bucket", owner="grouping.test")
    by_external = resolve_grouping_foundation_context(
        ao,
        _external_key(ao, np.array([[1.0, 1.5, 2.0], [0.5, 1.0, 1.5]], dtype=float)),
        owner="grouping.test",
    )
    by_bins = resolve_grouping_foundation_context(
        ao,
        GroupingBinSpec(source="time_s", bins=np.array([-0.1, 0.5, 1.5, 2.5])),
        owner="grouping.test",
    )
    assert by_coord.keys[0].kind == "coord"
    assert by_data_var.keys[0].kind == "data_var"
    assert by_external.keys[0].kind == "external"
    assert by_bins.keys[0].kind == "bin"
    assert by_bins.keys[0].data.dims == ("trial", "sample")


def test_group_core_p9a_003_grouping_context_resolution_uses_schema_roles_not_heuristic_defaults() -> None:
    """ID: GROUP_CORE_P9A_003_grouping_context_resolution_uses_schema_roles_not_heuristic_defaults."""
    raw = AnalysisObject(
        xr.Dataset(
            {"signal": (("trial", "sample"), np.ones((2, 3), dtype=float))},
            coords={"trial": np.array(["t0", "t1"], dtype=object), "sample": np.array([0, 1, 2], dtype=int)},
        )
    )
    with pytest.raises(ValueError, match="requires declared roles"):
        resolve_grouping_foundation_context(raw, "sample", owner="grouping.test")


def test_group_core_p9a_004_multi_key_grouping_composition_is_deterministic() -> None:
    """ID: GROUP_CORE_P9A_004_multi_key_grouping_composition_is_deterministic."""
    ao = _grouping_ao()
    external = _external_key(ao, np.array([[10, 11, 12], [13, 14, 15]], dtype=float), name="ext")
    context = resolve_grouping_foundation_context(
        ao,
        ("time_s", external, GroupingBinSpec(source="time_s", bins=[-0.1, 0.5, 1.5, 2.5])),
        owner="grouping.test",
    )
    assert tuple(key.index for key in context.keys) == (0, 1, 2)
    assert tuple(key.name for key in context.keys) == ("time_s", "ext", "time_s__bin")


def test_group_core_p9a_005_grouping_key_alignment_probe_avoids_dense_reference_allocation_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: GROUP_CORE_P9A_005_grouping_key_alignment_probe_avoids_dense_reference_allocation_at_construction."""
    ao = _grouping_ao()

    def _raise_dense_zeros(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("grouping key resolution must not allocate dense np.zeros reference payloads")

    monkeypatch.setattr(key_resolve_module.np, "zeros", _raise_dense_zeros)
    context = resolve_grouping_foundation_context(ao, "phase", owner="grouping.test")
    assert context.keys[0].name == "phase"


def test_group_hard_p9a_001_dotted_grouping_paths_are_rejected() -> None:
    """ID: GROUP_HARD_P9A_001_dotted_grouping_paths_are_rejected."""
    ao = _grouping_ao()
    with pytest.raises(ValueError, match="dotted grouping paths"):
        resolve_grouping_foundation_context(ao, "phase.value", owner="grouping.test")


def test_group_hard_p9a_002_invalid_or_ambiguous_grouping_keys_fail_closed() -> None:
    """ID: GROUP_HARD_P9A_002_invalid_or_ambiguous_grouping_keys_fail_closed."""
    ao = _grouping_ao()
    with pytest.raises(TypeError, match="key must"):
        resolve_grouping_foundation_context(ao, {"invalid": "key"}, owner="grouping.test")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not found"):
        resolve_grouping_foundation_context(ao, "missing_key", owner="grouping.test")
    with pytest.raises(TypeError, match="nested key containers"):
        resolve_grouping_foundation_context(ao, ("time_s", ("bucket",)), owner="grouping.test")  # type: ignore[arg-type]


def test_group_hard_p9a_004_grouping_key_alignment_requires_exact_label_match() -> None:
    """ID: GROUP_HARD_P9A_004_grouping_key_alignment_requires_exact_label_match."""
    ao = _grouping_ao()
    bad = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]], dtype=float),
        dims=("trial", "sample"),
        coords={"trial": ao.unsafe_data.coords["trial"], "sample": np.array([0, 1, 99], dtype=int)},
        name="bad_labels",
    )
    with pytest.raises(ValueError, match="exact|aligned|labels"):
        resolve_grouping_foundation_context(ao, bad, owner="grouping.test")


def test_group_hard_p9a_006_broadcast_intent_does_not_relax_grouping_key_alignment() -> None:
    """ID: GROUP_HARD_P9A_006_broadcast_intent_does_not_relax_grouping_key_alignment."""
    ao = _grouping_ao().b()
    sequence_only = _sequence_only_external_key(ao, np.array([0.0, 1.0, 2.0], dtype=float))
    with pytest.raises(ValueError, match="non-core dim names|exact|aligned|row dims"):
        resolve_grouping_foundation_context(ao, sequence_only, owner="grouping.test")


def test_grouping_foundation_accepts_batch_only_key_and_broadcasts_over_sequence_dim() -> None:
    ao = _grouping_ao()
    context = resolve_grouping_foundation_context(ao, "outcome", owner="grouping.test")
    data = context.keys[0].data
    assert data.dims == ("trial", "sample")
    np.testing.assert_array_equal(
        data.to_numpy(),
        np.array([["ok", "ok", "ok"], ["fail", "fail", "fail"]], dtype=object),
    )


def test_grouping_foundation_accepts_primary_batch_only_key_for_windowed_rows() -> None:
    ao = _windowed_grouping_ao()
    context = resolve_grouping_foundation_context(ao, "outcome", owner="grouping.test")
    data = context.keys[0].data
    assert data.dims == ("trial", "event", "tau")
    np.testing.assert_array_equal(
        data.isel(event=0, tau=0).to_numpy(),
        np.asarray(["ok", "fail"], dtype=object),
    )
    np.testing.assert_array_equal(
        data.isel(event=1, tau=2).to_numpy(),
        np.asarray(["ok", "fail"], dtype=object),
    )


def test_group_hard_p9a_005_default_na_key_policy_is_fail_closed_and_explicit_modes_are_deterministic() -> None:
    """ID: GROUP_HARD_P9A_005_default_na_key_policy_is_fail_closed."""
    ao = _grouping_ao()
    key = _external_key(ao, np.array([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]], dtype=float), name="na_key")
    with pytest.raises(ValueError, match="na_key_policy='error'"):
        resolve_grouping_foundation_context(ao, key, owner="grouping.test")
    dropped = resolve_grouping_foundation_context(
        ao,
        key,
        opts=GroupingFoundationOptions(na_key_policy="drop"),
        owner="grouping.test",
    )
    assert dropped.na_exclusion_mask is not None
    assert bool((~dropped.na_exclusion_mask).any())
    grouped = resolve_grouping_foundation_context(
        ao,
        key,
        opts=GroupingFoundationOptions(na_key_policy="group", na_group_label="NA_GROUP"),
        owner="grouping.test",
    )
    assert grouped.na_group_label == "NA_GROUP"
    assert bool((grouped.keys[0].data == "NA_GROUP").any())


def test_group_hard_p9a_007_default_na_group_label_collision_fails_closed() -> None:
    """ID: GROUP_HARD_P9A_007_default_na_group_label_collision_fails_closed."""
    ao = _grouping_ao()
    colliding = _external_key(
        ao,
        np.array([["__tal_na_group__", np.nan, "X"], ["Y", "Z", "K"]], dtype=object),
        name="default_colliding",
    )
    with pytest.raises(ValueError, match="collides"):
        resolve_grouping_foundation_context(
            ao,
            colliding,
            opts=GroupingFoundationOptions(na_key_policy="group"),
            owner="grouping.test",
        )


def test_group_hard_p9a_009_explicit_na_group_label_collision_fails_closed() -> None:
    """ID: GROUP_HARD_P9A_009_explicit_na_group_label_collision_fails_closed."""
    ao = _grouping_ao()
    colliding = _external_key(
        ao,
        np.array([["NA_GROUP", np.nan, "X"], ["Y", "Z", "K"]], dtype=object),
        name="colliding",
    )
    with pytest.raises(ValueError, match="collides"):
        resolve_grouping_foundation_context(
            ao,
            colliding,
            opts=GroupingFoundationOptions(na_key_policy="group", na_group_label="NA_GROUP"),
            owner="grouping.test",
        )


def test_group_hard_p9a_008_chunked_na_policy_scalar_checks_fail_closed_no_hidden_eager_compute() -> None:
    """ID: GROUP_HARD_P9A_008_chunked_na_policy_scalar_checks_fail_closed_no_hidden_eager_compute."""
    pytest.importorskip("dask.array")
    ao = _grouping_ao()
    key = _external_key(
        ao,
        np.array([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]], dtype=float),
        name="chunked_na_key",
    ).chunk({"trial": 1})
    with pytest.raises(ValueError, match="chunked grouping NA-policy scalar checks are not supported"):
        resolve_grouping_foundation_context(
            ao,
            key,
            opts=GroupingFoundationOptions(na_key_policy="group", na_group_label="NA_GROUP"),
            owner="grouping.test",
        )


@pytest.mark.parametrize(
    "labels",
    (
        ("A", "A", "B"),
        (float("nan"), float("nan"), "B"),
    ),
)
def test_group_hard_p9a_010_duplicate_bin_labels_fail_closed_in_foundation_validation(
    labels: tuple[object, object, object],
) -> None:
    """ID: GROUP_HARD_P9A_010_duplicate_bin_labels_fail_closed_in_foundation_validation."""
    ao = _grouping_ao()
    with pytest.raises(ValueError, match="labels must be unique|duplicate label"):
        resolve_grouping_foundation_context(
            ao,
            GroupingBinSpec(
                source="time_s",
                bins=np.array([-0.5, 0.5, 1.5, 2.5], dtype=float),
                labels=labels,
            ),
            owner="grouping.test",
        )


def test_group_hard_p9a_011_unhashable_bin_labels_fail_closed_at_foundation_boundary() -> None:
    """ID: GROUP_HARD_P9A_011_unhashable_bin_labels_fail_closed_at_foundation_boundary."""
    ao = _grouping_ao()
    with pytest.raises(ValueError, match="labels must be hashable|unhashable"):
        resolve_grouping_foundation_context(
            ao,
            GroupingBinSpec(
                source="time_s",
                bins=np.array([-0.5, 0.5, 1.5, 2.5], dtype=float),
                labels=[[1], [2], [3]],
            ),
            owner="grouping.test",
        )
