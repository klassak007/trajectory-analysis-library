from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import tal.spatial as spatial
import tal.spatial.position as position_module
from tal import AnalysisObject
from tal.core import ParamEvalOptions, ParamSyncOptions, synchronize_param
from tal.core.schema_errors import SchemaError
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.linalg import Array
from tal.spatial import LinearVelocity, Position
from tal.spatial.metadata import get_position_intent, get_position_rep
from tal.spatial.temporal.options import KinematicsDerivativeOptions
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames


def _position_dataset(
    *,
    var_name: str = "position",
    labels: tuple[str, str, str] = ("x", "y", "z"),
    values: np.ndarray | None = None,
) -> xr.Dataset:
    if values is None:
        values = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=float,
        )
    arr = xr.DataArray(
        values,
        dims=("sample", "axis"),
        coords={"sample": [0, 1], "axis": list(labels)},
        name=var_name,
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name=var_name),
        sequence_dim="sample",
        core_dims=("axis",),
        validate=True,
    )
    return ao.unsafe_data.copy(deep=True)


def _as_dataarray_with_schema(ds: xr.Dataset, *, var_name: str = "position") -> xr.DataArray:
    da = ds[var_name].copy(deep=True)
    da.attrs["tal"] = ds.attrs["tal"]
    return da


def _position_temporal_dataset(
    *,
    batched: bool = False,
    include_param: bool = True,
    nonmonotonic_param: bool = False,
    nonnumeric_param: bool = False,
) -> xr.Dataset:
    if batched:
        values = np.asarray(
            [
                [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
            ],
            dtype="float64",
        )
        coords: dict[str, object] = {
            "trial": ["t0", "t1"],
            "sample": [0, 1],
            "axis": ["x", "y", "z"],
            "group_size": ("trial", [2, 2]),
        }
        if include_param:
            coords["time_s"] = (("trial", "sample"), [[0.0, 0.5], [0.0, 1.0]])
        ds = xr.Dataset(
            data_vars={"position": (("trial", "sample", "axis"), values)},
            coords=coords,
        )
        return AnalysisObject.from_data(
            ds,
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("axis",),
            param_coord="time_s" if include_param else None,
            sequence_size_coord="group_size",
            validate=not nonnumeric_param,
        ).unsafe_data.copy(deep=True)
    values = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype="float64",
    )
    if nonmonotonic_param:
        time = [0.0, 1.0, 0.5]
    elif nonnumeric_param:
        time = ["t0", "t1", "t2"]
    else:
        time = [0.0, 0.5, 1.0]
    coords = {
        "sample": [0, 1, 2],
        "axis": ["x", "y", "z"],
        "group_size": xr.DataArray(np.asarray(3, dtype="int64"), dims=()),
    }
    if include_param:
        coords["time_s"] = ("sample", time)
    ds = xr.Dataset(
        data_vars={"position": (("sample", "axis"), values)},
        coords=coords,
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=("axis",),
        param_coord="time_s" if include_param else None,
        sequence_size_coord="group_size",
        validate=not nonnumeric_param,
    ).unsafe_data.copy(deep=True)


def test_spatial_core_001_position_constructor_accepts_ao_dataset_dataarray_deterministically() -> None:
    """ID: SPATIAL_CORE_001_position_constructor_accepts_ao_dataset_dataarray_deterministically."""
    ds = _position_dataset()
    ao = AnalysisObject._from_validated(ds)
    da = _as_dataarray_with_schema(ds)

    from_ao = Position(ao)
    from_ds = Position(ds)
    from_da = Position(da)

    assert isinstance(from_ao, Position)
    assert isinstance(from_ds, Position)
    assert isinstance(from_da, Position)
    assert get_position_rep(from_ao.unsafe_data, owner="test") == "cart"
    assert get_position_rep(from_ds.unsafe_data, owner="test") == "cart"
    assert get_position_rep(from_da.unsafe_data, owner="test") == "cart"


def test_spatial_core_002_position_type_routing_rewrap_behavior_deterministic() -> None:
    """ID: SPATIAL_CORE_002_position_type_routing_rewrap_behavior_deterministic."""
    pos = Position(_position_dataset())
    tagged = frame_retag(pos, parent="world", child="A", validate=True)
    delta = pos.as_delta(validate=True)
    out = tagged + delta

    assert isinstance(tagged, Position)
    assert isinstance(delta, Position)
    assert isinstance(out, Position)
    assert get_position_intent(delta.unsafe_data, owner="test") == "delta"
    assert get_position_intent(out.unsafe_data, owner="test") is None


def test_spatial_core_011_position_single_type_no_translation_class_semantics_locked() -> None:
    """ID: SPATIAL_CORE_011_position_single_type_no_translation_class_semantics_locked."""
    assert hasattr(spatial, "Position")
    assert not hasattr(spatial, "Translation")
    assert issubclass(Position, AnalysisObject)
    assert hasattr(Position, "as_delta")


def test_spatial_core_012_position_frame_state_inference_policy_deterministic() -> None:
    """ID: SPATIAL_CORE_012_position_frame_state_inference_policy_deterministic."""
    left = frame_retag(Position(_position_dataset(var_name="left")), parent="world", child="A", validate=True)
    right_unframed = Position(_position_dataset(var_name="right"))
    out_one_framed = left + right_unframed
    assert get_frames(out_one_framed.unsafe_data) == ("world", "A")

    right_chain = frame_retag(
        Position(_position_dataset(var_name="right_chain")),
        parent="A",
        child="B",
        validate=True,
    )
    out_chain = left + right_chain
    assert get_frames(out_chain.unsafe_data) == ("world", "B")

    unframed_delta = Position(_position_dataset(var_name="delta")).as_delta(validate=True)
    out_unframed = Position(_position_dataset(var_name="point")) + unframed_delta
    assert get_frames(out_unframed.unsafe_data) == (None, None)


def test_spatial_hard_001_position_constructor_type_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_001_position_constructor_type_mismatch_fail_closed."""
    with pytest.raises(TypeError, match="spatial.position.__init__"):
        Position(object())


def test_spatial_hard_002_position_constructor_semantic_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_002_position_constructor_semantic_mismatch_fail_closed."""
    ds_multi = _position_dataset().assign(extra=("sample", [1.0, 2.0]))
    with pytest.raises(ValueError, match="exactly one data variable"):
        Position(ds_multi)

    ds_bad_labels = _position_dataset(labels=("x", "y", "q"))
    with pytest.raises(ValueError, match="core labels must equal"):
        Position(ds_bad_labels)


def test_spatial_hard_009_position_unframed_ambiguous_op_requires_explicit_displacement_intent() -> None:
    """ID: SPATIAL_HARD_009_position_unframed_ambiguous_op_requires_explicit_displacement_intent."""
    left = Position(_position_dataset(var_name="left"))
    right = Position(_position_dataset(var_name="right"))
    with pytest.raises(ValueError, match="as_delta\\(\\)"):
        _ = left + right


def test_spatial_hard_010_position_constructor_malformed_frames_block_fails_closed() -> None:
    """ID: SPATIAL_HARD_010_position_constructor_malformed_frames_block_fails_closed."""
    ds = _position_dataset()
    tal = dict(ds.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    ext["frames"] = {"parent": "world", "child": "A", "extra": "bad"}
    tal["ext"] = ext
    ds.attrs["tal"] = tal

    with pytest.raises(SchemaError, match="tal.ext.frames.extra"):
        Position(ds)


def test_spatial_core_013_position_constructor_accepts_forward_compatible_spatial_roles_keys() -> None:
    """ID: SPATIAL_CORE_013_position_constructor_accepts_forward_compatible_spatial_roles_keys."""
    ds = _position_dataset()
    tal = dict(ds.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    spatial_block = dict(ext.get("spatial", {}))
    spatial_block["roles"] = {"position_intent": "delta", "rotation_hint": "quat"}
    ext["spatial"] = spatial_block
    tal["ext"] = ext
    ds.attrs["tal"] = tal

    pos = Position(ds)
    assert isinstance(pos, Position)
    assert get_position_intent(pos.unsafe_data, owner="test") == "delta"


def test_spatial_core_175_position_norm_magnitude_topology_and_metadata_preserved() -> None:
    """ID: SPATIAL_CORE_175_position_norm_magnitude_topology_and_metadata_preserved."""
    pos = Position(_position_temporal_dataset(batched=True))
    out_norm = pos.norm(ord=2)
    out_mag = pos.magnitude()

    assert isinstance(out_norm, Array)
    assert isinstance(out_mag, Array)
    assert tuple(out_norm.unsafe_data.data_vars) == ("datavar",)

    declared, sequence_dim, batch_dims, core_dims = read_roles(out_norm.unsafe_data)
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ()
    assert read_param_coord_name(out_norm.unsafe_data) == "time_s"
    assert read_sequence_size_coord_name(out_norm.unsafe_data) == "group_size"
    xr.testing.assert_identical(out_norm.unsafe_data, out_mag.unsafe_data)


def test_spatial_core_176_position_norm_value_parity_and_ord_passthrough() -> None:
    """ID: SPATIAL_CORE_176_position_norm_value_parity_and_ord_passthrough."""
    pos = Position(
        _position_dataset(
            values=np.asarray(
                [
                    [3.0, 4.0, 0.0],
                    [1.0, 2.0, 2.0],
                ],
                dtype="float64",
            )
        )
    )
    out_l2 = pos.norm(ord=2)
    out_l1 = pos.norm(ord=1)
    expected_l2 = np.asarray([5.0, 3.0], dtype="float64")
    expected_l1 = np.asarray([7.0, 5.0], dtype="float64")

    np.testing.assert_allclose(out_l2.unsafe_data["datavar"].values, expected_l2)
    np.testing.assert_allclose(out_l1.unsafe_data["datavar"].values, expected_l1)
    xr.testing.assert_identical(out_l2.unsafe_data, pos.magnitude().unsafe_data)


def test_spatial_hard_011_position_constructor_rejects_non_string_spatial_roles_keys() -> None:
    """ID: SPATIAL_HARD_011_position_constructor_rejects_non_string_spatial_roles_keys."""
    ds = _position_dataset()
    tal = dict(ds.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    spatial_block = dict(ext.get("spatial", {}))
    spatial_block["roles"] = {1: "bad"}
    ext["spatial"] = spatial_block
    tal["ext"] = ext
    ds.attrs["tal"] = tal

    with pytest.raises(ValueError, match="tal.ext.spatial.roles keys must be strings"):
        Position(ds)


def test_bcast_core_050_position_add_intent_resolution_precedes_alignment_and_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: BCAST_CORE_050_position_add_intent_resolution_precedes_alignment_and_broadcast."""
    left = Position(_position_dataset(var_name="left"))
    right = Position(_position_dataset(var_name="right"))

    def _boom(*_args: object, **_kwargs: object) -> AnalysisObject:
        raise AssertionError("position add should fail on intent before numeric alignment.")

    monkeypatch.setattr(position_module, "linalg_add", _boom)
    with pytest.raises(ValueError, match="as_delta\\(\\)"):
        _ = left + right


def test_bcast_hard_038_unframed_position_add_ambiguity_fails_before_alignment() -> None:
    """ID: BCAST_HARD_038_unframed_position_add_ambiguity_fails_before_alignment."""
    left = Position(_position_dataset(var_name="left")).b()
    right = Position(_position_dataset(var_name="right")).a(on="sequence")
    with pytest.raises(ValueError, match="as_delta\\(\\)"):
        _ = left + right


def test_spatial_core_103_temporal_vector_like_explicit_query_grid_interpolation_boundary() -> None:
    """ID: SPATIAL_CORE_103_temporal_vector_like_explicit_query_grid_interpolation_boundary."""
    pos = Position(_position_temporal_dataset())
    out = pos.param.at([0.25, 0.75])
    assert isinstance(out, Position)
    assert "sample" in out.unsafe_data["position"].dims
    assert "axis" in out.unsafe_data["position"].dims
    assert out.unsafe_data.sizes["sample"] == 2


def test_spatial_core_104_temporal_vector_like_resample_boundary_preserves_roles_frames_kind() -> None:
    """ID: SPATIAL_CORE_104_temporal_vector_like_resample_boundary_preserves_roles_frames_kind."""
    pos = frame_retag(Position(_position_temporal_dataset()), parent="world", child="body", validate=True)
    out = pos.param.resample_to([0.0, 0.5, 1.0])
    assert isinstance(out, Position)
    assert get_frames(out.unsafe_data) == ("world", "body")
    assert "sample" in out.unsafe_data["position"].dims
    assert "axis" in out.unsafe_data["position"].dims


def test_spatial_core_106_temporal_vector_like_valid_mask_and_query_validity_propagation_truthful() -> None:
    """ID: SPATIAL_CORE_106_temporal_vector_like_valid_mask_and_query_validity_propagation_truthful."""
    pos = Position(_position_temporal_dataset(batched=True))
    out = pos.param.resample_to(xr.DataArray([0.0, 0.5], dims=("sample",)))
    assert "valid" in out.unsafe_data.coords
    np.testing.assert_array_equal(out.unsafe_data.coords["group_size"].values, [2, 2])
    np.testing.assert_array_equal(out.unsafe_data.coords["valid"].sel(trial="t0").values, [True, True])


def test_spatial_core_108_temporal_vector_like_dask_lazy_boundary_preserved() -> None:
    """ID: SPATIAL_CORE_108_temporal_vector_like_dask_lazy_boundary_preserved."""
    dask_array = pytest.importorskip("dask.array")
    values = dask_array.from_array(
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float64"),
        chunks=(2, 3),
    )
    ds = xr.Dataset(
        data_vars={"position": (("sample", "axis"), values)},
        coords={
            "sample": [0, 1],
            "axis": ["x", "y", "z"],
            "time_s": ("sample", [0.0, 1.0]),
        },
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), param_coord="time_s")
    out = Position(ao).param.at([0.25, 0.75])
    assert isinstance(out, Position)
    assert getattr(out.unsafe_data["position"].data, "chunks", None) is not None


def test_spatial_core_109_temporal_vector_like_param_and_sequence_size_metadata_preserved() -> None:
    """ID: SPATIAL_CORE_109_temporal_vector_like_param_and_sequence_size_metadata_preserved."""
    pos = Position(_position_temporal_dataset(batched=True))
    out = pos.param.resample_to([0.25, 0.75])
    assert read_param_coord_name(out.unsafe_data) == "time_s"
    assert read_sequence_size_coord_name(out.unsafe_data) == "group_size"


def test_spatial_hard_124_temporal_vector_like_rejects_missing_or_nonnumeric_param_coord() -> None:
    """ID: SPATIAL_HARD_124_temporal_vector_like_rejects_missing_or_nonnumeric_param_coord."""
    missing = Position(_position_temporal_dataset(include_param=False))
    with pytest.raises(ValueError, match="require a resolved param_coord"):
        missing.param.at([0.5])

    with pytest.raises(SchemaError, match="schema.param_coord.dtype.invalid"):
        Position(_position_temporal_dataset(include_param=True, nonnumeric_param=True))


def test_spatial_hard_125_temporal_vector_like_rejects_nonmonotonic_param_domain() -> None:
    """ID: SPATIAL_HARD_125_temporal_vector_like_rejects_nonmonotonic_param_domain."""
    pos = Position(_position_temporal_dataset(nonmonotonic_param=True))
    with pytest.raises(ValueError, match="must be monotonic non-decreasing"):
        pos.param.at([0.25, 0.75])


def test_spatial_hard_126_temporal_vector_like_rejects_query_dim_collision_with_sequence_dim() -> None:
    """ID: SPATIAL_HARD_126_temporal_vector_like_rejects_query_dim_collision_with_sequence_dim."""
    pos = Position(_position_temporal_dataset())
    with pytest.raises(ValueError, match="query_dim must differ from sequence_dim"):
        pos.param.at([0.25, 0.75], opts=ParamEvalOptions(query_dim="sample"))


def test_spatial_hard_127_temporal_vector_like_rejects_non_numeric_sequence_payloads_for_interp() -> None:
    """ID: SPATIAL_HARD_127_temporal_vector_like_rejects_non_numeric_sequence_payloads_for_interp."""
    pos = Position(_position_temporal_dataset())
    pos.unsafe_data["label"] = xr.DataArray(["a", "b", "c"], dims=("sample",))
    with pytest.raises(TypeError, match="non-numeric sequence variable"):
        pos.param.at([0.5])


def test_spatial_hard_128_temporal_vector_like_fail_closed_on_unsupported_method() -> None:
    """ID: SPATIAL_HARD_128_temporal_vector_like_fail_closed_on_unsupported_method."""
    pos = Position(_position_temporal_dataset())
    with pytest.raises(ValueError, match="opts.method must be one of"):
        pos.param.at([0.5], opts=ParamEvalOptions(method="cubic"))  # type: ignore[arg-type]


def test_spatial_hard_129_temporal_vector_like_owner_prefixed_lazy_error_boundary() -> None:
    """ID: SPATIAL_HARD_129_temporal_vector_like_owner_prefixed_lazy_error_boundary."""
    pos = Position(_position_temporal_dataset())
    with pytest.raises(ValueError, match="param at/resample"):
        pos.param.at([0.5], opts=ParamEvalOptions(method="cubic"))  # type: ignore[arg-type]


def test_spatial_hard_150_temporal_vector_like_param_key_request_rejects_missing_or_non_numeric_param_coord() -> None:
    """ID: SPATIAL_HARD_150_temporal_vector_like_param_key_request_rejects_missing_or_non_numeric_param_coord."""
    pos = Position(_position_temporal_dataset(include_param=False))
    with pytest.raises(ValueError, match="not found in dataset coords"):
        pos.param.at([0.5], on="missing_param")

    ds = _position_temporal_dataset(include_param=False)
    ds = ds.assign_coords(time_str=("sample", ["t0", "t1", "t2"]))
    nonnumeric = Position(ds)
    with pytest.raises(ValueError, match="require ordered real numeric or datetime64 param_coord"):
        nonnumeric.param.at([0.5], on="time_str")


def test_spatial_core_110_temporal_vector_like_auto_grid_behavior_explicit_and_track_separated() -> None:
    """ID: SPATIAL_CORE_110_temporal_vector_like_auto_grid_behavior_explicit_and_track_separated."""
    pos = Position(_position_temporal_dataset())
    out = pos.param.resample_to([0.25, 0.75])
    assert isinstance(out, Position)
    assert not hasattr(pos, "at_param")
    assert not hasattr(pos, "resample_param")


def test_spatial_core_119_kinematics_derivative_baseline_velocity_from_position_boundary() -> None:
    """ID: SPATIAL_CORE_119_kinematics_derivative_baseline_velocity_from_position_boundary."""
    pos = Position(_position_temporal_dataset())
    out = pos.differentiate(validate=True)
    assert isinstance(out, LinearVelocity)
    assert out.unsafe_data.sizes["sample"] == pos.unsafe_data.sizes["sample"]
    np.testing.assert_allclose(
        out.unsafe_data["position"].values,
        np.asarray([[6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]], dtype="float64"),
    )


def test_spatial_hard_140_kinematics_derivative_requires_numeric_monotonic_param_domain() -> None:
    """ID: SPATIAL_HARD_140_kinematics_derivative_requires_numeric_monotonic_param_domain."""
    bad = Position(_position_temporal_dataset(nonmonotonic_param=True))
    with pytest.raises(ValueError, match="spatial\\.position\\.differentiate"):
        _ = bad.differentiate()


def test_spatial_hard_142_kinematics_temporal_ops_reject_missing_param_coord_when_required() -> None:
    """ID: SPATIAL_HARD_142_kinematics_temporal_ops_reject_missing_param_coord_when_required."""
    missing = Position(_position_temporal_dataset(include_param=False))
    with pytest.raises(ValueError, match="spatial\\.position\\.differentiate"):
        _ = missing.differentiate()


def test_spatial_hard_144_kinematics_temporal_ops_fail_closed_on_unsupported_method_or_order() -> None:
    """ID: SPATIAL_HARD_144_kinematics_temporal_ops_fail_closed_on_unsupported_method_or_order."""
    pos = Position(_position_temporal_dataset())
    with pytest.raises(ValueError, match="spatial\\.position\\.differentiate"):
        _ = pos.differentiate(opts=KinematicsDerivativeOptions(method="local_poly"))
    with pytest.raises(ValueError, match="spatial\\.position\\.differentiate"):
        _ = pos.differentiate(opts=KinematicsDerivativeOptions(order=2))


def test_spatial_hard_130_temporal_vector_like_chunked_auto_grid_fail_closed_with_guidance() -> None:
    """ID: SPATIAL_HARD_130_temporal_vector_like_chunked_auto_grid_fail_closed_with_guidance."""
    dask_array = pytest.importorskip("dask.array")
    left_ds = xr.Dataset(
        data_vars={"position": (("sample", "axis"), dask_array.from_array(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), chunks=(2, 3)))},
        coords={
            "sample": [0, 1, 2],
            "axis": ["x", "y", "z"],
            "time_s": ("sample", dask_array.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"position": (("sample", "axis"), dask_array.from_array(np.asarray([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0], [17.0, 18.0, 19.0]]), chunks=(2, 3)))},
        coords={
            "sample": [0, 1, 2],
            "axis": ["x", "y", "z"],
            "time_s": ("sample", dask_array.from_array(np.asarray([1.0, 2.0, 3.0]), chunks=2)),
        },
    )
    left = Position(
        AnalysisObject.from_data(
            left_ds,
            sequence_dim="sample",
            core_dims=("axis",),
            param_coord="time_s",
        )
    )
    right = Position(
        AnalysisObject.from_data(
            right_ds,
            sequence_dim="sample",
            core_dims=("axis",),
            param_coord="time_s",
        )
    )
    with pytest.raises(ValueError, match="pass an explicit grid"):
        synchronize_param([left, right], opts=ParamSyncOptions(join="outer", how="nearest"))
