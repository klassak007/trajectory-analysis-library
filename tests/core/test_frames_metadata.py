from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import tal.core as core
from tal.core import SchemaError, merge_schema, set_roles
from tal.utils.frame_schema import get_frames, set_frames


def _ds_sample_axis() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"value": (("sample", "axis"), np.ones((4, 3)))},
        coords={"sample": [0, 1, 2, 3], "axis": ["x", "y", "z"]},
    )


def _bound_ds() -> xr.Dataset:
    return set_roles(
        _ds_sample_axis(),
        sequence_dim="sample",
        batch_dims=(),
        core_dims=("axis",),
    )


def test_frame_core_001_get_set_frames_roundtrip_canonical_ext_namespace() -> None:
    """ID: FRAME_CORE_001_get_set_frames_roundtrip_canonical_ext_namespace."""
    ds = merge_schema(_bound_ds(), {"ext": {"demo": {"enabled": True}}}, validate=False)
    out = set_frames(ds, parent=" world ", child=" robot ")
    assert get_frames(out) == ("world", "robot")
    ext = out.attrs["tal"]["ext"]
    assert ext["demo"] == {"enabled": True}
    assert ext["frames"] == {"parent": "world", "child": "robot"}


def test_frame_core_002_set_frames_partial_update_and_clear_semantics() -> None:
    """ID: FRAME_CORE_002_set_frames_partial_update_and_clear_semantics."""
    base = set_frames(_bound_ds(), parent="map", child="base")

    keep_child = set_frames(base, parent="odom")
    assert get_frames(keep_child) == ("odom", "base")

    clear_parent = set_frames(base, parent=None)
    assert get_frames(clear_parent) == (None, "base")
    assert clear_parent.attrs["tal"]["ext"]["frames"] == {"child": "base"}

    cleared = set_frames(clear_parent, child=None)
    assert get_frames(cleared) == (None, None)
    assert "frames" not in cleared.attrs["tal"].get("ext", {})


def test_frame_hard_001_set_frames_rejects_invalid_frame_ids_fail_closed() -> None:
    """ID: FRAME_HARD_001_set_frames_rejects_invalid_frame_ids_fail_closed."""
    ds = _bound_ds()
    with pytest.raises(SchemaError) as err_empty:
        set_frames(ds, parent=" ")
    assert err_empty.value.code == "schema.frames.id.invalid"
    assert err_empty.value.path == "tal.ext.frames.parent"

    with pytest.raises(SchemaError) as err_type:
        set_frames(ds, child=123)  # type: ignore[arg-type]
    assert err_type.value.code == "schema.frames.id.invalid"
    assert err_type.value.path == "tal.ext.frames.child"

    malformed = merge_schema(ds, {"ext": {"frames": "bad"}}, validate=False)
    with pytest.raises(SchemaError) as err_bad_shape:
        get_frames(malformed)
    assert err_bad_shape.value.code == "schema.not_mapping"
    assert err_bad_shape.value.path == "tal.ext.frames"


def test_frame_hard_004_core_does_not_own_frame_metadata_api() -> None:
    """ID: FRAME_HARD_004_core_does_not_own_frame_metadata_api."""
    assert not hasattr(core, "get_frames")
    assert not hasattr(core, "set_frames")


def test_frame_hard_005_frame_schema_mixed_incomparable_keys_fail_closed_schema_error() -> None:
    """ID: FRAME_HARD_005_frame_schema_mixed_incomparable_keys_fail_closed_schema_error."""
    ds = _bound_ds()
    mixed = merge_schema(ds, {"ext": {"frames": {"parent": "world", 1: "bad"}}}, validate=False)
    with pytest.raises(SchemaError) as err_get:
        get_frames(mixed)
    assert err_get.value.code == "schema.frames.key.invalid"
    assert err_get.value.path == "tal.ext.frames"
    with pytest.raises(SchemaError) as err_set:
        set_frames(mixed, child="robot")
    assert err_set.value.code == "schema.frames.key.invalid"
    assert err_set.value.path == "tal.ext.frames"

    unknown = merge_schema(ds, {"ext": {"frames": {"parent": "world", "extra": "bad"}}}, validate=False)
    with pytest.raises(SchemaError) as err_unknown:
        get_frames(unknown)
    assert err_unknown.value.code == "schema.frames.unknown_key"
    assert err_unknown.value.path == "tal.ext.frames.extra"
