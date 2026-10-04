from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import (
    AnalysisObject,
    ComponentRegistryOptions,
    ComponentSpec,
    define_components,
    read_components,
)


def _base_ao(
    *,
    axis_labels: tuple[str, ...] = ("x", "y", "z"),
) -> AnalysisObject:
    values = np.arange(2 * 3 * len(axis_labels), dtype=float).reshape(2, 3, len(axis_labels))
    ds = xr.Dataset(
        {"value": (("trial", "sample", "axis"), values)},
        coords={
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "sample": np.asarray([0, 1, 2], dtype=np.int64),
            "axis": np.asarray(axis_labels, dtype=object),
            "phase": (("trial", "sample"), np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=float)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="phase",
        validate=True,
    )


def _registry_options() -> ComponentRegistryOptions:
    return ComponentRegistryOptions(
        registry={
            "position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value"),
            "heading": ComponentSpec(core_dim="axis", labels=("z",), var="value"),
        }
    )


def test_comp_backbone_core_001_registry_schema_roundtrip() -> None:
    """ID: COMP_BACKBONE_CORE_001_registry_schema_roundtrip."""
    ao = _base_ao()
    out = define_components(ao, opts=_registry_options(), validate=True)
    decoded = read_components(out)
    assert decoded == _registry_options().registry
    assert out.as_dataset(copy="none").attrs["tal"]["ext"]["components"]["version"] == 1
    assert list(out.as_dataset(copy="none").attrs["tal"]["ext"]["components"]["registry"]) == ["position", "heading"]


def test_comp_backbone_core_002_registry_rejects_unknown_core_dim() -> None:
    """ID: COMP_BACKBONE_CORE_002_registry_rejects_unknown_core_dim."""
    ao = _base_ao()
    opts = ComponentRegistryOptions(registry={"bad": ComponentSpec(core_dim="row", labels=("x",), var="value")})
    with pytest.raises(ValueError, match="core_dim"):
        _ = define_components(ao, opts=opts, validate=True)


def test_comp_backbone_core_003_registry_rejects_overlapping_label_sets() -> None:
    """ID: COMP_BACKBONE_CORE_003_registry_rejects_overlapping_label_sets."""
    ao = _base_ao()
    opts = ComponentRegistryOptions(
        registry={
            "a": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value"),
            "b": ComponentSpec(core_dim="axis", labels=("y",), var="value"),
        }
    )
    with pytest.raises(ValueError, match="overlaps"):
        _ = define_components(ao, opts=opts, validate=True)


def test_comp_backbone_core_010_registry_rejects_non_serializable_labels() -> None:
    """ID: COMP_BACKBONE_CORE_010_registry_rejects_non_serializable_labels."""
    ao = _base_ao()
    opts = ComponentRegistryOptions(registry={"bad": ComponentSpec(core_dim="axis", labels=(object(),), var="value")})
    with pytest.raises(ValueError, match="JSON scalar"):
        _ = define_components(ao, opts=opts, validate=True)


def test_comp_backbone_core_011_accessor_define_registry_parity() -> None:
    """ID: COMP_BACKBONE_CORE_011_accessor_define_registry_parity."""
    ao = _base_ao()
    opts = _registry_options()
    out_functional = define_components(ao, opts=opts, validate=True)
    out_accessor = ao.components.define(opts=opts, validate=True)
    xr.testing.assert_identical(out_functional.as_dataset(copy="none"), out_accessor.as_dataset(copy="none"))
    assert out_accessor.components.registry() == opts.registry


def test_comp_backbone_core_012_structural_partial_label_loss_prunes_component_entry() -> None:
    """ID: COMP_BACKBONE_CORE_012_structural_partial_label_loss_prunes_component_entry."""
    ao = _base_ao()
    out = define_components(
        ao,
        opts=ComponentRegistryOptions(
            registry={
                "xy": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value"),
                "z": ComponentSpec(core_dim="axis", labels=("z",), var="value"),
            }
        ),
        validate=True,
    )
    trimmed = out.sel(axis=["x", "y"])
    assert trimmed.components.registry() == {
        "xy": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value")
    }


def test_comp_backbone_core_007_structural_rename_rewrites_registry_truthfully() -> None:
    """ID: COMP_BACKBONE_CORE_007_structural_rename_rewrites_registry_truthfully."""
    ao = define_components(_base_ao(), opts=_registry_options(), validate=True)
    renamed = ao.rename({"axis": "component"})
    assert renamed.components.registry() == {
        "position": ComponentSpec(core_dim="component", labels=("x", "y"), var="value"),
        "heading": ComponentSpec(core_dim="component", labels=("z",), var="value"),
    }


def test_comp_backbone_core_008_structural_drop_prunes_registry_truthfully() -> None:
    """ID: COMP_BACKBONE_CORE_008_structural_drop_prunes_registry_truthfully."""
    ao = define_components(
        _base_ao(),
        opts=ComponentRegistryOptions(
            registry={
                "pose": ComponentSpec(core_dim="axis", labels=("x", "y", "z"), var="value"),
            }
        ),
        validate=True,
    )
    dropped = ao.drop_vars("value")
    assert dropped.components.registry() == {}
    ext = dropped.as_dataset(copy="none").attrs.get("tal", {}).get("ext", {})
    assert "components" not in ext


def test_comp_backbone_core_013_define_replace_false_merge_policy_deterministic() -> None:
    """ID: COMP_BACKBONE_CORE_013_define_replace_false_merge_policy_deterministic."""
    ao = _base_ao()
    first = define_components(
        ao,
        opts=ComponentRegistryOptions(
            registry={"position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value")}
        ),
        validate=True,
    )
    second = define_components(
        first,
        opts=ComponentRegistryOptions(
            registry={"heading": ComponentSpec(core_dim="axis", labels=("z",), var="value")},
            replace=False,
        ),
        validate=True,
    )
    assert list(second.components.registry().keys()) == ["position", "heading"]
    with pytest.raises(ValueError, match="already exists"):
        _ = define_components(
            second,
            opts=ComponentRegistryOptions(
                registry={"position": ComponentSpec(core_dim="axis", labels=("x",), var="value")},
                replace=False,
            ),
            validate=True,
        )


def test_comp_backbone_core_014_structural_unsupported_version_fails_closed() -> None:
    """ID: COMP_BACKBONE_CORE_014_structural_unsupported_version_fails_closed."""
    ao = define_components(_base_ao(), opts=_registry_options(), validate=True)
    bad_ds = ao.as_dataset(copy="none").copy(deep=True)
    bad_ds.attrs["tal"]["ext"]["components"]["version"] = 2
    bad_ao = AnalysisObject._from_unvalidated(bad_ds)
    with pytest.raises(ValueError, match="components.rewrite: tal.ext.components.version"):
        _ = bad_ao.rename({"axis": "component"})


def test_comp_backbone_core_015_structural_var_rename_rewrites_component_var() -> None:
    """ID: COMP_BACKBONE_CORE_015_structural_var_rename_rewrites_component_var."""
    ao = define_components(_base_ao(), opts=_registry_options(), validate=True)
    renamed = ao.rename({"value": "value2"})
    assert renamed.components.registry() == {
        "position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value2"),
        "heading": ComponentSpec(core_dim="axis", labels=("z",), var="value2"),
    }


def test_comp_backbone_core_016_version_type_strict_rejects_bool_true() -> None:
    """ID: COMP_BACKBONE_CORE_016_version_type_strict_rejects_bool_true."""
    ao = define_components(_base_ao(), opts=_registry_options(), validate=True)
    bad_ds = ao.as_dataset(copy="none").copy(deep=True)
    bad_ds.attrs["tal"]["ext"]["components"]["version"] = True
    bad_ao = AnalysisObject._from_unvalidated(bad_ds)
    with pytest.raises(ValueError, match="components.read: tal.ext.components.version"):
        _ = read_components(bad_ao)


def test_comp_backbone_core_017_structural_non_mapping_components_payload_fails_closed() -> None:
    """ID: COMP_BACKBONE_CORE_017_structural_non_mapping_components_payload_fails_closed."""
    ao = define_components(_base_ao(), opts=_registry_options(), validate=True)
    bad_ds = ao.as_dataset(copy="none").copy(deep=True)
    bad_ds.attrs["tal"]["ext"]["components"] = "malformed"
    bad_ao = AnalysisObject._from_unvalidated(bad_ds)
    with pytest.raises(ValueError, match="components.rewrite: tal.ext.components must be a mapping"):
        _ = bad_ao.rename({"axis": "component"})
