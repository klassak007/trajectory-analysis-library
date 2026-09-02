from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import (
    AnalysisObject,
    ComponentComposeOptions,
    ComponentExtractOptions,
    ComponentRegistryOptions,
    ComponentSpec,
    compose_components,
    define_components,
    extract_components,
    merge_schema,
    read_components,
)


def _registry_options() -> ComponentRegistryOptions:
    return ComponentRegistryOptions(
        registry={
            "position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value"),
            "heading": ComponentSpec(core_dim="axis", labels=("z",), var="value"),
        }
    )


def _multi_registry_options() -> ComponentRegistryOptions:
    return ComponentRegistryOptions(
        registry={
            "position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value"),
            "heading": ComponentSpec(core_dim="axis", labels=("z",), var="value_b"),
        }
    )


def _base_ao() -> AnalysisObject:
    values = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    ds = xr.Dataset(
        {"value": (("trial", "sample", "axis"), values)},
        coords={
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "sample": np.asarray([0, 1, 2], dtype=np.int64),
            "axis": np.asarray(["x", "y", "z"], dtype=object),
            "phase": (("trial", "sample"), np.asarray([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=float)),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="phase",
        validate=True,
    )
    return define_components(ao, opts=_registry_options(), validate=True)


def _multi_var_ao() -> AnalysisObject:
    ds = _base_ao().as_dataset(copy="none").copy(deep=True)
    ds["value_b"] = ds["value"] * -1.0
    ao = AnalysisObject._from_validated(ds)
    return define_components(ao, opts=_multi_registry_options(), validate=True)


def test_comp_backbone_core_027_compose_registry_order_and_label_order_deterministic() -> None:
    """ID: COMP_BACKBONE_CORE_027_compose_registry_order_and_label_order_deterministic."""
    base = _base_ao()
    extracted = extract_components(base)
    reversed_map = {"heading": extracted["heading"], "position": extracted["position"]}
    composed = compose_components(
        reversed_map,
        opts=ComponentComposeOptions(registry=_registry_options().registry),
        validate=True,
    )
    assert tuple(composed.as_dataset(copy="none").get_index("axis").tolist()) == ("x", "y", "z")
    xr.testing.assert_allclose(composed.as_dataset(copy="none")["value"], base.as_dataset(copy="none")["value"])


def test_comp_backbone_core_028_compose_name_set_must_match_registry_fail_closed() -> None:
    """ID: COMP_BACKBONE_CORE_028_compose_name_set_must_match_registry_fail_closed."""
    base = _base_ao()
    extracted = extract_components(base)
    registry = _registry_options().registry
    with pytest.raises(ValueError, match="must exactly match opts.registry keys"):
        _ = compose_components(
            {"position": extracted["position"]},
            opts=ComponentComposeOptions(registry=registry),
            validate=True,
        )
    with pytest.raises(ValueError, match="must exactly match opts.registry keys"):
        _ = compose_components(
            {"position": extracted["position"], "heading": extracted["heading"], "extra": extracted["heading"]},
            opts=ComponentComposeOptions(registry=registry),
            validate=True,
        )


def test_comp_backbone_core_029_compose_requires_shared_declared_roles_and_single_numeric_var() -> None:
    """ID: COMP_BACKBONE_CORE_029_compose_requires_shared_declared_roles_and_single_numeric_var."""
    base = _base_ao()
    extracted = extract_components(base)

    no_roles_ds = merge_schema(extracted["position"].as_dataset(copy="none"), patch={"core": {"roles": None}}, validate=False)
    no_roles = AnalysisObject._from_unvalidated(no_roles_ds)
    with pytest.raises(ValueError, match="declared roles with sequence_dim are required"):
        _ = compose_components(
            {"position": no_roles, "heading": extracted["heading"]},
            opts=ComponentComposeOptions(registry=_registry_options().registry),
            validate=True,
        )

    multi_var_ds = extracted["position"].as_dataset(copy="none").copy(deep=True)
    multi_var_ds["value_b"] = multi_var_ds["value"] * 2.0
    multi_var = AnalysisObject._from_unvalidated(multi_var_ds)
    with pytest.raises(ValueError, match="requires exactly one data variable when spec.var is not set"):
        _ = compose_components(
            {"position": multi_var},
            opts=ComponentComposeOptions(
                registry={"position": ComponentSpec(core_dim="axis", labels=("x", "y"), var=None)}
            ),
            validate=True,
        )


def test_comp_backbone_core_030_compose_requires_exact_component_label_set_and_unique_coords() -> None:
    """ID: COMP_BACKBONE_CORE_030_compose_requires_exact_component_label_set_and_unique_coords."""
    base = _base_ao()
    extracted = extract_components(base, opts=ComponentExtractOptions(names=("position",)))
    position = extracted["position"]
    with pytest.raises(ValueError, match="labels must exactly match declared labels"):
        _ = compose_components(
            {"position": position.sel(axis=["x"], validate=True)},
            opts=ComponentComposeOptions(
                registry={"position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value")}
            ),
            validate=True,
        )

    dup_ds = position.as_dataset(copy="none").copy(deep=True)
    dup_ds = dup_ds.assign_coords({"axis": np.asarray(["x", "x"], dtype=object)})
    duplicate_labels = AnalysisObject._from_unvalidated(dup_ds)
    with pytest.raises(ValueError, match="must be unique"):
        _ = compose_components(
            {"position": duplicate_labels},
            opts=ComponentComposeOptions(
                registry={"position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value")}
            ),
            validate=True,
        )


def test_comp_backbone_core_031_compose_functional_and_accessor_parity() -> None:
    """ID: COMP_BACKBONE_CORE_031_compose_functional_and_accessor_parity."""
    base = _base_ao()
    extracted = extract_components(base)
    opts = ComponentComposeOptions(registry=_registry_options().registry)
    functional = compose_components(extracted, opts=opts, validate=True)
    accessor = base.components.compose(extracted, opts=opts, validate=True)
    xr.testing.assert_identical(functional.as_dataset(copy="none"), accessor.as_dataset(copy="none"))


def test_comp_backbone_core_032_extract_then_compose_roundtrip_selected_components() -> None:
    """ID: COMP_BACKBONE_CORE_032_extract_then_compose_roundtrip_selected_components."""
    base = _base_ao()
    registry = {"position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value")}
    extracted = extract_components(base, opts=ComponentExtractOptions(names=("position",)))
    composed = compose_components(extracted, opts=ComponentComposeOptions(registry=registry), validate=True)
    assert tuple(composed.as_dataset(copy="none").get_index("axis").tolist()) == ("x", "y")
    xr.testing.assert_allclose(composed.as_dataset(copy="none")["value"], base.as_dataset(copy="none")["value"].sel(axis=["x", "y"]))
    assert read_components(composed) == registry


def test_comp_backbone_core_033_compose_then_extract_roundtrip_component_equivalence() -> None:
    """ID: COMP_BACKBONE_CORE_033_compose_then_extract_roundtrip_component_equivalence."""
    base = _base_ao()
    extracted = extract_components(base)
    composed = compose_components(
        extracted,
        opts=ComponentComposeOptions(registry=_registry_options().registry),
        validate=True,
    )
    roundtrip = extract_components(composed)
    for name in ("position", "heading"):
        xr.testing.assert_identical(roundtrip[name].as_dataset(copy="none"), extracted[name].as_dataset(copy="none"))


def test_comp_backbone_core_034_compose_output_var_single_var_policy_and_registry_truthfulness() -> None:
    """ID: COMP_BACKBONE_CORE_034_compose_output_var_single_var_policy_and_registry_truthfulness."""
    base = _base_ao()
    composed = compose_components(
        extract_components(base),
        opts=ComponentComposeOptions(registry=_registry_options().registry, output_var="component_value"),
        validate=True,
    )
    assert list(composed.as_dataset(copy="none").data_vars) == ["component_value"]
    assert read_components(composed) == {
        "position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="component_value"),
        "heading": ComponentSpec(core_dim="axis", labels=("z",), var="component_value"),
    }

    multi = _multi_var_ao()
    with pytest.raises(ValueError, match="opts.output_var requires composed output to have exactly one data variable"):
        _ = compose_components(
            extract_components(multi),
            opts=ComponentComposeOptions(registry=_multi_registry_options().registry, output_var="renamed"),
            validate=True,
        )


def test_comp_backbone_core_035_compose_non_dim_core_coord_fails_closed_owner_error() -> None:
    """ID: COMP_BACKBONE_CORE_035_compose_non_dim_core_coord_fails_closed_owner_error."""
    base = _base_ao()
    extracted = extract_components(base)
    malformed_ds = extracted["position"].as_dataset(copy="none").rename({"axis": "axis_dim"})
    malformed_ds = malformed_ds.assign_coords({"axis": ("axis_dim", np.asarray(["x", "y"], dtype=object))})
    malformed = AnalysisObject._from_unvalidated(malformed_ds)
    with pytest.raises(ValueError, match=r"components\.compose: .*axis.*"):
        _ = compose_components(
            {"position": malformed, "heading": extracted["heading"]},
            opts=ComponentComposeOptions(registry=_registry_options().registry),
            validate=True,
        )
