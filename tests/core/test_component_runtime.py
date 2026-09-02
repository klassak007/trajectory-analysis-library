from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import pytest
import xarray as xr

from tal.core import (
    AnalysisObject,
    ComponentExtractOptions,
    ComponentPatchOptions,
    ComponentRegistryOptions,
    ComponentSpec,
    define_components,
    extract_components,
    patch_components,
    read_components,
)
from tal.core.schema_read import read_roles


class _DuplicateItemsMapping(Mapping[str, object]):
    def __init__(self, items: list[tuple[str, object]]) -> None:
        self._items = list(items)

    def __iter__(self) -> Iterator[str]:
        for key, _ in self._items:
            yield key

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, key: str) -> object:
        for item_key, value in reversed(self._items):
            if item_key == key:
                return value
        raise KeyError(key)

    def items(self) -> list[tuple[str, object]]:
        return list(self._items)


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
    return define_components(
        ao,
        opts=ComponentRegistryOptions(
            registry={
                "position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value"),
                "heading": ComponentSpec(core_dim="axis", labels=("z",), var="value_b"),
            }
        ),
        validate=True,
    )


def _registry_options() -> ComponentRegistryOptions:
    return ComponentRegistryOptions(
        registry={
            "position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value"),
            "heading": ComponentSpec(core_dim="axis", labels=("z",), var="value"),
        }
    )


def _offset_component(component: AnalysisObject, *, delta: float) -> AnalysisObject:
    ds = component.as_dataset(copy="none").copy(deep=True)
    name = str(next(iter(ds.data_vars)))
    ds[name] = ds[name] + delta
    return AnalysisObject._from_unvalidated(ds)


def test_comp_backbone_core_018_extract_registry_order_and_name_subset_deterministic() -> None:
    """ID: COMP_BACKBONE_CORE_018_extract_registry_order_and_name_subset_deterministic."""
    ao = _base_ao()
    full = extract_components(ao)
    assert list(full.keys()) == ["position", "heading"]
    subset = extract_components(
        ao,
        opts=ComponentExtractOptions(names=("heading", "position"), output_var="component_value"),
    )
    assert list(subset.keys()) == ["heading", "position"]
    for out in subset.values():
        assert list(out.as_dataset(copy="none").data_vars) == ["component_value"]


def test_comp_backbone_core_019_extract_unknown_component_name_fail_closed() -> None:
    """ID: COMP_BACKBONE_CORE_019_extract_unknown_component_name_fail_closed."""
    ao = _base_ao()
    with pytest.raises(ValueError, match="unknown component names"):
        _ = extract_components(ao, opts=ComponentExtractOptions(names=("missing",)))


def test_comp_backbone_core_020_extract_preserves_schema_truthfulness_and_label_order() -> None:
    """ID: COMP_BACKBONE_CORE_020_extract_preserves_schema_truthfulness_and_label_order."""
    ao = _base_ao()
    extracted = extract_components(ao, opts=ComponentExtractOptions(names=("position",)))
    position = extracted["position"]
    labels = tuple(position.as_dataset(copy="none").get_index("axis").tolist())
    assert labels == ("x", "y")
    assert read_components(position) == {
        "position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="value")
    }
    roles_declared, sequence_dim, batch_dims, core_dims = read_roles(position.as_dataset(copy="none"))
    assert roles_declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ("axis",)


def test_comp_backbone_core_026_extract_output_var_preserves_component_registry_var_truthfully() -> None:
    """ID: COMP_BACKBONE_CORE_026_extract_output_var_preserves_component_registry_var_truthfully."""
    ao = _base_ao()
    extracted = extract_components(
        ao,
        opts=ComponentExtractOptions(names=("position",), output_var="component_value"),
    )
    position = extracted["position"]
    assert list(position.as_dataset(copy="none").data_vars) == ["component_value"]
    assert read_components(position) == {
        "position": ComponentSpec(core_dim="axis", labels=("x", "y"), var="component_value")
    }


def test_comp_backbone_core_021_patch_functional_and_accessor_parity() -> None:
    """ID: COMP_BACKBONE_CORE_021_patch_functional_and_accessor_parity."""
    ao = _base_ao()
    position_patch = _offset_component(extract_components(ao, opts=ComponentExtractOptions(names=("position",)))["position"], delta=100.0)
    functional = patch_components(
        ao,
        {"position": position_patch},
        opts=ComponentPatchOptions(on_overlap="error"),
        validate=True,
    )
    accessor = ao.components.patch(
        {"position": position_patch},
        opts=ComponentPatchOptions(on_overlap="error"),
        validate=True,
    )
    xr.testing.assert_identical(functional.as_dataset(copy="none"), accessor.as_dataset(copy="none"))


def test_comp_backbone_core_022_patch_unknown_component_name_fail_closed() -> None:
    """ID: COMP_BACKBONE_CORE_022_patch_unknown_component_name_fail_closed."""
    ao = _base_ao()
    position_patch = extract_components(ao, opts=ComponentExtractOptions(names=("position",)))["position"]
    with pytest.raises(ValueError, match="unknown component name"):
        _ = patch_components(
            ao,
            {"missing": position_patch},
            opts=ComponentPatchOptions(on_overlap="error"),
            validate=True,
        )


def test_comp_backbone_core_023_patch_requires_exact_component_label_set() -> None:
    """ID: COMP_BACKBONE_CORE_023_patch_requires_exact_component_label_set."""
    ao = _base_ao()
    position_patch = extract_components(ao, opts=ComponentExtractOptions(names=("position",)))["position"]
    missing_label_patch = position_patch.sel(axis=["x"], validate=True)
    with pytest.raises(ValueError, match="must exactly match declared labels"):
        _ = patch_components(
            ao,
            {"position": missing_label_patch},
            opts=ComponentPatchOptions(on_overlap="error"),
            validate=True,
        )


def test_comp_backbone_core_024_patch_overlap_policy_error_vs_replace_deterministic() -> None:
    """ID: COMP_BACKBONE_CORE_024_patch_overlap_policy_error_vs_replace_deterministic."""
    ao = _base_ao()
    position_patch = extract_components(ao, opts=ComponentExtractOptions(names=("position",)))["position"]
    patch_a = _offset_component(position_patch, delta=10.0)
    patch_b = _offset_component(position_patch, delta=20.0)
    duplicate_mapping = _DuplicateItemsMapping([("position", patch_a), ("position", patch_b)])
    with pytest.raises(ValueError, match="overlapping patch labels"):
        _ = patch_components(
            ao,
            duplicate_mapping,
            opts=ComponentPatchOptions(on_overlap="error"),
            validate=True,
        )
    replaced = patch_components(
        ao,
        duplicate_mapping,
        opts=ComponentPatchOptions(on_overlap="replace"),
        validate=True,
    )
    expected = patch_b.as_dataset(copy="none")["value"]
    xr.testing.assert_allclose(replaced.as_dataset(copy="none")["value"].sel(axis=["x", "y"]), expected)


def test_comp_backbone_core_025_extract_then_patch_roundtrip_for_selected_components() -> None:
    """ID: COMP_BACKBONE_CORE_025_extract_then_patch_roundtrip_for_selected_components."""
    ao = _base_ao()
    extracted = extract_components(ao, opts=ComponentExtractOptions(names=("position",)))
    position_patch = _offset_component(extracted["position"], delta=-50.0)
    patched = patch_components(
        ao,
        {"position": position_patch},
        opts=ComponentPatchOptions(on_overlap="error", output_var="value_new"),
        validate=True,
    )
    xr.testing.assert_allclose(patched.as_dataset(copy="none")["value_new"].sel(axis=["x", "y"]), position_patch.as_dataset(copy="none")["value"])
    xr.testing.assert_allclose(patched.as_dataset(copy="none")["value_new"].sel(axis=["z"]), ao.as_dataset(copy="none")["value"].sel(axis=["z"]))


def test_patch_components_output_var_rejects_multi_target_runtime() -> None:
    ao = _multi_var_ao()
    extracted = extract_components(ao)
    with pytest.raises(ValueError, match="opts.output_var requires all patched components to target one base variable"):
        _ = patch_components(
            ao,
            {"position": extracted["position"], "heading": extracted["heading"]},
            opts=ComponentPatchOptions(on_overlap="error", output_var="renamed"),
            validate=True,
        )


def test_comp_backbone_core_036_patch_non_dim_core_coord_fails_closed_owner_error() -> None:
    """ID: COMP_BACKBONE_CORE_036_patch_non_dim_core_coord_fails_closed_owner_error."""
    ao = _base_ao()
    position_patch = extract_components(ao, opts=ComponentExtractOptions(names=("position",)))["position"]
    malformed_ds = position_patch.as_dataset(copy="none").rename({"axis": "axis_dim"})
    malformed_ds = malformed_ds.assign_coords({"axis": ("axis_dim", np.asarray(["x", "y"], dtype=object))})
    malformed_patch = AnalysisObject._from_unvalidated(malformed_ds)
    with pytest.raises(ValueError, match=r"components\.patch: .*axis.*"):
        _ = patch_components(
            ao,
            {"position": malformed_patch},
            opts=ComponentPatchOptions(on_overlap="error"),
            validate=True,
        )
