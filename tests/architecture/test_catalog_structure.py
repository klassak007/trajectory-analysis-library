from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture._budget import file_loc, function_lengths


def test_arch_cat_p11a_001_catalog_implementation_resides_outside_tal_core() -> None:
    """ID: ARCH_CAT_P11A_001_catalog_implementation_resides_outside_tal_core."""
    assert Path("tal/catalog/catalog.py").exists()
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "class Catalog" not in text


def test_arch_cat_p11a_002_no_core_import_dependency_on_tal_catalog() -> None:
    """ID: ARCH_CAT_P11A_002_no_core_import_dependency_on_tal_catalog."""
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tal.catalog" not in text
        assert "from ..catalog" not in text
        assert "from tal.catalog" not in text


def test_arch_cat_p11a_003_catalog_surface_methods_remain_thin_and_owner_delegated() -> None:
    """ID: ARCH_CAT_P11A_003_catalog_surface_methods_remain_thin_and_owner_delegated."""
    text = Path("tal/catalog/catalog.py").read_text(encoding="utf-8")
    assert "coerce_catalog_init_options(" in text
    assert "detect_catalog_backend(" in text
    assert "normalize_catalog_payload(" in text
    assert "resolve_catalog_batch_dim(" in text
    assert "normalize_label_selector(" in text
    assert "normalize_position_selector(" in text
    assert "select_by_labels(" in text
    assert "select_by_positions(" in text


def test_arch_cat_p11a_004_catalog_grouping_policy_is_constructor_scoped_not_mutated_post_construction() -> None:
    """ID: ARCH_CAT_P11A_004_catalog_grouping_policy_is_constructor_scoped_not_mutated_post_construction."""
    text = Path("tal/catalog/catalog.py").read_text(encoding="utf-8")
    assert "def set_group(" not in text
    assert "def regroup(" not in text
    assert "@batch_dim.setter" not in text


def test_arch_cat_p11a_005_catalog_isel_index_normalization_policy_is_centralized() -> None:
    """ID: ARCH_CAT_P11A_005_catalog_isel_index_normalization_policy_is_centralized."""
    text = Path("tal/catalog/selection.py").read_text(encoding="utf-8")
    assert "def _normalize_effective_position_selector(" in text
    assert "size=_group_axis_size(state)" in text
    assert "_normalize_effective_position_selector(" in text
    assert "_datatree_position_plan(tree, selector=normalized)" in text
    assert "positions=positions" in text
    assert "_datatree_labels_at_positions(children, positions=positions)" in text
    assert "idx + size if idx < 0 else idx" in text


def test_arch_cat_p11a_006_datatree_root_projection_has_single_selection_owner() -> None:
    """ID: ARCH_CAT_P11A_006_datatree_root_projection_has_single_selection_owner."""
    selection = Path("tal/catalog/selection.py").read_text(encoding="utf-8")
    metadata = Path("tal/catalog/metadata_domain.py").read_text(encoding="utf-8")
    module = ast.parse(selection)
    owners = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_project_datatree_root_dataset"
    ]
    assert len(owners) == 1
    consumer = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_datatree_select_labels"
    )
    consumer_calls = {
        node.func.id
        for node in ast.walk(consumer)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_project_datatree_root_dataset" in consumer_calls
    calls = {
        node.func.attr
        for node in ast.walk(owners[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "isel" in calls
    assert "sel" not in calls
    assert "tree.to_dataset(inherit=False)" in selection
    assert "children[label].to_dataset(inherit=False)" in selection
    assert "_project_datatree_root_dataset(" not in metadata


def test_arch_cat_p11a_007_datatree_selection_refreshes_nonempty_extract_template() -> None:
    """ID: ARCH_CAT_P11A_007_datatree_selection_refreshes_nonempty_extract_template."""
    text = Path("tal/catalog/selection.py").read_text(encoding="utf-8")
    module = ast.parse(text)
    owner = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_selected_datatree_template"
    )
    owner_calls = {
        call.func.id
        for call in ast.walk(owner)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "datatree_extract_template" in owner_calls
    assert text.count("template=_selected_datatree_template(") == 2


def test_arch_cat_p11a_008_datatree_root_batch_alignment_has_single_backend_owner() -> None:
    """ID: ARCH_CAT_P11A_008_datatree_root_batch_alignment_has_single_backend_owner."""
    sources = {
        name: Path(f"tal/catalog/{name}.py").read_text(encoding="utf-8")
        for name in ("backends", "selection", "extract")
    }
    modules = {name: ast.parse(source) for name, source in sources.items()}
    owners = [
        node
        for node in modules["backends"].body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_datatree_root_batch_alignment"
    ]
    assert len(owners) == 1
    consumers = (
        ("backends", "datatree_extract_template"),
        ("selection", "_datatree_select_labels"),
        ("extract", "_validated_datatree_children"),
    )
    for module_name, function_name in consumers:
        consumer = next(
            node
            for node in modules[module_name].body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        calls = {
            call.func.id
            for call in ast.walk(consumer)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        assert "validate_datatree_root_batch_alignment" in calls
    message = "does not match DataTree group count"
    assert message in sources["backends"]
    assert message not in sources["selection"]
    assert message not in sources["extract"]


def test_cat_doc_p11a_001_catalog_location_optional_role_and_browse_only_semantics_documented() -> None:
    """ID: CAT_DOC_P11A_001_catalog_location_optional_role_and_browse_only_semantics_documented."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8")
    assert "# Catalog" in text
    assert "Catalog" in text
    assert "AnalysisObject" in text


def test_cat_doc_p11a_002_catalog_grouping_is_constructor_scoped_and_regroup_mutation_is_not_public() -> None:
    """ID: CAT_DOC_P11A_002_catalog_grouping_is_constructor_scoped_and_regroup_mutation_is_not_public."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8")
    assert "batch_dim" in text
    assert "post-construction regrouping" in text


def test_cat_doc_p11a_003_catalog_isel_normalized_duplicate_policy_documented() -> None:
    """ID: CAT_DOC_P11A_003_catalog_isel_normalized_duplicate_policy_documented."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8")
    assert "catalog.isel" in text
    assert "Unsupported selector" in text


def test_cat_doc_p11a_004_datatree_selection_root_metadata_semantics_documented() -> None:
    """ID: CAT_DOC_P11A_004_datatree_selection_root_metadata_semantics_documented."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8").lower()
    assert "root batch metadata" in text
    assert "child order" in text
    assert "catalog construction fails closed" in text


def test_catalog_owner_budget_and_schema_write_boundary() -> None:
    """Catalog modules stay within AGENTS budgets and avoid direct tal-attrs writes."""
    for path in sorted(Path("tal/catalog").glob("*.py")):
        assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
        for name, length in function_lengths(path).items():
            assert length <= 50, f"{path}:{name} exceeds function budget ({length} > 50)."
        text = path.read_text(encoding="utf-8")
        assert 'attrs["tal"]' not in text
        assert "attrs['tal']" not in text


def test_arch_cat_p11b_001_query_extract_reuse_shared_finalize_and_grouping_owners() -> None:
    """ID: ARCH_CAT_P11B_001_query_extract_reuse_shared_finalize_and_grouping_owners."""
    extract_text = Path("tal/catalog/extract.py").read_text(encoding="utf-8")
    query_text = Path("tal/catalog/query.py").read_text(encoding="utf-8")
    assert "finalize_with_schema" in extract_text
    assert "set_left_packed_validity_or_prune_from_size_coord" in extract_text
    assert "ensure_grouping_context_for_query_extract" in query_text


def test_arch_cat_p11b_002_single_query_planner_owner_enforced_no_duplicate_policy_helpers() -> None:
    """ID: ARCH_CAT_P11B_002_single_query_planner_owner_enforced_no_duplicate_policy_helpers."""
    planner_text = Path("tal/catalog/query_plan.py").read_text(encoding="utf-8")
    query_text = Path("tal/catalog/query.py").read_text(encoding="utf-8")
    catalog_text = Path("tal/catalog/catalog.py").read_text(encoding="utf-8")
    assert "def normalize_catalog_query_plan(" in planner_text
    assert "normalize_catalog_query_plan(" in query_text
    assert "normalize_catalog_query_plan(" not in catalog_text


def test_arch_cat_p11b_003_catalog_query_extract_logic_resides_outside_tal_core() -> None:
    """ID: ARCH_CAT_P11B_003_catalog_query_extract_logic_resides_outside_tal_core."""
    assert Path("tal/catalog/query.py").exists()
    assert Path("tal/catalog/extract.py").exists()
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "Catalog.query" not in text
        assert "Catalog.extract" not in text


def test_arch_cat_p11b_004_query_extract_do_not_depend_on_post_construction_group_mutators() -> None:
    """ID: ARCH_CAT_P11B_004_query_extract_do_not_depend_on_post_construction_group_mutators."""
    text = Path("tal/catalog/catalog.py").read_text(encoding="utf-8")
    assert "def set_group(" not in text
    assert "def regroup(" not in text
    assert "query(" in text
    assert "extract(" in text


def test_arch_cat_p11b_005_batch_dim_collision_validation_is_centralized_and_reused() -> None:
    """ID: ARCH_CAT_P11B_005_batch_dim_collision_validation_is_owned_in_catalog_backends_and_extract."""
    backends = Path("tal/catalog/backends.py").read_text(encoding="utf-8")
    extract = Path("tal/catalog/extract.py").read_text(encoding="utf-8")
    coords = Path("tal/catalog/extract_coords.py").read_text(encoding="utf-8")
    assert "def _validate_explicit_dataset_batch_dim(" in backends
    assert "def validate_batch_dim_role_compatibility(" in backends
    assert "def _first_child_dim_collision(" in backends
    assert "validate_batch_dim_role_compatibility(" in extract
    assert "def _require_nonconflicting_batch_dim(" not in extract
    assert "def _expand_child_row(" in coords
    assert "collides with child payload dims" in coords


def test_arch_cat_p11b_006_datatree_extract_payload_view_has_single_backend_owner() -> None:
    """ID: ARCH_CAT_P11B_006_datatree_extract_payload_view_has_single_backend_owner."""
    backends = Path("tal/catalog/backends.py").read_text(encoding="utf-8")
    extract = Path("tal/catalog/extract_coords.py").read_text(encoding="utf-8")
    backend_module = ast.parse(backends)
    extract_module = ast.parse(extract)
    owners = [
        node
        for node in backend_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "datatree_child_payload_dataset"
    ]
    assert len(owners) == 1
    template = next(
        node
        for node in backend_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "datatree_extract_template"
    )
    consumer = next(
        node
        for node in extract_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "plan_datatree_child_payloads"
    )
    for node in (template, consumer):
        calls = {
            call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        assert "datatree_child_payload_dataset" in calls
    template_source = ast.get_source_segment(backends, template)
    assert "copy(deep=False)" in template_source
    owner_source = ast.get_source_segment(backends, owners[0])
    assert "child.to_dataset(inherit=False)" in owner_source
    assert "child.parent" in owner_source
    assert "parent.to_dataset(inherit=False)" in owner_source
    assert "parent_ds.coords" in owner_source
    assert "_root_payload_coord(" in owner_source
    assert "_inherited_payload_indexes(" in owner_source
    assert "xr.Coordinates(" in owner_source
    index_owner = next(
        node
        for node in backend_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_inherited_payload_indexes"
    )
    index_owner_source = ast.get_source_segment(backends, index_owner)
    assert "parent.xindexes.group_by_index()" in index_owner_source
    root_coord_owner = next(
        node
        for node in backend_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_root_payload_coord"
    )
    root_coord_source = ast.get_source_segment(backends, root_coord_owner)
    assert "batch_dim not in coord.dims" in root_coord_source
    assert "coord.isel({batch_dim: batch_position}, drop=True)" in root_coord_source
    assert "batch_position=0" in template_source


def test_arch_cat_p11b_007_datatree_extract_concat_is_structural_and_lazy() -> None:
    """ID: ARCH_CAT_P11B_007_datatree_extract_concat_is_structural_and_lazy."""
    text = Path("tal/catalog/extract_coords.py").read_text(encoding="utf-8")
    orchestrator = Path("tal/catalog/extract.py").read_text(encoding="utf-8")
    module = ast.parse(text)
    owners = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_batch_coord_specs"
    ]
    assert len(owners) == 1
    consumer = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "plan_datatree_child_payloads"
    )
    calls = {
        call.func.id
        for call in ast.walk(consumer)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "_batch_coord_specs" in calls
    specs_source = ast.get_source_segment(text, owners[0])
    assert "_is_dimension_coord" in specs_source
    dimension_helper = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_is_dimension_coord"
    )
    dimension_source = ast.get_source_segment(text, dimension_helper)
    assert "name in ds.dims" in dimension_source
    assert "ds.coords[name].dims == (name,)" in dimension_source
    assert "xindexes" not in dimension_source
    concat = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "concat_datatree_rows"
    )
    concat_source = ast.get_source_segment(text, concat)
    assert 'coords="minimal"' in concat_source
    assert 'coords="different"' not in concat_source
    spec_calls = {
        call.func.id
        for call in ast.walk(owners[0])
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "_require_compatible_coord_dims" in spec_calls
    resolver = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolved_child_coord"
    )
    resolver_calls = {
        call.func.id
        for call in ast.walk(resolver)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "_require_compatible_coord_dims" in resolver_calls
    assert "_coord_in_template_order" in resolver_calls
    row_owner = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_expand_child_row"
    )
    collision_guards = [node for node in row_owner.body if isinstance(node, ast.Try)]
    assert len(collision_guards) == 1
    assert len(collision_guards[0].body) == 1
    guarded_calls = {
        call.func.id
        for call in ast.walk(collision_guards[0])
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    guarded_attrs = {
        call.func.attr
        for call in ast.walk(collision_guards[0])
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    assert guarded_attrs == {"expand_dims"}
    assert "_child_batch_coords" not in guarded_calls
    row_calls = {
        call.func.id
        for call in ast.walk(row_owner)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "_without_batch_dependent_coords" in row_calls
    assert "_child_batch_coords" in row_calls
    for delegated in (
        "plan_datatree_child_payloads(",
        "expand_datatree_rows(",
        "concat_datatree_rows(",
    ):
        assert delegated in orchestrator


def test_arch_cat_p11b_008_result_value_isolation_has_single_owner() -> None:
    """ID: ARCH_CAT_P11B_008_result_value_isolation_has_single_owner."""
    ownership_text = Path("tal/catalog/ownership.py").read_text(encoding="utf-8")
    extract_text = Path("tal/catalog/extract.py").read_text(encoding="utf-8")
    ownership_module = ast.parse(ownership_text)
    extract_module = ast.parse(extract_text)
    owners = [
        node
        for node in ownership_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "isolate_result_values"
    ]
    assert len(owners) == 1
    owner_source = ast.get_source_segment(ownership_text, owners[0])
    assert "coord.variable.copy(deep=True)" in owner_source
    assert "batch_dim not in coord.dims" in owner_source
    assert "_dtype_requires_nested_value_copy(coord.dtype)" in owner_source
    assert "_dtype_requires_nested_value_copy(variable.dtype)" in owner_source
    assert "ds.data_vars.items()" in owner_source
    assert "_copy_nested_value_array(" in owner_source
    assert "_copy_registered_xindexes(out, owner=owner)" in owner_source
    assert "_assign_unindexed_coords(out, detached_coords)" in owner_source
    assert "owner=owner" in owner_source
    coord_owner = next(
        node
        for node in ownership_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_assign_unindexed_coords"
    )
    coord_source = ast.get_source_segment(ownership_text, coord_owner)
    assert "xr.Coordinates(coords, indexes={})" in coord_source
    object_owner = next(
        node
        for node in ownership_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_deepcopy_object_array"
    )
    object_source = ast.get_source_segment(ownership_text, object_owner)
    assert 'dask="parallelized"' in object_source
    assert 'kwargs={"owner": owner, "name": name}' in object_source
    assert "_deepcopy_or_error(" in object_source
    consumers = (
        (extract_module, "_extract_datatree_payload"),
        (extract_module, "_extract_empty_datatree_payload"),
    )
    for module, function_name in consumers:
        consumer = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        calls = {
            call.func.id
            for call in ast.walk(consumer)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        assert "isolate_result_values" in calls


def test_arch_cat_p11b_013_payload_variable_topology_has_single_owner() -> None:
    """ID: ARCH_CAT_P11B_013_payload_variable_topology_has_single_owner."""
    coord_text = Path("tal/catalog/extract_coords.py").read_text(encoding="utf-8")
    extract_text = Path("tal/catalog/extract.py").read_text(encoding="utf-8")
    coord_module = ast.parse(coord_text)
    extract_module = ast.parse(extract_text)
    owners = [
        node
        for node in coord_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "canonicalize_selected_payload_vars"
    ]
    assert len(owners) == 1
    owner_calls = {
        call.func.id
        for call in ast.walk(owners[0])
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "_canonicalize_payload_vars" in owner_calls
    helper = next(
        node
        for node in coord_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_canonicalize_payload_vars"
    )
    helper_source = ast.get_source_segment(coord_text, helper)
    assert "set(dims) != set(target)" in helper_source
    assert "incompatible dimensions across DataTree groups" in helper_source
    assert ".transpose(*target)" in helper_source
    assert ".variable" in helper_source
    assert "ds.assign(updates)" in helper_source
    consumer = next(
        node
        for node in extract_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_select_datatree_payload_vars"
    )
    consumer_calls = {
        call.func.id
        for call in ast.walk(consumer)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "canonicalize_selected_payload_vars" in consumer_calls
    orchestrator = next(
        node
        for node in extract_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_extract_datatree_payload"
    )
    orchestrator_calls = {
        call.func.id
        for call in ast.walk(orchestrator)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "_select_datatree_payload_vars" in orchestrator_calls


def test_arch_cat_p11b_011_result_isolation_follows_metadata_promotion() -> None:
    """ID: ARCH_CAT_P11B_011_result_isolation_runs_after_metadata_promotion."""
    coord_text = Path("tal/catalog/extract_coords.py").read_text(encoding="utf-8")
    extract_text = Path("tal/catalog/extract.py").read_text(encoding="utf-8")
    coord_module = ast.parse(coord_text)
    extract_module = ast.parse(extract_text)
    concat = next(
        node
        for node in coord_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "concat_datatree_rows"
    )
    concat_calls = {
        call.func.id
        for call in ast.walk(concat)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "isolate_result_values" not in concat_calls
    owner = next(
        node
        for node in extract_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_extract_datatree_payload"
    )
    owner_source = ast.get_source_segment(extract_text, owner)
    assert owner_source.index("_promote_datatree_metadata(") < owner_source.index(
        "isolate_result_values("
    )


def test_arch_cat_p11b_012_dataset_extract_has_explicit_ownership_boundary() -> None:
    """ID: ARCH_CAT_P11B_012_dataset_extract_has_explicit_ownership_boundary."""
    ownership_text = Path("tal/catalog/ownership.py").read_text(encoding="utf-8")
    extract_text = Path("tal/catalog/extract.py").read_text(encoding="utf-8")
    ownership_module = ast.parse(ownership_text)
    extract_module = ast.parse(extract_text)
    owners = [
        node
        for node in ownership_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "isolate_dataset_result_values"
    ]
    assert len(owners) == 1
    owner_source = ast.get_source_segment(ownership_text, owners[0])
    assert "_copy_dataset_result_array(variable," in owner_source
    assert "_copy_dataset_result_array(coord," in owner_source
    copy_owner = next(
        node
        for node in ownership_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_copy_dataset_result_array"
    )
    copy_source = ast.get_source_segment(ownership_text, copy_owner)
    assert "pd.CategoricalDtype" in copy_source
    assert "_deepcopy_categorical_array(" in copy_source
    assert "_dtype_has_object_values(array.dtype)" in copy_source
    assert "deep=array.chunks is None" in copy_source
    consumer = next(
        node
        for node in extract_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_extract_dataset_payload"
    )
    calls = {
        call.func.id
        for call in ast.walk(consumer)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "isolate_dataset_result_values" in calls
    assert "isolate_result_metadata" in calls


def test_arch_cat_p11b_009_result_metadata_isolation_has_single_owner() -> None:
    """ID: ARCH_CAT_P11B_009_result_metadata_isolation_has_single_owner."""
    ownership_text = Path("tal/catalog/ownership.py").read_text(encoding="utf-8")
    extract_text = Path("tal/catalog/extract.py").read_text(encoding="utf-8")
    ownership_module = ast.parse(ownership_text)
    owners = [
        node
        for node in ownership_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "isolate_result_metadata"
    ]
    assert len(owners) == 1
    owner_source = ast.get_source_segment(ownership_text, owners[0])
    assert "ds.copy(deep=False)" in owner_source
    assert "_deepcopy_or_error(ds.attrs" in owner_source
    assert "_deepcopy_or_error(ds.encoding" in owner_source
    assert "_deepcopy_or_error(" in owner_source
    assert extract_text.count("isolate_result_metadata(") == 3
    extract_module = ast.parse(extract_text)
    for function_name in (
        "_extract_dataset_payload",
        "_extract_datatree_payload",
        "_extract_empty_datatree_payload",
    ):
        consumer = next(
            node
            for node in extract_module.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        calls = {
            call.func.id
            for call in ast.walk(consumer)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        assert "isolate_result_metadata" in calls


def test_arch_cat_p11b_014_catalog_ownership_has_single_domain_owner() -> None:
    """ID: ARCH_CAT_P11B_014_catalog_ownership_has_single_domain_owner."""
    ownership_text = Path("tal/catalog/ownership.py").read_text(encoding="utf-8")
    catalog_text = Path("tal/catalog/catalog.py").read_text(encoding="utf-8")
    extract_text = Path("tal/catalog/extract.py").read_text(encoding="utf-8")
    coord_text = Path("tal/catalog/extract_coords.py").read_text(encoding="utf-8")
    module = ast.parse(ownership_text)
    owners = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "copy_catalog_payload_for_public_access"
    ]
    assert len(owners) == 1
    assert "copy_catalog_payload_for_public_access(" in catalog_text
    assert "self._state.data.copy(deep=True)" not in catalog_text
    assert "from .ownership import" in extract_text
    assert "deepcopy" not in coord_text
    kernel = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_deepcopy_or_error"
    )
    kernel_source = ast.get_source_segment(ownership_text, kernel)
    assert "except Exception as exc" in kernel_source
    assert "raise ValueError" in kernel_source
    assert "from exc" in kernel_source
    dtype_owners = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dtype_has_object_values"
    ]
    assert len(dtype_owners) == 1
    dtype_source = ast.get_source_segment(ownership_text, dtype_owners[0])
    assert 'getattr(dtype, "hasobject", False)' in dtype_source
    nested_dtype_owners = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dtype_requires_nested_value_copy"
    ]
    assert len(nested_dtype_owners) == 1
    nested_dtype_source = ast.get_source_segment(
        ownership_text,
        nested_dtype_owners[0],
    )
    assert "pd.CategoricalDtype" in nested_dtype_source
    assert "_dtype_has_object_values(dtype)" in nested_dtype_source
    categorical_owners = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_deepcopy_categorical_array"
    ]
    assert len(categorical_owners) == 1
    categorical_source = ast.get_source_segment(
        ownership_text,
        categorical_owners[0],
    )
    assert "_deepcopy_pandas_index(" in categorical_source
    assert "pd.Categorical.from_codes(" in categorical_source
    index_owners = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_copy_registered_xindexes"
    ]
    assert len(index_owners) == 1
    index_source = ast.get_source_segment(ownership_text, index_owners[0])
    assert "ds.xindexes.group_by_index()" in index_source
    assert "xr.Coordinates(" in index_source
    assert "drop_indexes(" in index_source
    assert "ds.variables[name]" in index_source
    assert "_restore_index_coordinate_metadata(" in index_source
    assert ownership_text.count("_copy_registered_xindexes(out, owner=owner)") == 2
    assert "_copy_registered_xindex(" in index_source
    assert "_detach_index_coord_variables(" in index_source
    assert "owner=owner" in index_source
    metadata_owner = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_restore_index_coordinate_metadata"
    )
    metadata_source = ast.get_source_segment(ownership_text, metadata_owner)
    assert "dict(source.attrs)" in metadata_source
    assert "dict(source.encoding)" in metadata_source
    index_copy_owner = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_copy_registered_xindex"
    )
    index_copy_source = ast.get_source_segment(ownership_text, index_copy_owner)
    assert "_rebuild_pandas_xindex(" in index_copy_source
    assert "index.copy(deep=True)" in index_copy_source
    assert "object-bearing" in index_copy_source
    pandas_owner = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_deepcopy_pandas_index"
    )
    pandas_source = ast.get_source_segment(ownership_text, pandas_owner)
    assert "pd.CategoricalIndex" in pandas_source
    assert "_deepcopy_index_object_values(" in pandas_source


def test_arch_cat_p11b_010_extract_coord_metadata_reconciliation_has_metadata_owner() -> None:
    """ID: ARCH_CAT_P11B_010_extract_coord_metadata_reconciliation_has_metadata_owner."""
    metadata_text = Path("tal/catalog/metadata_domain.py").read_text(encoding="utf-8")
    extract_text = Path("tal/catalog/extract.py").read_text(encoding="utf-8")
    metadata_module = ast.parse(metadata_text)
    owner = next(
        node
        for node in metadata_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_extract_metadata_promotion"
    )
    owner_calls = {
        call.func.id
        for call in ast.walk(owner)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "collect_extract_metadata_columns" in owner_calls
    assert "_extract_metadata_projection_state" in owner_calls
    assert "_metadata_promotion_is_disabled" in owner_calls
    assert "promote_extract_metadata" in owner_calls
    owner_source = ast.get_source_segment(metadata_text, owner)
    assert "protected_coord_names" in owner_source
    assert "protected.union(" in owner_source
    assert owner_source.index("_metadata_promotion_is_disabled(") < owner_source.index(
        "collect_extract_metadata_columns("
    )
    extract_module = ast.parse(extract_text)
    consumer = next(
        node
        for node in extract_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_promote_datatree_metadata"
    )
    consumer_calls = {
        call.func.id
        for call in ast.walk(consumer)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert consumer_calls == {"apply_extract_metadata_promotion"}
    orchestrator = next(
        node
        for node in extract_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_extract_datatree_payload"
    )
    orchestrator_calls = {
        call.func.id
        for call in ast.walk(orchestrator)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "structural_scalar_metadata_coord_names" in orchestrator_calls
    assert "_schema_owned_extract_coord_names" in orchestrator_calls
    empty_orchestrator = next(
        node
        for node in extract_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_extract_empty_datatree_payload"
    )
    empty_calls = {
        call.func.id
        for call in ast.walk(empty_orchestrator)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "expand_empty_datatree_template" in empty_calls
    assert "_promote_datatree_metadata" in empty_calls
    assert "_schema_owned_extract_coord_names" in empty_calls
    attrs_owner = next(
        node
        for node in metadata_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_datatree_attrs_values"
    )
    attrs_source = ast.get_source_segment(metadata_text, attrs_owner)
    assert "_object_vector(out)" in attrs_source
    equality_owner = next(
        node
        for node in metadata_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_equal_or_nan"
    )
    equality_source = ast.get_source_segment(metadata_text, equality_owner)
    assert "_is_missing_scalar(left)" in equality_source
    assert "_is_missing_scalar(right)" in equality_source


def test_cat_doc_p11b_001_query_empty_and_large_dataset_default_semantics_documented() -> None:
    """ID: CAT_DOC_P11B_001_query_empty_and_large_dataset_default_semantics_documented."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8")
    assert "query" in text
    assert "where" in text
    assert "metadata" in text


def test_cat_doc_p11b_002_extract_metadata_promotion_and_sequence_size_coord_policy_documented() -> None:
    """ID: CAT_DOC_P11B_002_extract_metadata_promotion_and_sequence_size_coord_policy_documented."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8")
    assert "extract" in text
    assert "Metadata promotion" in text
    assert "AnalysisObject" in text
    assert "own their invariant coordinate buffers" in text
    assert "mutable objects stored in payload variables" in text
    assert "variable encodings are also detached" in text
    assert "`Catalog.data` returns an independently owned public payload" in text
    assert "failures retain the calling Catalog" in text
    assert "Pandas extension-dtype indexes" in text
    assert "Registered auxiliary" in text
    assert "Schema-owned parameter and sequence-size coordinates" in text
    assert "both metadata targets are" in text
    assert "Empty DataTree selections apply the same" in text
    assert "one metadata cell per group" in text


def test_cat_doc_p11b_003_query_extract_grouping_is_constructor_scoped_and_immutable() -> None:
    """ID: CAT_DOC_P11B_003_query_extract_grouping_is_constructor_scoped_and_immutable."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8")
    assert "Grouping is resolved at construction" in text
    assert "extract" in text
