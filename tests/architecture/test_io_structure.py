from __future__ import annotations

import ast
from pathlib import Path

import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.io import finalize as io_finalize
from tal.io import zarr_io as io_zarr
from tests.architecture._budget import (
    file_loc,
    function_control_depths,
    function_lengths,
    function_parameter_counts,
)
from tests.architecture._schema_write import has_tal_schema_write


def _called_leaf_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        function = item.func
        if isinstance(function, ast.Name):
            names.add(function.id)
        elif isinstance(function, ast.Attribute):
            names.add(function.attr)
    return names


class _DirectCallVisitor(ast.NodeVisitor):
    def __init__(self, root: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.root = root
        self.leaf_names: set[str] = set()
        self.name_calls: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.root:
            for statement in node.body:
                self.visit(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is self.root:
            for statement in node.body:
                self.visit(statement)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if isinstance(function, ast.Name):
            self.leaf_names.add(function.id)
            self.name_calls.add(function.id)
        elif isinstance(function, ast.Attribute):
            self.leaf_names.add(function.attr)
        self.generic_visit(node)


class _BoundNameVisitor(ast.NodeVisitor):
    def __init__(
        self,
        root: ast.FunctionDef | ast.AsyncFunctionDef | None = None,
    ) -> None:
        self.root = root
        self.names = set() if root is None else _argument_names(root.args)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node is self.root:
            for statement in node.body:
                self.visit(statement)
            return
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(alias.asname or alias.name.split(".", maxsplit=1)[0] for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self.names.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest is not None:
            self.names.add(node.rest)
        self.generic_visit(node)


def _argument_names(args: ast.arguments) -> set[str]:
    positional = (*args.posonlyargs, *args.args, *args.kwonlyargs)
    names = {arg.arg for arg in positional}
    names.update(arg.arg for arg in (args.vararg, args.kwarg) if arg is not None)
    return names


def _bound_names(
    node: ast.AST,
    *,
    root: ast.FunctionDef | ast.AsyncFunctionDef | None = None,
) -> set[str]:
    visitor = _BoundNameVisitor(root)
    visitor.visit(node)
    return visitor.names


def _direct_called_leaf_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    visitor = _DirectCallVisitor(node)
    visitor.visit(node)
    return visitor.leaf_names


def _direct_called_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    visitor = _DirectCallVisitor(node)
    visitor.visit(node)
    return visitor.name_calls


def _call_owners(text: str, *, called_name: str) -> set[str]:
    tree = ast.parse(text)
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and called_name in _direct_called_leaf_names(node)
    }


def _attribute_owners(text: str, *, attribute: str) -> set[str]:
    tree = ast.parse(text)
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(item, ast.Attribute) and item.attr == attribute
            for item in ast.walk(node)
        )
    }


def _imports_symbol(text: str, *, module: str, name: str) -> bool:
    tree = ast.parse(text)
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == module
        and any(alias.name == name for alias in node.names)
        for node in ast.walk(tree)
    )


def _defined_function_names(text: str) -> set[str]:
    return {
        node.name
        for node in ast.parse(text).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _defined_class_names(text: str) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.ClassDef)
    }


def _imported_from_names(text: str, *, module: str) -> dict[str, str]:
    imports: dict[str, str] = {}
    for node in ast.parse(text).body:
        if not isinstance(node, ast.ImportFrom) or node.module != module:
            continue
        imports.update({alias.asname or alias.name: alias.name for alias in node.names})
    return imports


def _active_imported_from_names(text: str, *, module: str) -> dict[str, str]:
    imports: dict[str, str] = {}
    for statement in ast.parse(text).body:
        if isinstance(statement, ast.ImportFrom) and statement.module == module:
            imports.update(
                (alias.asname or alias.name, alias.name)
                for alias in statement.names
                if alias.name != "*"
            )
            continue
        for name in _bound_names(statement):
            imports.pop(name, None)
    return imports


def _imports_module(text: str, *, module: str) -> bool:
    tree = ast.parse(text)
    return any(
        (
            isinstance(node, ast.Import)
            and any(alias.name == module for alias in node.names)
        )
        or (isinstance(node, ast.ImportFrom) and node.module == module)
        for node in ast.walk(tree)
    )


def _compares_to_literal(text: str, *, literal: str) -> bool:
    return any(
        isinstance(node, ast.Compare)
        and any(
            isinstance(item, ast.Constant) and item.value == literal
            for item in (node.left, *node.comparators)
        )
        for node in ast.walk(ast.parse(text))
    )


def _function_call_graph(text: str) -> dict[str, set[str]]:
    functions = {
        node.name: node
        for node in ast.parse(text).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    names = set(functions)
    return {
        name: (
            _direct_called_names(node) - _bound_names(node, root=node)
        ).intersection(names)
        for name, node in functions.items()
    }


def _reachable_function_names(text: str, *, root: str) -> set[str]:
    graph = _function_call_graph(text)
    pending = [root]
    reached: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        pending.extend(graph.get(name, set()) - reached)
    return reached


def _reachable_imported_calls(text: str, *, module: str, root: str) -> set[str]:
    """Return imported calls in the root's static module-level call graph.

    Branch execution is deliberately owned by behavioral sentinel tests rather
    than a partial reimplementation of Python control flow.
    """
    reachable = _reachable_function_names(text, root=root)
    call_owners = {
        node.name: _direct_called_names(node) - _bound_names(node, root=node)
        for node in ast.parse(text).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return {
        original
        for local, original in _active_imported_from_names(text, module=module).items()
        if any(local in call_owners.get(owner, set()) for owner in reachable)
    }


_FINALIZE_VALID_ALIAS_PROBE = """
from finalize import finalize_loaded_dataset as finalize
def helper():
    return finalize()
def read_analysis_object_zarr():
    return helper()
"""

_FINALIZE_UNLINKED_PROBES = (
    """from finalize import finalize_loaded_dataset as finalize
def read_analysis_object_zarr():
    return backend.finalize()
def dead_helper():
    return finalize()
""",
    """def read_analysis_object_zarr():
    from finalize import finalize_loaded_dataset
    return finalize_loaded_dataset()
""",
    """from finalize import finalize_loaded_dataset
def read_analysis_object_zarr(finalize_loaded_dataset):
    return finalize_loaded_dataset()
""",
    """from finalize import finalize_loaded_dataset
def finalize_loaded_dataset():
    return None
def read_analysis_object_zarr():
    return finalize_loaded_dataset()
""",
    """from finalize import finalize_loaded_dataset
def read_analysis_object_zarr():
    finalize_loaded_dataset = lambda: None
    return finalize_loaded_dataset()
""",
    """from finalize import finalize_loaded_dataset
def helper():
    return finalize_loaded_dataset()
def read_analysis_object_zarr():
    def helper():
        return None
    return helper()
""",
    """from finalize import finalize_loaded_dataset
match object():
    case finalize_loaded_dataset:
        pass
def read_analysis_object_zarr():
    return finalize_loaded_dataset()
""",
    """from finalize import finalize_loaded_dataset
def read_analysis_object_zarr(value):
    match value:
        case finalize_loaded_dataset:
            pass
    return finalize_loaded_dataset()
""",
)


def test_arch_io_p10a_001_io_parsing_logic_is_outside_tal_core() -> None:
    """ID: ARCH_IO_P10A_001_io_parsing_logic_is_outside_tal_core."""
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "from tal.io" not in text
        assert "import tal.io" not in text
        assert "from_csv(" not in text
        assert "from_zarr(" not in text


def test_arch_io_p10a_002_ao_direct_io_finalization_reuses_core_schema_owners() -> None:
    """ID: ARCH_IO_P10A_002_ao_direct_io_finalization_reuses_core_schema_owners."""
    finalize_text = Path("tal/io/finalize.py").read_text(encoding="utf-8")
    zarr_text = Path("tal/io/zarr_io.py").read_text(encoding="utf-8")
    for name in ("CoreSchemaFinalizeSpec", "finalize_with_schema"):
        assert _imports_symbol(
            finalize_text,
            module="tal.core.orchestration.schema_finalize",
            name=name,
        )
    assert _imports_symbol(
        finalize_text,
        module="tal.core.orchestration.context",
        name="resolve_dataset_context",
    )
    assert _imports_symbol(zarr_text, module="finalize", name="finalize_loaded_dataset")
    assert _call_owners(zarr_text, called_name="finalize_loaded_dataset")


def test_arch_io_p10a_004_analysisobject_io_surface_installation_is_wired_from_top_level_package_init() -> None:
    """ID: ARCH_IO_P10A_004_analysisobject_io_surface_installation_is_wired_from_top_level_package_init."""
    top_text = Path("tal/__init__.py").read_text(encoding="utf-8")
    assert _imports_symbol(
        top_text,
        module="io",
        name="install_analysis_object_io_surface",
    )
    assert "install_analysis_object_io_surface" in _called_leaf_names(ast.parse(top_text))


def test_arch_io_p10a_005_io_readers_construct_requested_cls_after_shared_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ARCH_IO_P10A_005_io_readers_construct_requested_cls_after_shared_finalization."""
    source = AnalysisObject.from_data(
        xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    events: list[str] = []

    def finalize_sentinel(
        source_ao: AnalysisObject,
        result: xr.Dataset,
        **_kwargs: object,
    ) -> AnalysisObject:
        events.append("finalize")
        marked = result.assign_attrs({**result.attrs, "io_finalized": True})
        return source_ao.__class__(marked)

    class RequestedAO(AnalysisObject):
        def __init__(self, data: xr.Dataset) -> None:
            events.append("construct")
            assert data.attrs.get("io_finalized") is True
            super().__init__(data)

    monkeypatch.setattr(io_finalize, "finalize_with_schema", finalize_sentinel)
    out = io_finalize.finalize_loaded_dataset(
        RequestedAO,
        source.unsafe_data,
        validate=True,
        owner="architecture.finalize_order",
    )
    assert isinstance(out, RequestedAO)
    assert events == ["finalize", "construct"]


def test_arch_io_p10a_006_finalize_source_subclass_constructor_path_avoids_broad_exception_catch() -> None:
    """ID: ARCH_IO_P10A_006_finalize_source_subclass_constructor_path_avoids_broad_exception_catch."""
    source = AnalysisObject.from_data(
        xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )

    class CrashingAO(AnalysisObject):
        def __init__(self, data: xr.Dataset) -> None:
            super().__init__(data)
            raise RuntimeError("constructor failure")

    with pytest.raises(RuntimeError, match="constructor failure"):
        io_finalize.finalize_loaded_dataset(
            CrashingAO,
            source.unsafe_data,
            validate=True,
            owner="architecture.constructor_failure",
        )


def test_arch_io_p10a_007_zarr_validity_materialization_has_single_ingress_owner() -> None:
    """ID: ARCH_IO_P10A_007_zarr_validity_materialization_has_single_ingress_owner."""
    zarr_text = Path("tal/io/zarr_io.py").read_text(encoding="utf-8")
    eager_methods = {"asarray", "compute", "item", "load", "persist", "to_numpy"}
    eager_owners = set().union(
        *(_call_owners(zarr_text, called_name=name) for name in eager_methods),
        _attribute_owners(zarr_text, attribute="values"),
    )
    assert len(eager_owners) == 1
    eager_owner = next(iter(eager_owners))
    for root in ("read_analysis_object_zarr", "write_analysis_object_zarr"):
        assert eager_owner in _reachable_function_names(zarr_text, root=root)
    for path in (Path("tal/io/finalize.py"), Path("tal/io/surface.py")):
        text = path.read_text(encoding="utf-8")
        assert not set().union(
            *(_call_owners(text, called_name=name) for name in eager_methods),
            _attribute_owners(text, attribute="values"),
        )
    assert not _imports_module(
        zarr_text,
        module="tal.core.schema_validate.phase_validity",
    )


def test_arch_io_p10a_008_ao_direct_csv_persistence_is_absent() -> None:
    """ID: ARCH_IO_P10A_008_ao_direct_csv_persistence_is_absent."""
    for name in ("csv_io.py", "_csv_codec.py", "_csv_staging.py", "metadata.py"):
        assert not (Path("tal/io") / name).exists()
    surface_text = Path("tal/io/surface.py").read_text(encoding="utf-8")
    init_text = Path("tal/io/__init__.py").read_text(encoding="utf-8")
    assert "to_csv" not in surface_text
    assert "from_csv" not in surface_text
    assert "AOCsv" not in init_text
    assert "read_csv_logs" in init_text
    assert "write_csv_logs" in init_text
    generated = Path("docs/api/_generated/io")
    assert not (generated / "tal.AnalysisObject.from_csv.rst").exists()
    assert not (generated / "tal.io.AnalysisObjectIOAccessor.to_csv.rst").exists()


def _assert_public_zarr_executes_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = AnalysisObject.from_data(
        xr.Dataset({"value": ("sample", [1.0])}),
        sequence_dim="sample",
        core_dims=(),
    ).unsafe_data
    calls: list[tuple[type[AnalysisObject], xr.Dataset, bool, str]] = []

    def finalize_sentinel(
        cls: type[AnalysisObject],
        ds: xr.Dataset,
        *,
        validate: bool,
        owner: str,
    ) -> AnalysisObject:
        calls.append((cls, ds, validate, owner))
        return cls(ds)

    monkeypatch.setattr(io_zarr.xr, "open_zarr", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(io_zarr, "finalize_loaded_dataset", finalize_sentinel)
    out = AnalysisObject.from_zarr("sentinel.zarr")
    assert calls == [(AnalysisObject, source, True, "AnalysisObject.from_zarr")]
    assert isinstance(out, AnalysisObject)


def test_arch_io_p10a_009_finalization_owners_are_linked_and_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ARCH_IO_P10A_009_finalization_owners_are_linked_and_executed."""
    finalize_text = Path("tal/io/finalize.py").read_text(encoding="utf-8")
    zarr_text = Path("tal/io/zarr_io.py").read_text(encoding="utf-8")

    assert _reachable_imported_calls(
        _FINALIZE_VALID_ALIAS_PROBE,
        module="finalize",
        root="read_analysis_object_zarr",
    ) == {"finalize_loaded_dataset"}
    for probe in _FINALIZE_UNLINKED_PROBES:
        assert not _reachable_imported_calls(
            probe,
            module="finalize",
            root="read_analysis_object_zarr",
        )

    assert {"CoreSchemaFinalizeSpec", "finalize_with_schema"}.issubset(
        _reachable_imported_calls(
            finalize_text,
            module="tal.core.orchestration.schema_finalize",
            root="finalize_loaded_dataset",
        )
    )
    assert "resolve_dataset_context" in _reachable_imported_calls(
        finalize_text,
        module="tal.core.orchestration.context",
        root="finalize_loaded_dataset",
    )
    assert "finalize_loaded_dataset" in _reachable_imported_calls(
        zarr_text,
        module="finalize",
        root="read_analysis_object_zarr",
    )
    _assert_public_zarr_executes_finalize(monkeypatch)


def test_arch_io_p10a_010_io_functions_respect_parameter_budget() -> None:
    """ID: ARCH_IO_P10A_010_io_functions_respect_parameter_budget."""
    helper_probe = """
class First:
    def repeated(self, /, positional, *items, keyword, **extras):
        return None
class Second:
    def repeated(self):
        return None
"""
    assert function_parameter_counts(source=helper_probe) == {
        "First.repeated": 5,
        "Second.repeated": 1,
    }
    for path in sorted(Path("tal/io").glob("*.py")):
        counts = function_parameter_counts(path)
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        node_count = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(parsed)
        )
        assert len(counts) == node_count, f"{path} budget keys are not unique."
        for name, count in counts.items():
            assert count <= 10, f"{path}:{name} exceeds parameter budget ({count} > 10)."


def test_io_hard_p10a_003_no_schema_bypass_writes_on_ao_direct_io_paths() -> None:
    """ID: IO_HARD_P10A_003_no_schema_bypass_writes_on_ao_direct_io_paths."""
    for path in sorted(Path("tal/io").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not has_tal_schema_write(text)


def test_arch_io_p10a_011_schema_write_guard_covers_attrs_replacement() -> None:
    """ID: ARCH_IO_P10A_011_schema_write_guard_covers_attrs_replacement."""
    assert has_tal_schema_write('ds.attrs["tal"] = schema')
    assert has_tal_schema_write('ds.attrs = {**ds.attrs, "tal": schema}')
    assert has_tal_schema_write('ds.attrs.update({"tal": schema})')
    assert has_tal_schema_write("ds.assign_attrs(tal=schema)")
    assert not has_tal_schema_write("out = merge_schema(ds, schema)")
    export_text = Path("tal/io/csv_export.py").read_text(encoding="utf-8")
    assert _imports_symbol(
        export_text,
        module="tal.core.schema",
        name="merge_schema",
    )
    assert "merge_schema" in _reachable_imported_calls(
        export_text,
        module="tal.core.schema",
        root="coerce_csv_export_source",
    )


def test_arch_io_p10a_012_schema_write_guard_covers_local_mapping_mutation() -> None:
    """ID: ARCH_IO_P10A_012_schema_write_guard_covers_local_mapping_mutation."""
    assert has_tal_schema_write(
        'attrs = dict(ds.attrs)\nattrs.update({"tal": schema})\nds.attrs = attrs'
    )
    assert has_tal_schema_write(
        'attrs = dict(ds.attrs)\nattrs.setdefault("tal", schema)\nds.attrs = attrs'
    )
    assert has_tal_schema_write(
        'attrs = dict(ds.attrs)\nattrs |= {"tal": schema}\nds.attrs = attrs'
    )


def test_io_doc_p10a_001_ao_direct_io_is_canonical_documented() -> None:
    """ID: IO_DOC_P10A_001_ao_direct_io_is_canonical_documented."""
    text = " ".join(Path("docs/api/io.md").read_text(encoding="utf-8").lower().split())
    assert "ao-direct" in text
    assert "ao.io.to_zarr" in text
    assert "intentionally has no `ao.io.to_csv" in text
    assert "structurally validates persisted" in text
    assert "non-resident sequence-size coordinate" in text
    assert "other chunked" in text
    assert "payload variables remain lazy" in text


def test_io_doc_p10a_002_analysisobject_io_top_level_installation_guarantee_documented() -> None:
    """ID: IO_DOC_P10A_002_analysisobject_io_top_level_installation_guarantee_documented."""
    text = Path("docs/api/io.md").read_text(encoding="utf-8").lower()
    assert "analysisobjectioaccessor" in text
    assert "analysisobject.from_zarr" in text


def test_io_doc_p10a_003_csv_log_boundary_is_lossy_documented() -> None:
    """ID: IO_DOC_P10A_003_csv_log_boundary_is_lossy_documented."""
    text = " ".join(Path("docs/api/io.md").read_text(encoding="utf-8").lower().split())
    assert "lossy" in text
    assert "not inverse operations" in text
    assert "no sidecar" in text
    assert "may eagerly materialize" in text
    assert "payload-nul behavior" in text
    assert "python/pandas own csv grammar" in text


def test_io_doc_p10a_004_zarr_persistence_boundary_is_semantic_not_identity() -> None:
    """ID: IO_DOC_P10A_004_zarr_persistence_boundary_is_semantic_not_identity."""
    text = " ".join(Path("docs/api/io.md").read_text(encoding="utf-8").lower().split())
    assert "zarr is tal's canonical analysisobject persistence boundary" in text
    assert "validated tal schema and decoded array values" in text
    assert "does not promise representation identity" in text
    for limitation in ("attribute container", "index class", "encoding", "chunk topology"):
        assert limitation in text


def test_io_doc_p10b_001_csv_ros_adapter_policies_documented() -> None:
    """ID: IO_DOC_P10B_001_csv_ros_adapter_policies_documented."""
    text = Path("docs/api/io.md").read_text(encoding="utf-8").lower()
    assert "read_csv_logs" in text
    assert "read_ros_logs" in text
    assert "timestamp" in text


def test_io_doc_p10b_002_ingest_and_export_label_ordering_policies_documented() -> None:
    """ID: IO_DOC_P10B_002_ingest_and_export_label_ordering_policies_documented."""
    text = Path("docs/api/io.md").read_text(encoding="utf-8").lower()
    assert "caller order" in text
    assert "glob" in text


def test_io_doc_p10b_003_csv_export_validity_defaults_and_strict_explicit_value_column_parsing_documented() -> None:
    """ID: IO_DOC_P10B_003_csv_export_validity_defaults_and_strict_explicit_value_column_parsing_documented."""
    text = " ".join(Path("docs/api/io.md").read_text(encoding="utf-8").lower().split())
    assert "sequence-size metadata" in text
    assert "value-column" in text
    assert "configured dask scheduler" in text
    assert "trusted, stable-filesystem" in text
    assert "python/pandas own csv grammar" in text
    assert "batch is not transactional" in text
    assert "overridden declared validity coordinate is omitted" in text


def test_io_doc_p10b_004_reserved_adapter_metadata_key_policy_documented() -> None:
    """ID: IO_DOC_P10B_004_reserved_adapter_metadata_key_policy_documented."""
    text = " ".join(Path("docs/api/io.md").read_text(encoding="utf-8").lower().split())
    assert "`tal` attr key" in text
    assert "schema" in text
    assert "separate xarray namespace" in text
    assert "batch coordinate named `tal`" in text
    assert "adapter-generated identities" in text
    assert "preflighted" in text


def test_arch_io_p10b_001_adapter_parsing_logic_does_not_enter_tal_core() -> None:
    """ID: ARCH_IO_P10B_001_adapter_parsing_logic_does_not_enter_tal_core."""
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "read_csv_logs(" not in text
        assert "read_ros_logs(" not in text
        assert "write_csv_logs(" not in text


def test_arch_io_p10b_002_adapters_reuse_core_schema_finalization_boundaries() -> None:
    """ID: ARCH_IO_P10B_002_adapters_reuse_core_schema_finalization_boundaries."""
    adapter_finalize_text = Path("tal/io/adapter_finalize.py").read_text(encoding="utf-8")
    csv_logs_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    ros_logs_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    for name in ("CoreSchemaFinalizeSpec", "finalize_with_schema"):
        assert _imports_symbol(
            adapter_finalize_text,
            module="tal.core.orchestration.schema_finalize",
            name=name,
        )
    assert not {
        "set_roles",
        "set_param_coord",
        "set_validity",
    }.intersection(_called_leaf_names(ast.parse(adapter_finalize_text)))
    for consumer, root in (
        (csv_logs_text, "read_csv_logs"),
        (ros_logs_text, "read_ros_logs"),
    ):
        assert _imports_module(consumer, module="adapter_finalize")
        assert _reachable_imported_calls(
            consumer,
            module="adapter_finalize",
            root=root,
        )
        assert not _imports_module(
            consumer,
            module="tal.core.orchestration.schema_finalize",
        )


def test_arch_io_p10b_016_owned_adapter_finalize_avoids_generic_external_ingress() -> None:
    """ID: ARCH_IO_P10B_016_owned_adapter_finalize_avoids_generic_external_ingress."""
    adapter_text = Path("tal/io/adapter_finalize.py").read_text(encoding="utf-8")
    assert not {
        "AnalysisObject",
        "coerce_analysis_object_input",
        "copy",
        "finalize_loaded_dataset",
    }.intersection(_called_leaf_names(ast.parse(adapter_text)))
    consumers = {
        path.name
        for path in Path("tal/io").glob("*.py")
        if _imports_module(path.read_text(encoding="utf-8"), module="adapter_finalize")
    }
    assert consumers == {"csv_logs.py", "ros_logs.py"}


def test_arch_io_p10b_shared_path_and_label_policy_owners_are_centralized() -> None:
    """ID: ARCH_IO_P10B_004_adapter_path_and_label_policy_is_centralized."""
    consumers = (
        (Path("tal/io/csv_logs.py"), "read_csv_logs"),
        (Path("tal/io/csv_export.py"), "execute_csv_export"),
        (Path("tal/io/ros_logs.py"), "read_ros_logs"),
    )
    for path, root in consumers:
        text = path.read_text(encoding="utf-8")
        assert _imports_module(text, module="adapter_paths")
        assert _reachable_imported_calls(text, module="adapter_paths", root=root)
        assert not _imports_module(text, module="glob")
        assert not _imports_module(text, module="stat")
        assert not _imports_module(text, module="unicodedata")

    export_text = Path("tal/io/csv_export.py").read_text(encoding="utf-8")
    export_boundaries = _reachable_imported_calls(
        export_text,
        module="adapter_paths",
        root="execute_csv_export",
    )
    assert {
        "existing_filesystem_identity",
        "filesystem_collision_key",
        "normalize_export_label_path",
        "resolve_export_root",
        "snapshot_export_destinations",
        "stringify_export_identity",
    }.issubset(export_boundaries)
    direct_path_inspection = set().union(
        *(
            _call_owners(export_text, called_name=name)
            for name in ("exists", "is_dir", "is_file", "lstat", "resolve", "stat")
        )
    )
    assert not direct_path_inspection


def test_arch_io_p10b_021_csv_commit_parent_plan_has_path_owner() -> None:
    """ID: ARCH_IO_P10B_021_csv_commit_parent_plan_has_path_owner."""
    commit_text = Path("tal/io/csv_commit.py").read_text(encoding="utf-8")
    assert _imports_symbol(
        commit_text,
        module="adapter_paths",
        name="plan_export_parent_directories",
    )
    assert "plan_export_parent_directories" in _reachable_imported_calls(
        commit_text,
        module="adapter_paths",
        root="prepare_csv_commit",
    )
    assert "exists" not in _called_leaf_names(ast.parse(commit_text))
    paths_text = Path("tal/io/adapter_paths.py").read_text(encoding="utf-8")
    assert "filesystem_collision_key" in _reachable_function_names(
        paths_text,
        root="plan_export_parent_directories",
    )


def test_arch_io_p10b_022_public_csv_entrypoints_delegate_to_boundary_owners() -> None:
    """ID: ARCH_IO_P10B_022_public_csv_entrypoints_delegate_to_boundary_owners."""
    ingest_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    export_text = Path("tal/io/csv_export.py").read_text(encoding="utf-8")
    for module in ("adapter_paths", "csv_commit"):
        assert _imports_module(export_text, module=module)
        assert _reachable_imported_calls(
            export_text,
            module=module,
            root="execute_csv_export",
        )
    assert _imports_module(ingest_text, module="csv_validation")
    assert "require_valid_csv_header" in _reachable_imported_calls(
        ingest_text,
        module="csv_validation",
        root="read_csv_logs",
    )
    assert _imports_module(export_text, module="csv_export_context")
    assert "resolve_csv_export_context" in _reachable_imported_calls(
        export_text,
        module="csv_export_context",
        root="execute_csv_export",
    )
    context_text = Path("tal/io/csv_export_context.py").read_text(encoding="utf-8")
    assert "set_validity" in _reachable_imported_calls(
        context_text,
        module="tal.core.schema",
        root="resolve_csv_export_context",
    )
    assert "read_sequence_size_coord_name" in _reachable_imported_calls(
        context_text,
        module="tal.core.schema_read",
        root="resolve_csv_export_context",
    )


def test_arch_io_p10b_005_csv_export_size_and_value_parse_policies_are_centralized() -> None:
    """ID: ARCH_IO_P10B_005_csv_export_size_coord_and_value_parse_policies_are_centralized."""
    csv_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    export_text = Path("tal/io/csv_export.py").read_text(encoding="utf-8")
    sizes_text = Path("tal/io/csv_sizes.py").read_text(encoding="utf-8")
    assert _imports_module(csv_text, module="csv_time")
    assert _imports_module(export_text, module="csv_sizes")
    assert _imports_module(sizes_text, module="tal.core.validity_values")
    time_boundaries = _reachable_imported_calls(
        csv_text,
        module="csv_time",
        root="read_csv_logs",
    )
    size_boundaries = _reachable_imported_calls(
        export_text,
        module="csv_sizes",
        root="execute_csv_export",
    )
    assert time_boundaries
    assert size_boundaries
    owned_size_boundaries = size_boundaries.intersection(
        _defined_function_names(sizes_text)
    )
    assert owned_size_boundaries
    for root in owned_size_boundaries:
        assert _reachable_imported_calls(
            sizes_text,
            module="tal.core.validity_values",
            root=root,
        )
    assert not _imports_module(export_text, module="tal.core.validity_values")
    assert not _imports_module(
        export_text,
        module="tal.core.orchestration.context",
    )
    assert "resolve_dataset_context" not in _called_leaf_names(ast.parse(export_text))
    context_text = Path("tal/io/csv_export_context.py").read_text(encoding="utf-8")
    assert _imports_symbol(context_text, module="tal.core.schema", name="set_validity")


def test_arch_io_p10b_006_reserved_metadata_key_validation_is_centralized_in_adapter_metadata_owner() -> None:
    """ID: ARCH_IO_P10B_006_reserved_metadata_key_validation_is_centralized_in_adapter_metadata_owner."""
    metadata_text = Path("tal/io/adapter_metadata.py").read_text(encoding="utf-8")
    csv_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    ros_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    assert not _imports_module(metadata_text, module="pandas")
    reserved_owners = _call_owners(
        metadata_text,
        called_name="_require_attr_metadata_name",
    )
    assert {
        "require_generated_metadata_preflight",
        "_require_metadata_name_available",
    }.issubset(reserved_owners)
    for consumer, root in ((csv_text, "read_csv_logs"), (ros_text, "read_ros_logs")):
        assert _imports_module(consumer, module="adapter_metadata")
        assert _reachable_imported_calls(consumer, module="adapter_metadata", root=root)
        assert not _compares_to_literal(consumer, literal="tal")


def test_arch_io_p10b_017_active_error_cleanup_has_shared_io_owner() -> None:
    """ID: ARCH_IO_P10B_017_active_error_cleanup_has_shared_io_owner."""
    cleanup_text = Path("tal/io/adapter_cleanup.py").read_text(encoding="utf-8")
    assert any(
        isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "BaseException"
        for node in ast.walk(ast.parse(cleanup_text))
    )

    expected_consumers = (
        ("adapter_temp.py", "owned_temporary_directory"),
        ("ros_reader.py", "iter_ros_messages"),
        ("ros_reader.py", "owned_ros_message_stream"),
        ("zarr_io.py", "read_analysis_object_zarr"),
    )
    for filename, root in expected_consumers:
        text = Path("tal/io", filename).read_text(encoding="utf-8")
        assert _imports_module(text, module="adapter_cleanup")
        assert _reachable_imported_calls(text, module="adapter_cleanup", root=root)


def test_arch_io_p10b_007_csv_export_column_name_normalization_collision_checks_are_centralized() -> None:
    """ID: ARCH_IO_P10B_007_csv_export_column_name_normalization_collision_checks_are_centralized."""
    export_text = Path("tal/io/csv_export.py").read_text(encoding="utf-8")
    validation_text = Path("tal/io/csv_validation.py").read_text(encoding="utf-8")
    assert _imports_module(export_text, module="csv_validation")
    assert _reachable_imported_calls(
        export_text,
        module="csv_validation",
        root="execute_csv_export",
    )
    assert not _imports_module(validation_text, module="pandas")


def test_arch_io_p10b_008_csv_path_resolution_is_preflight_only() -> None:
    """ID: ARCH_IO_P10B_008_csv_path_resolution_is_preflight_only."""
    paths_text = Path("tal/io/adapter_paths.py").read_text(encoding="utf-8")
    export_text = Path("tal/io/csv_export.py").read_text(encoding="utf-8")
    commit_text = Path("tal/io/csv_commit.py").read_text(encoding="utf-8")
    assert not any(
        _imports_module(paths_text, module=name)
        for name in ("os", "shutil", "tempfile")
    )
    assert not any(
        _imports_module(export_text, module=name)
        for name in ("os", "shutil", "tempfile")
    )
    commit_functions = _defined_function_names(commit_text)
    imported_boundary = _imported_from_names(export_text, module="csv_commit")
    imported_functions = set(imported_boundary.values()).intersection(commit_functions)
    assert imported_functions
    reachable_boundary = _reachable_imported_calls(
        export_text,
        module="csv_commit",
        root="execute_csv_export",
    )
    assert imported_functions.issubset(reachable_boundary)
    production_reachable = set().union(
        *(
            _reachable_function_names(commit_text, root=name)
            for name in imported_functions
        )
    )
    assert commit_functions.issubset(production_reachable)
    assert not {
        original
        for original in imported_boundary.values()
        if original.startswith("_")
    }


def test_arch_io_p10b_009_ros_filters_connections_before_streamed_deserialization() -> None:
    """ID: ARCH_IO_P10B_009_ros_filters_connections_before_streamed_deserialization."""
    ros_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    reader_text = Path("tal/io/ros_reader.py").read_text(encoding="utf-8")
    assert _imports_module(ros_text, module="ros_reader")
    assert _imports_module(reader_text, module="ros_payload")
    assert _reachable_imported_calls(
        ros_text,
        module="ros_reader",
        root="read_ros_logs",
    )
    reader_payload_calls = _reachable_imported_calls(
        reader_text,
        module="ros_payload",
        root="iter_ros_messages",
    )
    assert reader_payload_calls
    assert "require_message_family" in reader_payload_calls
    assert "require_message_family" in _reachable_imported_calls(
        reader_text,
        module="ros_payload",
        root="_select_ros_connections",
    )
    assert "require_message_family" not in _reachable_imported_calls(
        reader_text,
        module="ros_payload",
        root="_decode_ros_message",
    )
    assert not _imports_symbol(
        ros_text,
        module="ros_payload",
        name="require_message_family",
    )
    assert not _call_owners(ros_text, called_name="require_message_family")
    assert not _imports_module(reader_text, module="xarray")


def test_arch_io_p10b_023_ros_consumer_cleanup_reenters_reader_owner() -> None:
    """ID: ARCH_IO_P10B_023_ros_consumer_cleanup_reenters_reader_owner."""
    ros_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    reachable = _reachable_imported_calls(
        ros_text,
        module="ros_reader",
        root="read_ros_logs",
    )
    assert {"iter_ros_messages", "owned_ros_message_stream"}.issubset(reachable)


def test_arch_io_p10b_015_ros_payload_layout_collision_preflight_is_domain_local() -> None:
    """ID: ARCH_IO_P10B_015_ros_payload_layout_collision_preflight_is_domain_local."""
    ros_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    options_text = Path("tal/io/options.py").read_text(encoding="utf-8")
    assert _imports_module(ros_text, module="ros_payload")
    assert not _imports_module(options_text, module="ros_payload")


def test_arch_io_p10b_013_ros_payload_kernel_has_domain_owner() -> None:
    """ID: ARCH_IO_P10B_013_ros_payload_kernel_has_domain_owner."""
    payload_text = Path("tal/io/ros_payload.py").read_text(encoding="utf-8")
    ros_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    assert _imports_module(ros_text, module="ros_payload")
    assert _reachable_imported_calls(
        ros_text,
        module="ros_payload",
        root="read_ros_logs",
    )
    assert not _imports_module(payload_text, module="xarray")
    assert not any(
        node.module and node.module.startswith("tal.core")
        for node in ast.parse(payload_text).body
        if isinstance(node, ast.ImportFrom)
    )


def test_arch_io_p10b_010_adapter_time_policy_has_shared_owner() -> None:
    """ID: ARCH_IO_P10B_010_adapter_time_policy_has_shared_owner."""
    csv_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    ros_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    for consumer, root in ((csv_text, "read_csv_logs"), (ros_text, "read_ros_logs")):
        assert _imports_module(consumer, module="adapter_time")
        assert _reachable_imported_calls(consumer, module="adapter_time", root=root)


def test_arch_io_p10b_011_ingest_array_spooling_has_shared_owner() -> None:
    """ID: ARCH_IO_P10B_011_ingest_array_spooling_has_shared_owner."""
    spool_text = Path("tal/io/adapter_spool.py").read_text(encoding="utf-8")
    csv_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    ros_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    temp_text = Path("tal/io/adapter_temp.py").read_text(encoding="utf-8")
    for consumer, root in ((csv_text, "read_csv_logs"), (ros_text, "read_ros_logs")):
        assert _imports_module(consumer, module="adapter_spool")
        assert _imports_module(consumer, module="adapter_temp")
        assert _reachable_imported_calls(consumer, module="adapter_spool", root=root)
        assert _reachable_imported_calls(consumer, module="adapter_temp", root=root)
    assert _imports_module(temp_text, module="tempfile")
    assert not _imports_module(spool_text, module="xarray")


def test_arch_io_p10b_014_csv_raw_validation_has_shared_domain_owner() -> None:
    """ID: ARCH_IO_P10B_014_csv_raw_validation_has_shared_domain_owner."""
    validation = Path("tal/io/csv_validation.py").read_text(encoding="utf-8")
    ingest = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    export = Path("tal/io/csv_export.py").read_text(encoding="utf-8")
    assert _imports_module(validation, module="csv")
    assert not _imports_module(validation, module="pandas")
    assert _imports_module(ingest, module="csv_validation")
    assert _imports_module(export, module="csv_validation")
    assert _reachable_imported_calls(
        ingest,
        module="csv_validation",
        root="read_csv_logs",
    )
    assert _reachable_imported_calls(
        export,
        module="csv_validation",
        root="execute_csv_export",
    )


def test_arch_io_p10b_012_exact_ros_time_has_domain_owner() -> None:
    """ID: ARCH_IO_P10B_012_exact_ros_time_has_domain_owner."""
    time_text = Path("tal/io/ros_time.py").read_text(encoding="utf-8")
    ros_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    assert not _call_owners(time_text, called_name="float")
    assert _imports_module(ros_text, module="ros_time")
    assert _reachable_imported_calls(
        ros_text,
        module="ros_time",
        root="read_ros_logs",
    )
    assert not _imports_module(ros_text, module="csv_logs")


def test_arch_io_p10b_018_csv_metadata_parsing_has_csv_domain_owner() -> None:
    """ID: ARCH_IO_P10B_018_csv_metadata_parsing_has_csv_domain_owner."""
    generic_text = Path("tal/io/adapter_metadata.py").read_text(encoding="utf-8")
    csv_metadata_text = Path("tal/io/csv_metadata.py").read_text(encoding="utf-8")
    csv_logs_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    assert not _imports_module(generic_text, module="pandas")
    assert _imports_module(csv_metadata_text, module="pandas")
    assert _imports_module(csv_metadata_text, module="adapter_metadata")
    assert _imports_module(csv_logs_text, module="csv_metadata")
    boundaries = _reachable_imported_calls(
        csv_logs_text,
        module="csv_metadata",
        root="read_csv_logs",
    )
    assert boundaries
    owned_boundaries = boundaries.intersection(
        _defined_function_names(csv_metadata_text)
    )
    assert owned_boundaries
    for root in owned_boundaries:
        assert _reachable_imported_calls(
            csv_metadata_text,
            module="adapter_metadata",
            root=root,
        )
    consumers = {
        path.name
        for path in Path("tal/io").glob("*.py")
        if _imports_module(path.read_text(encoding="utf-8"), module="csv_metadata")
    }
    assert consumers == {"csv_logs.py"}


def test_arch_io_p10b_019_cleanup_failure_test_double_has_shared_owner() -> None:
    """ID: ARCH_IO_P10B_019_cleanup_failure_test_double_has_shared_owner."""
    helper_text = Path("tests/_io_helpers.py").read_text(encoding="utf-8")
    public_factories = {
        name for name in _defined_function_names(helper_text) if not name.startswith("_")
    }
    assert public_factories

    consumers = (
        Path("tests/core/test_io_csv_ingest.py"),
        Path("tests/core/test_io_ros_ingest.py"),
    )
    for path in consumers:
        text = path.read_text(encoding="utf-8")
        imported = _imported_from_names(text, module="tests._io_helpers")
        called = {
            original
            for local, original in imported.items()
            if _call_owners(text, called_name=local)
        }
        assert called.intersection(public_factories)
        assert not any(
            "cleanup" in name.casefold()
            for name in _defined_class_names(text)
        )


def test_arch_io_p10b_020_csv_control_flow_stays_within_nesting_budget() -> None:
    """ID: ARCH_IO_P10B_020_csv_control_flow_stays_within_nesting_budget."""
    for path in sorted(Path("tal/io").glob("*.py")):
        for name, depth in function_control_depths(path).items():
            assert depth <= 2, f"{path}:{name} exceeds nesting budget ({depth} > 2)."


def test_io_owner_budget_and_schema_write_boundary() -> None:
    """IO modules stay within AGENTS budgets and avoid direct tal-attrs writes."""
    for path in sorted(Path("tal/io").glob("*.py")):
        assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
        for name, length in function_lengths(path).items():
            assert length <= 50, f"{path}:{name} exceeds function budget ({length} > 50)."
        text = path.read_text(encoding="utf-8")
        assert not has_tal_schema_write(text)
