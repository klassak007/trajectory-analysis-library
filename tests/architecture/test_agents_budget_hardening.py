from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.architecture._budget import function_loc


def _function_node(path: str, name: str) -> ast.FunctionDef:
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in {path}")


def _max_nesting(node: ast.AST, depth: int = 0) -> int:
    nested = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.Match, ast.AsyncFor)):
            nested.append(_max_nesting(child, depth + 1))
            continue
        nested.append(_max_nesting(child, depth))
    return max([depth, *nested]) if nested else depth


def test_arch_agents_006_hotspot_function_length_budget_guard() -> None:
    """ID: ARCH_AGENTS_006_hotspot_function_length_budget_guard."""
    checks = [
        ("tal/core/combine_ops/concat_topology.py", "aligned_sequence_inputs"),
        ("tal/core/combine_ops/concat_core.py", "concat_core"),
        ("tal/core/combine_ops/concat_core.py", "concat_core_contexts"),
        ("tal/core/combine_ops/merge.py", "merge_contexts"),
        ("tal/linalg/ops/solve.py", "solve"),
        ("tal/linalg/ops/solve.py", "_prepare_solve_runtime"),
        ("tal/core/ufunc_ops/orchestrate.py", "prepare_binary_ao_context"),
    ]
    for path, name in checks:
        node = _function_node(path, name)
        length = function_loc(node, path=path)
        assert length <= 50, f"{path}:{name} exceeds function length budget ({length} > 50)"


def test_arch_agents_007_hotspot_param_count_budget_guard() -> None:
    """ID: ARCH_AGENTS_007_hotspot_param_count_budget_guard."""
    node = _function_node("tal/core/param_engine/map_build.py", "_apply_linear_interior")
    params = len(node.args.args) + len(node.args.kwonlyargs)
    assert params <= 10, f"map_build._apply_linear_interior exceeds parameter budget ({params} > 10)"


def test_arch_agents_008_non_deferred_nesting_hotspot_guard() -> None:
    """ID: ARCH_AGENTS_008_non_deferred_nesting_hotspot_guard."""
    checks = [
        ("tal/core/param_engine/query_grid.py", "normalize_query_grid"),
        ("tal/core/param_ops/sync_autogrid_backend.py", "_rows_join"),
        ("tal/core/combine_ops/merge.py", "_merge_override_left_biased_fill_holes"),
    ]
    for path, name in checks:
        depth = _max_nesting(_function_node(path, name))
        assert depth <= 2, f"{path}:{name} exceeds nesting guard ({depth} > 2)"


def test_arch_agents_009_non_deferred_nesting_reopened_hotspots_guard() -> None:
    """ID: ARCH_AGENTS_009_non_deferred_nesting_reopened_hotspots_guard."""
    checks = [
        ("tal/core/orchestration/topology_batch.py", "restore_combine_batch_axis"),
        ("tal/core/combine_ops/merge.py", "_assert_compat_no_chunked_overlap"),
        ("tal/core/combine_ops/metadata.py", "resolve_core_dims"),
    ]
    for path, name in checks:
        depth = _max_nesting(_function_node(path, name))
        assert depth <= 2, f"{path}:{name} exceeds nesting guard ({depth} > 2)"


def test_arch_agents_010_unique_numeric_regression_id_prefix_guard() -> None:
    """ID: ARCH_AGENTS_010_unique_numeric_regression_id_prefix_guard."""
    id_pat = re.compile(r"ID:\s*([A-Za-z0-9_]+)")
    base_pat = re.compile(r"^([A-Z]+(?:_[A-Z]+)*_\d{3})")
    seen: dict[str, tuple[str, int]] = {}
    dupes: list[str] = []
    for path in sorted(Path("tests").rglob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        for match in id_pat.finditer(text):
            full = match.group(1)
            base = base_pat.match(full)
            if base is None:
                continue
            key = base.group(1)
            line = text.count("\n", 0, match.start()) + 1
            prev = seen.get(key)
            if prev is None:
                seen[key] = (str(path), line)
                continue
            dupes.append(f"{key}: {prev[0]}:{prev[1]} and {path}:{line}")
    assert not dupes, "duplicate numeric regression ID prefixes found:\n" + "\n".join(dupes)
