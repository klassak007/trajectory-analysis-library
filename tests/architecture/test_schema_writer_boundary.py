import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = REPO_ROOT / "tal"
APPROVED_WRITERS = {
    (SCAN_ROOT / "core" / "schema.py").resolve(),
    (SCAN_ROOT / "core" / "schema_validate" / "finalize.py").resolve(),
}
IGNORED_PATH_PARTS = {"tests", "tal_v2"}


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for path in SCAN_ROOT.rglob("*.py"):
        if any(part in IGNORED_PATH_PARTS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _is_tal_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "tal"


class _TalWriteDetector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.lines: set[int] = set()
        self.tal_aliases_stack: list[set[str]] = [set()]
        self.attrs_aliases_stack: list[set[str]] = [{"attrs"}]

    @property
    def tal_aliases(self) -> set[str]:
        return self.tal_aliases_stack[-1]

    @property
    def attrs_aliases(self) -> set[str]:
        return self.attrs_aliases_stack[-1]

    def _push_scope(self) -> None:
        self.tal_aliases_stack.append(set(self.tal_aliases))
        self.attrs_aliases_stack.append(set(self.attrs_aliases))

    def _pop_scope(self) -> None:
        self.tal_aliases_stack.pop()
        self.attrs_aliases_stack.pop()

    def _expr_is_attrs_root(self, expr: ast.AST) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in self.attrs_aliases
        if isinstance(expr, ast.Attribute):
            return expr.attr == "attrs"
        return False

    def _is_tal_subscript(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Subscript):
            return False
        return self._expr_is_attrs_root(node.value) and _is_tal_literal(node.slice)

    def _expr_is_tal_root(self, expr: ast.AST) -> bool:
        if self._is_tal_subscript(expr):
            return True
        if isinstance(expr, ast.Name):
            return expr.id in self.tal_aliases
        if isinstance(expr, ast.Subscript):
            return self._expr_is_tal_root(expr.value)
        return False

    def _target_writes_tal(self, target: ast.AST) -> bool:
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(self._target_writes_tal(item) for item in target.elts)
        if isinstance(target, ast.Subscript):
            return self._expr_is_tal_root(target)
        return False

    def _mark_aliases(self, targets: list[ast.AST], value: ast.AST) -> None:
        for target in targets:
            if isinstance(target, ast.Name):
                if self._expr_is_tal_root(value):
                    self.tal_aliases.add(target.id)
                if self._expr_is_attrs_root(value):
                    self.attrs_aliases.add(target.id)

    def _call_arg_has_tal_key(self, arg: ast.AST) -> bool:
        if isinstance(arg, ast.Dict):
            return any(_is_tal_literal(key) for key in arg.keys if key is not None)
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
            return arg.func.id == "dict" and any(k.arg == "tal" for k in arg.keywords if k.arg)
        if isinstance(arg, (ast.List, ast.Tuple, ast.Set)):
            for elt in arg.elts:
                if not isinstance(elt, ast.Tuple) or len(elt.elts) != 2:
                    continue
                if _is_tal_literal(elt.elts[0]):
                    return True
        return False

    def _attrs_call_writes_tal(self, call: ast.Call) -> bool:
        if not isinstance(call.func, ast.Attribute) or not self._expr_is_attrs_root(call.func.value):
            return False
        if call.func.attr == "setdefault":
            if call.args and _is_tal_literal(call.args[0]):
                return True
            for kw in call.keywords:
                if kw.arg == "key" and _is_tal_literal(kw.value):
                    return True
            return False
        if call.func.attr != "update":
            return False
        if any(kw.arg == "tal" for kw in call.keywords if kw.arg):
            return True
        return any(self._call_arg_has_tal_key(arg) for arg in call.args)

    def _tal_root_mutating_call(self, call: ast.Call) -> bool:
        if not isinstance(call.func, ast.Attribute):
            return False
        if call.func.attr not in {"clear", "pop", "popitem", "setdefault", "update"}:
            return False
        return self._expr_is_tal_root(call.func.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._mark_aliases(list(node.targets), node.value)
        if any(self._target_writes_tal(target) for target in node.targets):
            self.lines.add(node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value = node.value if node.value is not None else ast.Constant(None)
        self._mark_aliases([node.target], value)
        if self._target_writes_tal(node.target):
            self.lines.add(node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._target_writes_tal(node.target):
            self.lines.add(node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._attrs_call_writes_tal(node) or self._tal_root_mutating_call(node):
            self.lines.add(node.lineno)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()


def _tal_write_lines_from_source(source: str) -> list[int]:
    tree = ast.parse(source)
    detector = _TalWriteDetector()
    detector.visit(tree)
    return sorted(detector.lines)


def _tal_attr_write_lines(path: Path) -> list[int]:
    source = path.read_text(encoding="utf-8")
    return _tal_write_lines_from_source(source)


def test_arch_schemawrite_001_only_approved_modules_write_tal_attrs() -> None:
    """ID: ARCH_SCHEMAWRITE_001_only_approved_modules_write_tal_attrs."""
    violations: list[str] = []
    for path in _iter_python_files():
        lines = _tal_attr_write_lines(path)
        if not lines or path.resolve() in APPROVED_WRITERS:
            continue
        rel = path.relative_to(REPO_ROOT)
        locations = ",".join(str(line) for line in sorted(lines))
        violations.append(f"{rel}:{locations}")
    assert not violations, (
        "Direct attrs['tal'] writes are restricted to canonical writer modules. "
        f"Violations: {violations}"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("attrs['tal'] = {}", [1]),
        ("attrs['tal']['x'] = 1", [1]),
        ("attrs_alias = attrs\nattrs_alias['tal'] = {}", [2]),
        ("attrs.setdefault('tal', {})", [1]),
        ("attrs.update({'tal': {}})", [1]),
        ("attrs_alias = attrs\nattrs_alias.update({'tal': {}})", [2]),
        ("attrs.update(tal={})", [1]),
        ("tal_schema = attrs['tal']\ntal_schema['x'] = 1", [2]),
        ("attrs['tal'].update({'x': 1})", [1]),
    ],
)
def test_arch_schemawrite_002_detector_flags_indirect_patterns(
    source: str, expected: list[int]
) -> None:
    """ID: ARCH_SCHEMAWRITE_002_detector_flags_indirect_patterns."""
    assert _tal_write_lines_from_source(source) == expected


def test_arch_schemawrite_003_detector_ignores_read_only_alias() -> None:
    """ID: ARCH_SCHEMAWRITE_003_detector_ignores_read_only_alias."""
    source_tal = "tal_schema = attrs['tal']\nvalue = tal_schema.get('x')\n"
    source_attrs = "attrs_alias = attrs\nvalue = attrs_alias.get('tal')\n"
    assert _tal_write_lines_from_source(source_tal) == []
    assert _tal_write_lines_from_source(source_attrs) == []


def test_arch_schemawrite_004_detector_flags_attrs_alias_writes() -> None:
    """ID: ARCH_SCHEMAWRITE_004_detector_flags_attrs_alias_writes."""
    source = "attrs_alias = attrs\nattrs_alias['tal'] = {}\n"
    assert _tal_write_lines_from_source(source) == [2]


def test_arch_schemawrite_005_new_cleanup_owner_modules_remain_schema_write_free() -> None:
    """ID: ARCH_SCHEMAWRITE_005_new_cleanup_owner_modules_remain_schema_write_free."""
    paths = [
        REPO_ROOT / "tal" / "core" / "orchestration" / "context.py",
        REPO_ROOT / "tal" / "core" / "orchestration" / "schema_finalize.py",
        REPO_ROOT / "tal" / "core" / "validity_finalize.py",
        REPO_ROOT / "tal" / "core" / "event_ops" / "finalize.py",
        REPO_ROOT / "tal" / "linalg" / "component_context.py",
    ]
    for path in paths:
        assert path.exists(), path
        assert _tal_attr_write_lines(path) == [], path
