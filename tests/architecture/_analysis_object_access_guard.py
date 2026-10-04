"""Direct-syntax checks for the AnalysisObject Dataset ownership boundary.

The scanner recognizes only direct attribute access and a literal
``as_dataset(copy="none")`` keyword. It deliberately does not infer receiver
types, follow aliases, evaluate annotations, or model Python control flow.
Observable ownership semantics belong to the public runtime tests.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

_ANALYSIS_OBJECT_PATH = "tal/core/analysis_object.py"
_DATASET_OWNER_PATH = "tal/core/dataset_ownership.py"


@dataclass(frozen=True)
class AccessViolation:
    """One direct ownership-boundary violation."""

    path: str
    line: int
    access: str


def _path_suffix(path: str) -> str:
    normalized = Path(path).as_posix()
    marker = normalized.find("tal/")
    return normalized[marker:] if marker >= 0 else normalized


def _is_self_attribute(node: ast.Attribute) -> bool:
    return isinstance(node.value, ast.Name) and node.value.id == "self"


def _is_literal_raw_view(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "as_dataset":
        return False
    return any(
        keyword.arg == "copy"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "none"
        for keyword in node.keywords
    )


class _DirectAccessVisitor(ast.NodeVisitor):
    def __init__(self, *, path: str) -> None:
        self.path = _path_suffix(path)
        self.violations: list[AccessViolation] = []
        self._classes: list[str] = []

    def _record(self, node: ast.AST, access: str) -> None:
        self.violations.append(
            AccessViolation(self.path, getattr(node, "lineno", 0), access)
        )

    def _analysis_object_access_is_allowed(self, node: ast.Attribute) -> bool:
        return self._classes[-1:] == ["AnalysisObject"] and _is_self_attribute(node)

    def _raw_backing_access_is_allowed(self, node: ast.Attribute) -> bool:
        if self.path == _DATASET_OWNER_PATH:
            return isinstance(node.ctx, ast.Load)
        if self.path == _ANALYSIS_OBJECT_PATH:
            return self._analysis_object_access_is_allowed(node)
        return False

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "unsafe_data":
            self._record(node, "unsafe_data")
        elif node.attr == "_data" and not self._raw_backing_access_is_allowed(node):
            self._record(node, "_data")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_literal_raw_view(node):
            self._record(node, 'as_dataset(copy="none")')
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._classes.append(node.name)
        self.generic_visit(node)
        self._classes.pop()


def analysis_object_access_violations(
    source: str,
    *,
    path: str = "<source>",
) -> tuple[AccessViolation, ...]:
    """Return violations visible from the scanner's direct-syntax boundary."""
    visitor = _DirectAccessVisitor(path=path)
    visitor.visit(ast.parse(source))
    return tuple(visitor.violations)


def production_access_violations() -> tuple[AccessViolation, ...]:
    """Audit Python production sources under the direct ownership convention."""
    violations: list[AccessViolation] = []
    for path in sorted(Path("tal").rglob("*.py")):
        violations.extend(
            analysis_object_access_violations(
                path.read_text(encoding="utf-8"),
                path=path.as_posix(),
            )
        )
    return tuple(violations)


__all__ = [
    "AccessViolation",
    "analysis_object_access_violations",
    "production_access_violations",
]
