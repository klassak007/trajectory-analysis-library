from __future__ import annotations

import ast
from collections.abc import Iterable
from functools import lru_cache
from io import StringIO
from pathlib import Path
import tokenize


def _iter_docstring_nodes(node: ast.AST) -> Iterable[ast.AST]:
    if isinstance(node, ast.Module):
        body = node.body
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = node.body
    else:
        return
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            yield body[0]
    for child in ast.iter_child_nodes(node):
        yield from _iter_docstring_nodes(child)


def _line_range(node: ast.AST) -> range:
    end = getattr(node, "end_lineno", None) or getattr(node, "lineno", None)
    start = getattr(node, "lineno", None)
    if start is None or end is None:
        return range(0)
    return range(start, end + 1)


def _comment_lines(source: str) -> set[int]:
    out: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                out.add(tok.start[0])
    except tokenize.TokenError:
        return out
    return out


def _excluded_lines(source: str) -> set[int]:
    module = ast.parse(source)
    out = _comment_lines(source)
    for node in _iter_docstring_nodes(module):
        out.update(_line_range(node))
    return out


@lru_cache(maxsize=512)
def _read_source(path_like: str) -> str:
    return Path(path_like).read_text(encoding="utf-8")


def _normalize_source(*, path: str | Path | None = None, source: str | None = None) -> str:
    if source is not None:
        return source
    if path is None:
        raise ValueError("budget helper requires either `path` or `source`.")
    return _read_source(str(path))


def file_loc(*, path: str | Path | None = None, source: str | None = None) -> int:
    text = _normalize_source(path=path, source=source)
    excluded = _excluded_lines(text)
    return sum(1 for idx, _ in enumerate(text.splitlines(), start=1) if idx not in excluded)


def executable_source(*, path: str | Path | None = None, source: str | None = None) -> str:
    text = _normalize_source(path=path, source=source)
    excluded = _excluded_lines(text)
    return "\n".join(
        line
        for idx, line in enumerate(text.splitlines(), start=1)
        if idx not in excluded
    )


def function_loc(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    path: str | Path | None = None,
    source: str | None = None,
) -> int:
    text = _normalize_source(path=path, source=source)
    excluded = _excluded_lines(text)
    span = _line_range(node)
    return sum(1 for idx in span if idx not in excluded)


def function_lengths(path: str | Path) -> dict[str, int]:
    text = _normalize_source(path=path)
    module = ast.parse(text)
    return {
        name: function_loc(node, source=text)
        for name, node in _qualified_function_nodes(module).items()
    }


class _QualifiedFunctionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self._scope: list[str] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        key = ".".join((*self._scope, node.name))
        if key in self.nodes:
            key = f"{key}@{node.lineno}"
        self.nodes[key] = node
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


def _qualified_function_nodes(
    module: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    visitor = _QualifiedFunctionVisitor()
    visitor.visit(module)
    return visitor.nodes


def function_parameter_counts(
    path: str | Path | None = None,
    *,
    source: str | None = None,
) -> dict[str, int]:
    """Return declared parameter counts under qualified function names."""
    module = ast.parse(_normalize_source(path=path, source=source))
    return {
        name: (
            len(node.args.posonlyargs)
            + len(node.args.args)
            + len(node.args.kwonlyargs)
            + int(node.args.vararg is not None)
            + int(node.args.kwarg is not None)
        )
        for name, node in _qualified_function_nodes(module).items()
    }


_CONTROL_NODES = (
    ast.For,
    ast.AsyncFor,
    ast.If,
    ast.Match,
    ast.match_case,
    ast.Try,
    ast.TryStar,
    ast.While,
    ast.With,
    ast.AsyncWith,
)


def _control_depth(node: ast.AST, *, depth: int, root: ast.AST) -> int:
    maximum = depth
    for child in ast.iter_child_nodes(node):
        if child is not root and isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
        ):
            continue
        child_depth = depth + 1 if isinstance(child, _CONTROL_NODES) else depth
        is_elif = (
            isinstance(node, ast.If)
            and isinstance(child, ast.If)
            and node.orelse == [child]
        )
        if is_elif:
            child_depth = depth
        maximum = max(
            maximum,
            _control_depth(child, depth=child_depth, root=root),
        )
    return maximum


def function_control_depths(
    path: str | Path | None = None,
    *,
    source: str | None = None,
) -> dict[str, int]:
    """Return maximum nested control-flow depth for each function."""
    module = ast.parse(_normalize_source(path=path, source=source))
    return {
        name: _control_depth(node, depth=0, root=node)
        for name, node in _qualified_function_nodes(module).items()
    }
