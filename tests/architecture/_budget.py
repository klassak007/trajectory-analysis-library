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
    out: dict[str, int] = {}
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = function_loc(node, source=text)
    return out
