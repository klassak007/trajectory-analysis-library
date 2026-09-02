"""Shared lexical-binding primitives for architecture guards."""

from __future__ import annotations

import ast


def argument_names(args: ast.arguments) -> set[str]:
    positional = (*args.posonlyargs, *args.args, *args.kwonlyargs)
    names = {arg.arg for arg in positional}
    names.update(arg.arg for arg in (args.vararg, args.kwarg) if arg is not None)
    return names


def target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return target_names(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        return set().union(*(target_names(item) for item in target.elts))
    return set()


def function_signature_expressions(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    expressions: list[ast.AST] = [*node.decorator_list]
    expressions.extend(default for default in node.args.defaults)
    expressions.extend(default for default in node.args.kw_defaults if default is not None)
    arguments = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        node.args.vararg,
        node.args.kwarg,
    )
    expressions.extend(
        argument.annotation
        for argument in arguments
        if argument is not None and argument.annotation is not None
    )
    if node.returns is not None:
        expressions.append(node.returns)
    expressions.extend(getattr(node, "type_params", ()))
    return tuple(expressions)


def class_outer_expressions(node: ast.ClassDef) -> tuple[ast.AST, ...]:
    expressions: list[ast.AST] = [*node.decorator_list, *node.bases]
    expressions.extend(keyword.value for keyword in node.keywords)
    expressions.extend(getattr(node, "type_params", ()))
    return tuple(expressions)


def pattern_names(pattern: ast.pattern) -> set[str]:
    names = {
        node.name
        for node in ast.walk(pattern)
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name is not None
    }
    names.update(
        node.rest
        for node in ast.walk(pattern)
        if isinstance(node, ast.MatchMapping) and node.rest is not None
    )
    return names


class _FunctionBindingCollector(ast.NodeVisitor):
    """Collect bindings in one function without entering nested scopes."""

    def __init__(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.root = node
        self.local_names = argument_names(node.args)
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()

    def collect(self) -> set[str]:
        for statement in self.root.body:
            self.visit(statement)
        return self.local_names - self.global_names - self.nonlocal_names

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.local_names.add(node.id)

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        self.local_names.update(
            alias.asname or alias.name.partition(".")[0] for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.local_names.update(
            alias.asname or alias.name for alias in node.names if alias.name != "*"
        )

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.local_names.add(node.name)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for case in node.cases:
            self.local_names.update(pattern_names(case.pattern))
            if case.guard is not None:
                self.visit(case.guard)
            for statement in case.body:
                self.visit(statement)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.local_names.add(node.name)
        for expression in function_signature_expressions(node):
            self.visit(expression)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.local_names.add(node.name)
        for expression in class_outer_expressions(node):
            self.visit(expression)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def _visit_comprehension(self, node: ast.AST, values: tuple[ast.AST, ...]) -> None:
        for value in values:
            self.visit(value)
        for generator in node.generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, (node.elt,))

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, (node.key, node.value))


def function_local_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    return _FunctionBindingCollector(node).collect()


__all__ = [
    "argument_names",
    "class_outer_expressions",
    "function_local_names",
    "function_signature_expressions",
    "pattern_names",
    "target_names",
]
