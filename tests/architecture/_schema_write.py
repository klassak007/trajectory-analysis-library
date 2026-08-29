from __future__ import annotations

import ast
from enum import IntFlag, auto
from pathlib import Path


class _Origin(IntFlag):
    NONE = 0
    ATTRS_ROOT = auto()
    ATTRS_COPY = auto()
    TAL_ROOT = auto()
    TAL_COPY = auto()
    TAL_MAPPING = auto()


_FlowState = dict[str, _Origin]


class _LexicalChildScopeCollector(ast.NodeVisitor):
    """Collect immediately nested lexical scopes without entering them."""

    def __init__(self) -> None:
        self.nodes: list[
            ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
        ] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.nodes.append(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.nodes.append(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.nodes.append(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.nodes.append(node)


def _lexical_child_scopes(
    block: list[ast.stmt],
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda, ...]:
    collector = _LexicalChildScopeCollector()
    for statement in block:
        collector.visit(statement)
    return tuple(collector.nodes)


def _is_tal_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "tal"


def _argument_names(args: ast.arguments) -> set[str]:
    positional = (*args.posonlyargs, *args.args, *args.kwonlyargs)
    names = {arg.arg for arg in positional}
    names.update(arg.arg for arg in (args.vararg, args.kwarg) if arg is not None)
    return names


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        return set().union(*(_target_names(item) for item in target.elts))
    return set()


def _function_signature_expressions(
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


def _class_outer_expressions(node: ast.ClassDef) -> tuple[ast.AST, ...]:
    expressions: list[ast.AST] = [*node.decorator_list, *node.bases]
    expressions.extend(keyword.value for keyword in node.keywords)
    expressions.extend(getattr(node, "type_params", ()))
    return tuple(expressions)


def _pattern_names(pattern: ast.pattern) -> set[str]:
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
    """Collect names bound in one function without entering nested scopes."""

    def __init__(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.root = node
        self.local_names = _argument_names(node.args)
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
            self.local_names.update(_pattern_names(case.pattern))
            if case.guard is not None:
                self.visit(case.guard)
            for statement in case.body:
                self.visit(statement)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.local_names.add(node.name)
        for expression in _function_signature_expressions(node):
            self.visit(expression)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.local_names.add(node.name)
        for expression in _class_outer_expressions(node):
            self.visit(expression)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def _visit_comprehension(self, node: ast.AST, values: list[ast.AST]) -> None:
        for value in values:
            self.visit(value)
        for generator in node.generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, [node.key, node.value])


def _function_local_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return _FunctionBindingCollector(node).collect()


class _TalWriteDetector(ast.NodeVisitor):
    """Find direct writes to the TAL schema, including local attrs rebuilds."""

    def __init__(self) -> None:
        self.lines: set[int] = set()
        self._origin_scopes: list[_FlowState] = [{"attrs": _Origin.ATTRS_ROOT}]
        self._potential_scopes: list[_FlowState] = [{"attrs": _Origin.ATTRS_ROOT}]
        self._scope_kinds = ["normal"]
        self._scope_locals: list[set[str]] = [set()]

    @property
    def origins(self) -> _FlowState:
        return self._origin_scopes[-1]

    def _push_scope(
        self,
        *,
        kind: str = "normal",
        local_names: set[str] | None = None,
        inherited: _FlowState | None = None,
    ) -> None:
        base = self.origins if inherited is None else inherited
        self._origin_scopes.append(dict(base))
        self._potential_scopes.append(dict(base))
        self._scope_kinds.append(kind)
        self._scope_locals.append(set() if local_names is None else local_names)

    def _pop_scope(self) -> None:
        self._origin_scopes.pop()
        self._potential_scopes.pop()
        self._scope_kinds.pop()
        self._scope_locals.pop()

    def _snapshot_state(self) -> _FlowState:
        return dict(self.origins)

    def _restore_state(self, state: _FlowState) -> None:
        self.origins.clear()
        self.origins.update(state)

    def _join_states(self, states: list[_FlowState]) -> _FlowState:
        joined: _FlowState = {}
        for state in states:
            for name, origin in state.items():
                joined[name] = joined.get(name, _Origin.NONE) | origin
        return joined

    def _visit_block_from(self, state: _FlowState, block: list[ast.stmt]) -> _FlowState:
        self._restore_state(state)
        for statement in block:
            self.visit(statement)
        return self._snapshot_state()

    def _clear_names(self, names: set[str]) -> None:
        for name in names:
            self.origins.pop(name, None)

    def _clear_potential_names(self, names: set[str]) -> None:
        for name in names:
            self._potential_scopes[-1].pop(name, None)

    def _set_name_origin(self, name: str, origin: _Origin) -> None:
        index = len(self._origin_scopes) - 1
        while True:
            state = self._origin_scopes[index]
            if origin == _Origin.NONE:
                state.pop(name, None)
            else:
                state[name] = origin
                potential = self._potential_scopes[index]
                potential[name] = potential.get(name, _Origin.NONE) | origin
            if (
                self._scope_kinds[index] != "comprehension"
                or name in self._scope_locals[index]
            ):
                return
            index -= 1

    def _add_name_origin(self, name: str, origin: _Origin) -> None:
        self._set_name_origin(name, self.origins.get(name, _Origin.NONE) | origin)

    def _mapping_copy_origins(self, origin: _Origin) -> _Origin:
        copied = _Origin.NONE
        if origin & (_Origin.ATTRS_ROOT | _Origin.ATTRS_COPY):
            copied |= _Origin.ATTRS_COPY
        if origin & (_Origin.TAL_ROOT | _Origin.TAL_COPY):
            copied |= _Origin.TAL_COPY
        if origin & _Origin.TAL_MAPPING:
            copied |= _Origin.TAL_MAPPING
        return copied

    def _dict_origins(self, expr: ast.Dict) -> _Origin:
        origin = _Origin.NONE
        for key, value in zip(expr.keys, expr.values, strict=True):
            if _is_tal_literal(key):
                origin |= _Origin.TAL_MAPPING
            elif key is None:
                origin |= self._mapping_copy_origins(self._expr_origins(value))
        return origin

    def _call_origins(self, expr: ast.Call) -> _Origin:
        origin = (
            _Origin.TAL_MAPPING
            if any(keyword.arg == "tal" for keyword in expr.keywords)
            else _Origin.NONE
        )
        if isinstance(expr.func, ast.Name) and expr.func.id == "dict":
            for argument in expr.args:
                origin |= self._mapping_copy_origins(self._expr_origins(argument))
            return origin
        if not isinstance(expr.func, ast.Attribute):
            return origin
        receiver = self._expr_origins(expr.func.value)
        if expr.func.attr == "copy":
            return origin | self._mapping_copy_origins(receiver)
        if expr.func.attr != "get":
            return origin
        if receiver & (_Origin.ATTRS_ROOT | _Origin.ATTRS_COPY):
            if self._key_call_targets_tal(expr):
                origin |= _Origin.TAL_ROOT
        elif receiver & (_Origin.TAL_ROOT | _Origin.TAL_COPY):
            if expr.args or any(keyword.arg == "key" for keyword in expr.keywords):
                origin |= _Origin.TAL_ROOT
        return origin

    def _expr_origins(self, expr: ast.AST) -> _Origin:
        if isinstance(expr, ast.Name):
            return self.origins.get(expr.id, _Origin.NONE)
        if isinstance(expr, ast.Attribute):
            return _Origin.ATTRS_ROOT if expr.attr == "attrs" else _Origin.NONE
        if isinstance(expr, ast.NamedExpr):
            return self._expr_origins(expr.value)
        if isinstance(expr, ast.Subscript):
            receiver = self._expr_origins(expr.value)
            if (
                receiver & (_Origin.ATTRS_ROOT | _Origin.ATTRS_COPY)
                and _is_tal_literal(expr.slice)
            ):
                return _Origin.TAL_ROOT
            if receiver & (_Origin.TAL_ROOT | _Origin.TAL_COPY):
                return _Origin.TAL_ROOT
            return _Origin.NONE
        if isinstance(expr, ast.Dict):
            return self._dict_origins(expr)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            sources = self._expr_origins(expr.left) | self._expr_origins(expr.right)
            return self._mapping_copy_origins(sources)
        if isinstance(expr, ast.IfExp):
            return self._expr_origins(expr.body) | self._expr_origins(expr.orelse)
        if isinstance(expr, ast.BoolOp):
            origin = _Origin.NONE
            for value in expr.values:
                origin |= self._expr_origins(value)
            return origin
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            contains_tal = any(
                isinstance(item, (ast.List, ast.Tuple))
                and len(item.elts) == 2
                and _is_tal_literal(item.elts[0])
                for item in expr.elts
            )
            return _Origin.TAL_MAPPING if contains_tal else _Origin.NONE
        if isinstance(expr, ast.Call):
            return self._call_origins(expr)
        return _Origin.NONE

    def _expr_is_attrs_root(self, expr: ast.AST) -> bool:
        return bool(self._expr_origins(expr) & _Origin.ATTRS_ROOT)

    def _is_direct_tal_subscript(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Subscript)
            and self._expr_is_attrs_root(node.value)
            and _is_tal_literal(node.slice)
        )

    def _is_local_tal_subscript(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Subscript) or not _is_tal_literal(node.slice):
            return False
        return isinstance(node.value, ast.Name)

    def _expr_is_tal_root(self, expr: ast.AST) -> bool:
        return bool(self._expr_origins(expr) & _Origin.TAL_ROOT)

    def _target_writes_tal(self, target: ast.AST) -> bool:
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(self._target_writes_tal(item) for item in target.elts)
        if not isinstance(target, ast.Subscript):
            return False
        return self._is_direct_tal_subscript(target) or self._expr_is_tal_root(target.value)

    def _expr_has_tal_mapping(self, expr: ast.AST) -> bool:
        tal_mapping_origins = (
            _Origin.ATTRS_ROOT | _Origin.ATTRS_COPY | _Origin.TAL_MAPPING
        )
        return bool(self._expr_origins(expr) & tal_mapping_origins)

    def _call_supplies_tal(self, call: ast.Call) -> bool:
        if any(keyword.arg == "tal" for keyword in call.keywords):
            return True
        return any(self._expr_has_tal_mapping(arg) for arg in call.args)

    def _key_call_targets_tal(self, call: ast.Call) -> bool:
        if call.args and _is_tal_literal(call.args[0]):
            return True
        return any(
            keyword.arg == "key" and _is_tal_literal(keyword.value)
            for keyword in call.keywords
        )

    def _attrs_call_writes_tal(self, call: ast.Call) -> bool:
        if not isinstance(call.func, ast.Attribute):
            return False
        if call.func.attr == "assign_attrs":
            return self._call_supplies_tal(call)
        if not self._expr_is_attrs_root(call.func.value):
            return False
        if call.func.attr == "update":
            return self._call_supplies_tal(call)
        if call.func.attr in {"__setitem__", "__delitem__", "pop", "setdefault"}:
            return self._key_call_targets_tal(call)
        if call.func.attr == "__ior__":
            return self._call_supplies_tal(call)
        return call.func.attr in {"clear", "popitem"}

    def _tal_root_mutating_call(self, call: ast.Call) -> bool:
        return (
            isinstance(call.func, ast.Attribute)
            and call.func.attr
            in {
                "__delitem__",
                "__ior__",
                "__setitem__",
                "clear",
                "pop",
                "popitem",
                "setdefault",
                "update",
            }
            and self._expr_is_tal_root(call.func.value)
        )

    def _local_call_changes_tal(self, call: ast.Call) -> str | None:
        if not isinstance(call.func, ast.Attribute):
            return None
        receiver = call.func.value
        if not isinstance(receiver, ast.Name):
            return None
        name = receiver.id
        if call.func.attr == "update" and self._call_supplies_tal(call):
            return name
        if call.func.attr == "__ior__" and self._call_supplies_tal(call):
            return name
        if call.func.attr in {"__setitem__", "__delitem__", "pop", "setdefault"}:
            return name if self._key_call_targets_tal(call) else None
        origin = self.origins.get(name, _Origin.NONE)
        if call.func.attr in {"clear", "popitem"} and origin & (
            _Origin.ATTRS_COPY | _Origin.TAL_MAPPING
        ):
            return name
        return None

    def _mark_local_tal_target(self, target: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._mark_local_tal_target(item)
            return
        candidate = target
        while isinstance(candidate, ast.Subscript):
            if self._is_local_tal_subscript(candidate):
                assert isinstance(candidate.value, ast.Name)
                self._add_name_origin(candidate.value.id, _Origin.TAL_MAPPING)
                return
            candidate = candidate.value

    def _assignment_bindings(
        self, target: ast.AST, value: ast.AST
    ) -> list[tuple[str, _Origin]]:
        if isinstance(target, ast.Name):
            return [(target.id, self._expr_origins(value))]
        if isinstance(target, ast.Starred):
            return [(name, _Origin.NONE) for name in _target_names(target)]
        if not isinstance(target, (ast.Tuple, ast.List)):
            return []
        if not isinstance(value, (ast.Tuple, ast.List)):
            return [(name, _Origin.NONE) for name in _target_names(target)]
        if any(isinstance(item, ast.Starred) for item in value.elts):
            return [(name, _Origin.NONE) for name in _target_names(target)]
        starred = [
            index
            for index, item in enumerate(target.elts)
            if isinstance(item, ast.Starred)
        ]
        if not starred:
            if len(target.elts) != len(value.elts):
                return [(name, _Origin.NONE) for name in _target_names(target)]
            pairs = zip(target.elts, value.elts, strict=True)
        else:
            star_index = starred[0]
            suffix_count = len(target.elts) - star_index - 1
            if len(value.elts) < len(target.elts) - 1:
                return [(name, _Origin.NONE) for name in _target_names(target)]
            target_suffix = target.elts[-suffix_count:] if suffix_count else ()
            value_suffix = value.elts[-suffix_count:] if suffix_count else ()
            pairs = zip(
                (*target.elts[:star_index], *target_suffix),
                (*value.elts[:star_index], *value_suffix),
                strict=True,
            )
        bindings: list[tuple[str, _Origin]] = []
        for item_target, item_value in pairs:
            bindings.extend(self._assignment_bindings(item_target, item_value))
        if starred:
            bindings.extend(self._assignment_bindings(target.elts[starred[0]], value))
        return bindings

    def _bind_ordinary_target(self, target: ast.AST) -> None:
        if self._target_writes_tal(target):
            self.lines.add(target.lineno)
        self._mark_local_tal_target(target)
        self._clear_names(_target_names(target))

    def _assignment_bindings_for_targets(
        self, targets: list[ast.AST], value: ast.AST
    ) -> list[tuple[str, _Origin]]:
        return [
            binding
            for target in targets
            for binding in self._assignment_bindings(target, value)
        ]

    def _visit_assignment(
        self,
        targets: list[ast.AST],
        value: ast.AST,
        bindings: list[tuple[str, _Origin]],
    ) -> None:
        for target in targets:
            if self._target_writes_tal(target):
                self.lines.add(target.lineno)
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "attrs"
                and self._expr_has_tal_mapping(value)
            ):
                self.lines.add(target.lineno)
            self._mark_local_tal_target(target)
        for name, origin in bindings:
            self._set_name_origin(name, origin)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        bindings = self._assignment_bindings_for_targets(list(node.targets), node.value)
        for target in node.targets:
            self.visit(target)
        self._visit_assignment(list(node.targets), node.value, bindings)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            bindings = self._assignment_bindings_for_targets([node.target], node.value)
        self.visit(node.target)
        if node.value is not None:
            self._visit_assignment([node.target], node.value, bindings)
        if node.annotation is not None:
            self.visit(node.annotation)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        bindings = self._assignment_bindings_for_targets([node.target], node.value)
        self.visit(node.target)
        self._visit_assignment([node.target], node.value, bindings)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        target_writes_tal = self._target_writes_tal(node.target)
        target_origin = self._expr_origins(node.target)
        target_is_attrs = (
            isinstance(node.target, ast.Attribute) and node.target.attr == "attrs"
        )
        self.visit(node.target)
        self.visit(node.value)
        if target_writes_tal or target_origin & _Origin.TAL_ROOT:
            self.lines.add(node.lineno)
        supplies_tal = self._expr_has_tal_mapping(node.value)
        if supplies_tal and (
            target_is_attrs or target_origin & _Origin.ATTRS_ROOT
        ):
            self.lines.add(node.lineno)
        self._mark_local_tal_target(node.target)
        if isinstance(node.target, ast.Name):
            origin = target_origin
            if isinstance(node.op, ast.BitOr) and supplies_tal:
                origin |= _Origin.TAL_MAPPING
            self._set_name_origin(node.target.id, origin)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if self._target_writes_tal(target):
                self.lines.add(node.lineno)
            self.visit(target)
            self._clear_names(_target_names(target))

    def visit_Call(self, node: ast.Call) -> None:
        if self._attrs_call_writes_tal(node) or self._tal_root_mutating_call(node):
            self.lines.add(node.lineno)
        changed = self._local_call_changes_tal(node)
        if changed is not None:
            self._add_name_origin(changed, _Origin.TAL_MAPPING)
        self.generic_visit(node)

    def _visit_expression_from(self, state: _FlowState, expression: ast.AST) -> _FlowState:
        self._restore_state(state)
        self.visit(expression)
        return self._snapshot_state()

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        base = self._snapshot_state()
        states = [
            self._visit_expression_from(base, node.body),
            self._visit_expression_from(base, node.orelse),
        ]
        self._restore_state(self._join_states(states))

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        states: list[_FlowState] = []
        state = self._snapshot_state()
        for value in node.values:
            state = self._visit_expression_from(state, value)
            states.append(state)
        self._restore_state(self._join_states(states))

    def visit_Compare(self, node: ast.Compare) -> None:
        self.visit(node.left)
        states: list[_FlowState] = []
        state = self._snapshot_state()
        for comparator in node.comparators:
            state = self._visit_expression_from(state, comparator)
            states.append(state)
        if states:
            self._restore_state(self._join_states(states))

    def visit_Import(self, node: ast.Import) -> None:
        self._clear_names(
            {alias.asname or alias.name.partition(".")[0] for alias in node.names}
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._clear_names(
            {
                alias.asname or alias.name
                for alias in node.names
                if alias.name != "*"
            }
        )

    def _visit_comprehension(self, node: ast.AST, values: list[ast.AST]) -> None:
        first, *remaining = node.generators
        self.visit(first.iter)
        local_names = set().union(
            *(_target_names(generator.target) for generator in node.generators)
        )
        self._push_scope(kind="comprehension", local_names=local_names)
        self._clear_names(local_names)
        self._clear_potential_names(local_names)
        for condition in first.ifs:
            self.visit(condition)
        for generator in remaining:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for value in values:
            self.visit(value)
        self._pop_scope()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, [node.key, node.value])

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        base = self._snapshot_state()
        states = [
            self._visit_block_from(base, node.body),
            self._visit_block_from(base, node.orelse),
        ]
        self._restore_state(self._join_states(states))

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        base = self._snapshot_state()
        states = [base]
        for case in node.cases:
            self._restore_state(base)
            self._clear_names(_pattern_names(case.pattern))
            self.visit(case.pattern)
            if case.guard is not None:
                self.visit(case.guard)
            states.append(self._visit_block_from(self._snapshot_state(), case.body))
        self._restore_state(self._join_states(states))

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        base = self._snapshot_state()
        entry = base
        while True:
            self._restore_state(entry)
            self._bind_ordinary_target(node.target)
            body = self._visit_block_from(self._snapshot_state(), node.body)
            joined = self._join_states([base, body])
            if joined == entry:
                break
            entry = joined
        normal_exit = self._visit_block_from(entry, node.orelse)
        self._restore_state(self._join_states([entry, body, normal_exit]))

    visit_AsyncFor = visit_For

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_ordinary_target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def visit_While(self, node: ast.While) -> None:
        base = self._snapshot_state()
        entry = base
        while True:
            tested = self._visit_expression_from(entry, node.test)
            body = self._visit_block_from(tested, node.body)
            joined = self._join_states([base, body])
            if joined == entry:
                break
            entry = joined
        normal_exit = self._visit_block_from(tested, node.orelse)
        self._restore_state(self._join_states([tested, body, normal_exit]))

    def visit_Try(self, node: ast.Try) -> None:
        base = self._snapshot_state()
        body = self._visit_block_from(base, node.body)
        success = self._visit_block_from(body, node.orelse)
        handler_base = self._join_states([base, body])
        states = [success]
        for handler in node.handlers:
            self._restore_state(handler_base)
            if handler.type is not None:
                self.visit(handler.type)
            if handler.name is not None:
                self._clear_names({handler.name})
            for statement in handler.body:
                self.visit(statement)
            if handler.name is not None:
                self._clear_names({handler.name})
            states.append(self._snapshot_state())
        if node.finalbody:
            states = [self._visit_block_from(state, node.finalbody) for state in states]
        self._restore_state(self._join_states(states))

    visit_TryStar = visit_Try

    def _visit_function_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for expression in _function_signature_expressions(node):
            self.visit(expression)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._visit_function_signature(node)
        inherited = None
        if self._scope_kinds[-1] == "class":
            inherited = self._origin_scopes[-2]
        self._push_scope(inherited=inherited)
        local_names = _function_local_names(node) | {node.name}
        self._clear_names(local_names)
        self._clear_potential_names(local_names)
        for statement in node.body:
            self.visit(statement)
        self._visit_child_scopes_from_potential(node.body)
        self._pop_scope()
        self._clear_names({node.name})

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in _class_outer_expressions(node):
            self.visit(expression)
        self._push_scope(kind="class")
        for statement in node.body:
            self.visit(statement)
        self._visit_child_scopes_from_potential(node.body)
        self._pop_scope()
        self._clear_names({node.name})

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        inherited = None
        if self._scope_kinds[-1] == "class":
            inherited = self._origin_scopes[-2]
        self._push_scope(inherited=inherited)
        argument_names = _argument_names(node.args)
        self._clear_names(argument_names)
        self._clear_potential_names(argument_names)
        self.visit(node.body)
        self._pop_scope()

    def _visit_child_scopes_from_potential(self, block: list[ast.stmt]) -> None:
        final = self._snapshot_state()
        potential = dict(self._potential_scopes[-1])
        for scope in _lexical_child_scopes(block):
            self._restore_state(potential)
            self.visit(scope)
        self._restore_state(final)

    def visit_Module(self, node: ast.Module) -> None:
        for statement in node.body:
            self.visit(statement)
        self._visit_child_scopes_from_potential(node.body)


def tal_write_lines_from_source(source: str) -> list[int]:
    """Return source lines that directly write or replace the TAL schema."""
    detector = _TalWriteDetector()
    detector.visit(ast.parse(source))
    return sorted(detector.lines)


def tal_attr_write_lines(path: Path) -> list[int]:
    """Return direct TAL-schema write lines in ``path``."""
    return tal_write_lines_from_source(path.read_text(encoding="utf-8"))


def has_tal_schema_write(source: str) -> bool:
    """Return whether source directly writes or replaces the TAL schema."""
    return bool(tal_write_lines_from_source(source))
