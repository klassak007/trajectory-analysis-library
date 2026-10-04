from __future__ import annotations

import ast
from collections.abc import Callable
from enum import IntFlag, auto
from pathlib import Path

from ._ast_scope import (
    argument_names as _argument_names,
    class_outer_expressions as _class_outer_expressions,
    function_local_names as _function_local_names,
    function_signature_expressions as _function_signature_expressions,
    pattern_names as _pattern_names,
    target_names as _target_names,
)


class _Origin(IntFlag):
    NONE = 0
    ATTRS_ROOT = auto()
    ATTRS_COPY = auto()
    TAL_ROOT = auto()
    TAL_COPY = auto()
    TAL_MAPPING = auto()
    COPY_MODULE = auto()
    COPY_FUNCTION = auto()
    XARRAY_MODULE = auto()
    XARRAY_DATASET = auto()
    XARRAY_DATA_ARRAY = auto()
    XARRAY_VARIABLE = auto()
    XARRAY_INDEX_VARIABLE = auto()


_FlowState = dict[str, _Origin]

_XARRAY_CONSTRUCTORS = {
    "Dataset": _Origin.XARRAY_DATASET,
    "DataArray": _Origin.XARRAY_DATA_ARRAY,
    "Variable": _Origin.XARRAY_VARIABLE,
    "IndexVariable": _Origin.XARRAY_INDEX_VARIABLE,
}
_XARRAY_ATTRS_POSITIONS = {
    _Origin.XARRAY_DATASET: 2,
    _Origin.XARRAY_DATA_ARRAY: 4,
    _Origin.XARRAY_VARIABLE: 2,
    _Origin.XARRAY_INDEX_VARIABLE: 2,
}


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


def _is_tal_pair(node: ast.AST) -> bool:
    return (
        isinstance(node, (ast.List, ast.Tuple))
        and len(node.elts) == 2
        and _is_tal_literal(node.elts[0])
    )


class _TalWriteDetector(ast.NodeVisitor):
    """Find direct writes to the TAL schema, including local attrs rebuilds."""

    def __init__(self) -> None:
        self.lines: set[int] = set()
        builtins = {
            "attrs": _Origin.ATTRS_ROOT,
            "xr": _Origin.XARRAY_MODULE,
            "xarray": _Origin.XARRAY_MODULE,
        }
        self._origin_scopes: list[_FlowState] = [dict(builtins)]
        self._potential_scopes: list[_FlowState] = [dict(builtins)]
        self._scope_kinds = ["normal"]
        self._scope_locals: list[set[str]] = [set()]
        self._loop_break_states: list[list[_FlowState]] = []
        self._loop_continue_states: list[list[_FlowState]] = []

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

    def _visit_block_with_entry_states(
        self,
        state: _FlowState,
        block: list[ast.stmt],
    ) -> tuple[_FlowState, list[_FlowState]]:
        self._restore_state(state)
        entries: list[_FlowState] = []
        for statement in block:
            entries.append(self._snapshot_state())
            self.visit(statement)
        return self._snapshot_state(), entries

    def _solve_loop(
        self,
        base: _FlowState,
        step: Callable[[_FlowState], tuple[_FlowState, _FlowState]],
    ) -> tuple[_FlowState, _FlowState, list[_FlowState]]:
        entry = base
        self._loop_break_states.append([])
        self._loop_continue_states.append([])
        try:
            while True:
                exit_state, body = step(entry)
                joined = self._join_states(
                    [base, body, *self._loop_continue_states[-1]]
                )
                if joined == entry:
                    return exit_state, body, self._loop_break_states[-1]
                entry = joined
        finally:
            self._loop_break_states.pop()
            self._loop_continue_states.pop()

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

    def _default_argument_origins(self, args: ast.arguments) -> _FlowState:
        positional = [*args.posonlyargs, *args.args]
        trailing = positional[len(positional) - len(args.defaults) :]
        bindings = {
            arg.arg: self._expr_origins(default)
            for arg, default in zip(trailing, args.defaults, strict=True)
        }
        bindings.update(
            {
                arg.arg: self._expr_origins(default)
                for arg, default in zip(
                    args.kwonlyargs,
                    args.kw_defaults,
                    strict=True,
                )
                if default is not None
            }
        )
        return bindings

    def _dict_origins(self, expr: ast.Dict) -> _Origin:
        origin = _Origin.NONE
        for key, value in zip(expr.keys, expr.values, strict=True):
            if _is_tal_literal(key):
                origin |= _Origin.TAL_MAPPING
            elif key is None:
                origin |= self._mapping_copy_origins(self._expr_origins(value))
        return origin

    def _mapping_iterable_origins(self, expr: ast.AST) -> _Origin:
        origin = self._expr_origins(expr)
        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Attribute):
            return origin
        if expr.func.attr not in {"items", "keys", "values", "__iter__"}:
            return origin
        return origin | self._expr_origins(expr.func.value)

    def _comprehension_origins(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
    ) -> _Origin:
        first, *remaining = node.generators
        sources = self._mapping_iterable_origins(first.iter)
        first_origins = self._iteration_target_origins(first.target, first.iter)
        local_names = set().union(
            *(_target_names(generator.target) for generator in node.generators)
        )
        self._push_scope(kind="comprehension", local_names=local_names)
        try:
            self._clear_names(local_names)
            self._clear_potential_names(local_names)
            self._bind_iteration_target(
                first.target,
                first.iter,
                origins=first_origins,
            )
            for generator in remaining:
                sources |= self._mapping_iterable_origins(generator.iter)
                self._bind_iteration_target(generator.target, generator.iter)
            values = (
                (node.key, node.value)
                if isinstance(node, ast.DictComp)
                else (node.elt,)
            )
            for value in values:
                sources |= self._expr_origins(value)
            if isinstance(node, ast.DictComp) and _is_tal_literal(node.key):
                sources |= _Origin.TAL_MAPPING
        finally:
            self._pop_scope()
        return self._mapping_copy_origins(sources)

    def _call_origins(self, expr: ast.Call) -> _Origin:
        origin = _Origin.NONE
        for keyword in expr.keywords:
            if keyword.arg == "tal":
                origin |= _Origin.TAL_MAPPING
            elif keyword.arg is None:
                origin |= self._mapping_copy_origins(
                    self._expr_origins(keyword.value)
                )
        if self._expr_origins(expr.func) & _Origin.COPY_FUNCTION:
            if expr.args:
                origin |= self._mapping_copy_origins(
                    self._expr_origins(expr.args[0])
                )
            for keyword in expr.keywords:
                if keyword.arg == "x":
                    origin |= self._mapping_copy_origins(
                        self._expr_origins(keyword.value)
                    )
            return origin
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
            if expr.attr == "attrs":
                return _Origin.ATTRS_ROOT
            receiver = self._expr_origins(expr.value)
            if receiver & _Origin.COPY_MODULE and expr.attr in {"copy", "deepcopy"}:
                return _Origin.COPY_FUNCTION
            if receiver & _Origin.XARRAY_MODULE:
                return _XARRAY_CONSTRUCTORS.get(expr.attr, _Origin.NONE)
            return _Origin.NONE
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
        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            return self._comprehension_origins(expr)
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
            contains_tal = _is_tal_pair(expr) or any(
                _is_tal_pair(item) for item in expr.elts
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
        values = [*call.args]
        values.extend(
            keyword.value for keyword in call.keywords if keyword.arg is None
        )
        return any(self._expr_has_tal_mapping(value) for value in values)

    def _call_transfers_tal_through_attrs(self, call: ast.Call) -> bool:
        constructor = self._expr_origins(call.func)
        positions = [
            position
            for origin, position in _XARRAY_ATTRS_POSITIONS.items()
            if constructor & origin
        ]
        if not positions:
            return False
        if any(
            keyword.arg == "attrs" and self._expr_has_tal_mapping(keyword.value)
            for keyword in call.keywords
        ):
            return True
        return any(
            len(call.args) > position
            and self._expr_has_tal_mapping(call.args[position])
            for position in positions
        )

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

    def _iteration_target_origins(
        self, target: ast.AST, iterable: ast.AST
    ) -> _FlowState:
        origins: _FlowState = {}
        if not isinstance(iterable, (ast.List, ast.Tuple, ast.Set)):
            return origins
        for element in iterable.elts:
            for name, origin in self._assignment_bindings(target, element):
                origins[name] = origins.get(name, _Origin.NONE) | origin
        return origins

    def _bind_iteration_target(
        self,
        target: ast.AST,
        iterable: ast.AST,
        *,
        origins: _FlowState | None = None,
    ) -> None:
        resolved = self._iteration_target_origins(target, iterable)
        if origins is not None:
            resolved = origins
        self._bind_ordinary_target(target)
        for name, origin in resolved.items():
            self._set_name_origin(name, origin)

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
        if (
            self._attrs_call_writes_tal(node)
            or self._tal_root_mutating_call(node)
            or self._call_transfers_tal_through_attrs(node)
        ):
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
        for alias in node.names:
            bound = alias.asname or alias.name.partition(".")[0]
            imports_xarray = alias.name == "xarray" or (
                alias.asname is None and alias.name.startswith("xarray.")
            )
            if imports_xarray:
                origin = _Origin.XARRAY_MODULE
            elif alias.name == "copy":
                origin = _Origin.COPY_MODULE
            else:
                origin = _Origin.NONE
            self._set_name_origin(bound, origin)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        is_xarray = node.module == "xarray" or bool(
            node.module and node.module.startswith("xarray.")
        )
        for alias in node.names:
            if alias.name == "*":
                continue
            bound = alias.asname or alias.name
            if is_xarray:
                origin = _XARRAY_CONSTRUCTORS.get(alias.name, _Origin.NONE)
            elif node.module == "copy" and alias.name in {"copy", "deepcopy"}:
                origin = _Origin.COPY_FUNCTION
            else:
                origin = _Origin.NONE
            self._set_name_origin(bound, origin)

    def _visit_comprehension(self, node: ast.AST, values: list[ast.AST]) -> None:
        first, *remaining = node.generators
        self.visit(first.iter)
        first_origins = self._iteration_target_origins(first.target, first.iter)
        local_names = set().union(
            *(_target_names(generator.target) for generator in node.generators)
        )
        self._push_scope(kind="comprehension", local_names=local_names)
        self._clear_names(local_names)
        self._clear_potential_names(local_names)
        self._bind_iteration_target(
            first.target,
            first.iter,
            origins=first_origins,
        )
        for condition in first.ifs:
            self.visit(condition)
        for generator in remaining:
            self.visit(generator.iter)
            self._bind_iteration_target(generator.target, generator.iter)
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

        def step(entry: _FlowState) -> tuple[_FlowState, _FlowState]:
            self._restore_state(entry)
            self._bind_iteration_target(node.target, node.iter)
            body = self._visit_block_from(self._snapshot_state(), node.body)
            return entry, body

        exit_state, body, break_states = self._solve_loop(base, step)
        normal_exit = self._visit_block_from(exit_state, node.orelse)
        self._restore_state(
            self._join_states([exit_state, body, normal_exit, *break_states])
        )

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

        def step(entry: _FlowState) -> tuple[_FlowState, _FlowState]:
            tested = self._visit_expression_from(entry, node.test)
            return tested, self._visit_block_from(tested, node.body)

        tested, body, break_states = self._solve_loop(base, step)
        normal_exit = self._visit_block_from(tested, node.orelse)
        self._restore_state(
            self._join_states([tested, body, normal_exit, *break_states])
        )

    def visit_Break(self, node: ast.Break) -> None:
        if self._loop_break_states:
            self._loop_break_states[-1].append(self._snapshot_state())

    def visit_Continue(self, node: ast.Continue) -> None:
        if self._loop_continue_states:
            self._loop_continue_states[-1].append(self._snapshot_state())

    def visit_Try(self, node: ast.Try) -> None:
        base = self._snapshot_state()
        body, exception_entries = self._visit_block_with_entry_states(base, node.body)
        success = self._visit_block_from(body, node.orelse)
        handler_base = self._join_states([body, *exception_entries])
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
        default_origins = self._default_argument_origins(node.args)
        inherited = None
        if self._scope_kinds[-1] == "class":
            inherited = self._origin_scopes[-2]
        self._push_scope(inherited=inherited)
        local_names = _function_local_names(node) | {node.name}
        self._clear_names(local_names)
        self._clear_potential_names(local_names)
        for name, origin in default_origins.items():
            self._set_name_origin(name, origin)
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
        default_origins = self._default_argument_origins(node.args)
        inherited = None
        if self._scope_kinds[-1] == "class":
            inherited = self._origin_scopes[-2]
        self._push_scope(inherited=inherited)
        argument_names = _argument_names(node.args)
        self._clear_names(argument_names)
        self._clear_potential_names(argument_names)
        for name, origin in default_origins.items():
            self._set_name_origin(name, origin)
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
