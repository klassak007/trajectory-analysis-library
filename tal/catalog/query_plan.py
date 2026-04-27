from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .options import CatalogQueryOptions
from .query_types import (
    BooleanPredicate,
    ComparisonPredicate,
    FieldRef,
    MembershipPredicate,
    NotPredicate,
    Predicate,
    parse_field_ref,
    require_membership_values,
    require_predicate_mapping,
)

_COMPARISON_OPS = {"==", "!=", "<", "<=", ">", ">="}
_MEMBERSHIP_OPS = {"in", "not in"}
_BOOLEAN_OPS = {"and", "or"}


@dataclass(frozen=True)
class QueryFieldIndex:
    attr: frozenset[str]
    coord: frozenset[str]
    batch: frozenset[str]


@dataclass(frozen=True)
class CatalogQueryPlan:
    predicate: Predicate | None
    projections: tuple[FieldRef, ...]


def normalize_catalog_query_plan(
    *,
    where: object,
    filters: Mapping[str, object],
    field_index: QueryFieldIndex,
    options: CatalogQueryOptions,
    owner: str,
) -> CatalogQueryPlan:
    parsed_where = _parse_where(where, owner=owner)
    kwargs_pred = _build_kwargs_predicate(filters, owner=owner)
    combined = _combine_predicates(parsed_where, kwargs_pred)
    resolved = _resolve_predicate_fields(
        combined,
        field_index=field_index,
        unknown_field_policy=options.unknown_field_policy,
        owner=owner,
    )
    projections = tuple(_collect_projections(resolved))
    return CatalogQueryPlan(predicate=resolved, projections=projections)


def _parse_where(where: object, *, owner: str) -> Predicate | None:
    if where is None:
        return None
    payload = require_predicate_mapping(where, owner=owner)
    return _parse_predicate(payload, owner=owner)


def _build_kwargs_predicate(filters: Mapping[str, object], *, owner: str) -> Predicate | None:
    if not filters:
        return None
    args: list[Predicate] = []
    for key, value in filters.items():
        field = parse_field_ref(key, owner=owner)
        args.append(ComparisonPredicate(field=field, op="==", value=value))
    if len(args) == 1:
        return args[0]
    return BooleanPredicate(op="and", args=tuple(args))


def _combine_predicates(left: Predicate | None, right: Predicate | None) -> Predicate | None:
    if left is None:
        return right
    if right is None:
        return left
    return BooleanPredicate(op="and", args=(left, right))


def _parse_predicate(payload: Mapping[str, object], *, owner: str) -> Predicate:
    op = payload.get("op")
    if not isinstance(op, str):
        raise ValueError(f"{owner}: predicate mapping requires string key 'op'.")
    if op in _COMPARISON_OPS:
        return _parse_comparison_predicate(payload, op=op, owner=owner)
    if op in _MEMBERSHIP_OPS:
        return _parse_membership_predicate(payload, op=op, owner=owner)
    if op in _BOOLEAN_OPS:
        return _parse_boolean_predicate(payload, op=op, owner=owner)
    if op == "not":
        return _parse_not_predicate(payload, owner=owner)
    raise ValueError(
        f"{owner}: unsupported predicate operator {op!r}; allowed operators are "
        "==, !=, <, <=, >, >=, in, not in, and, or, not."
    )


def _parse_comparison_predicate(
    payload: Mapping[str, object],
    *,
    op: str,
    owner: str,
) -> ComparisonPredicate:
    if "field" not in payload:
        raise ValueError(f"{owner}: comparison predicate {op!r} requires key 'field'.")
    if "value" not in payload:
        raise ValueError(f"{owner}: comparison predicate {op!r} requires key 'value'.")
    field = parse_field_ref(payload["field"], owner=owner)
    return ComparisonPredicate(field=field, op=op, value=payload["value"])  # type: ignore[arg-type]


def _parse_membership_predicate(
    payload: Mapping[str, object],
    *,
    op: str,
    owner: str,
) -> MembershipPredicate:
    if "field" not in payload:
        raise ValueError(f"{owner}: membership predicate {op!r} requires key 'field'.")
    if "value" not in payload:
        raise ValueError(f"{owner}: membership predicate {op!r} requires key 'value'.")
    field = parse_field_ref(payload["field"], owner=owner)
    values = require_membership_values(payload["value"], owner=owner)
    return MembershipPredicate(field=field, op=op, values=values)  # type: ignore[arg-type]


def _parse_boolean_predicate(
    payload: Mapping[str, object],
    *,
    op: str,
    owner: str,
) -> BooleanPredicate:
    args_raw = payload.get("args")
    if not isinstance(args_raw, list) or not args_raw:
        raise ValueError(f"{owner}: boolean predicate {op!r} requires non-empty list key 'args'.")
    args = tuple(_parse_predicate(require_predicate_mapping(arg, owner=owner), owner=owner) for arg in args_raw)
    return BooleanPredicate(op=op, args=args)  # type: ignore[arg-type]


def _parse_not_predicate(payload: Mapping[str, object], *, owner: str) -> NotPredicate:
    arg_raw = payload.get("arg")
    if arg_raw is None:
        raise ValueError(f"{owner}: boolean predicate 'not' requires key 'arg'.")
    arg = _parse_predicate(require_predicate_mapping(arg_raw, owner=owner), owner=owner)
    return NotPredicate(arg=arg)


def _resolve_predicate_fields(
    predicate: Predicate | None,
    *,
    field_index: QueryFieldIndex,
    unknown_field_policy: str,
    owner: str,
) -> Predicate | None:
    if predicate is None:
        return None
    if isinstance(predicate, (ComparisonPredicate, MembershipPredicate)):
        return _resolve_leaf_predicate(
            predicate,
            field_index=field_index,
            unknown_field_policy=unknown_field_policy,
            owner=owner,
        )
    if isinstance(predicate, NotPredicate):
        resolved = _resolve_predicate_fields(
            predicate.arg,
            field_index=field_index,
            unknown_field_policy=unknown_field_policy,
            owner=owner,
        )
        return None if resolved is None else NotPredicate(arg=resolved)
    args = [
        _resolve_predicate_fields(
            arg,
            field_index=field_index,
            unknown_field_policy=unknown_field_policy,
            owner=owner,
        )
        for arg in predicate.args
    ]
    kept = tuple(arg for arg in args if arg is not None)
    if not kept:
        return None
    if len(kept) == 1:
        return kept[0]
    return BooleanPredicate(op=predicate.op, args=kept)


def _resolve_leaf_predicate(
    predicate: ComparisonPredicate | MembershipPredicate,
    *,
    field_index: QueryFieldIndex,
    unknown_field_policy: str,
    owner: str,
) -> Predicate | None:
    field = _resolve_field_ref(
        predicate.field,
        field_index=field_index,
        unknown_field_policy=unknown_field_policy,
        owner=owner,
    )
    if field is None:
        return None
    if isinstance(predicate, ComparisonPredicate):
        return ComparisonPredicate(field=field, op=predicate.op, value=predicate.value)
    return MembershipPredicate(field=field, op=predicate.op, values=predicate.values)


def _resolve_field_ref(
    field: FieldRef,
    *,
    field_index: QueryFieldIndex,
    unknown_field_policy: str,
    owner: str,
) -> FieldRef | None:
    if field.namespace is not None:
        names = _namespace_names(field_index, namespace=field.namespace)
        if field.name in names:
            return field
        if unknown_field_policy == "ignore":
            return None
        raise ValueError(f"{owner}: unknown metadata field {field.namespace}.{field.name!r}.")
    matches = _unqualified_matches(field.name, field_index=field_index)
    if len(matches) > 1:
        match_list = ", ".join(f"{ns}.{field.name}" for ns in matches)
        raise ValueError(
            f"{owner}: ambiguous unqualified metadata field {field.name!r}; matches {match_list}."
        )
    if not matches:
        if unknown_field_policy == "ignore":
            return None
        raise ValueError(f"{owner}: unknown metadata field {field.name!r}.")
    return FieldRef(namespace=matches[0], name=field.name)


def _namespace_names(field_index: QueryFieldIndex, *, namespace: str) -> frozenset[str]:
    if namespace == "attr":
        return field_index.attr
    if namespace == "coord":
        return field_index.coord
    return field_index.batch


def _unqualified_matches(name: str, *, field_index: QueryFieldIndex) -> tuple[str, ...]:
    matches: list[str] = []
    if name in field_index.attr:
        matches.append("attr")
    if name in field_index.coord:
        matches.append("coord")
    if name in field_index.batch:
        matches.append("batch")
    return tuple(matches)


def _collect_projections(predicate: Predicate | None) -> set[FieldRef]:
    out: set[FieldRef] = set()
    _walk_projection(predicate, out=out)
    return out


def _walk_projection(predicate: Predicate | None, *, out: set[FieldRef]) -> None:
    if predicate is None:
        return
    if isinstance(predicate, (ComparisonPredicate, MembershipPredicate)):
        out.add(predicate.field)
        return
    if isinstance(predicate, NotPredicate):
        _walk_projection(predicate.arg, out=out)
        return
    for arg in predicate.args:
        _walk_projection(arg, out=out)


__all__ = [
    "CatalogQueryPlan",
    "QueryFieldIndex",
    "normalize_catalog_query_plan",
]
