from __future__ import annotations

from collections.abc import Mapping, Sequence
import operator

import numpy as np

from .query_types import (
    BooleanPredicate,
    ComparisonPredicate,
    MembershipPredicate,
    NotPredicate,
    Predicate,
)
from .selection import select_by_labels
from .types import CatalogState

_ORDER_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def apply_query_predicate(
    state: CatalogState,
    *,
    predicate: Predicate | None,
    labels: Sequence[object],
    columns: Mapping[tuple[str, str], np.ndarray],
    owner: str,
) -> CatalogState:
    mask = evaluate_query_mask(predicate, labels=labels, columns=columns, owner=owner)
    selected = tuple(label for label, keep in zip(labels, mask, strict=True) if keep)
    return select_by_labels(state, selected, owner=owner)


def evaluate_query_mask(
    predicate: Predicate | None,
    *,
    labels: Sequence[object],
    columns: Mapping[tuple[str, str], np.ndarray],
    owner: str,
) -> np.ndarray:
    if predicate is None:
        return np.ones((len(labels),), dtype=bool)
    if isinstance(predicate, ComparisonPredicate):
        values = _field_values(columns, namespace=predicate.field.namespace, name=predicate.field.name, owner=owner)
        return _compare(values, op=predicate.op, rhs=predicate.value, owner=owner)
    if isinstance(predicate, MembershipPredicate):
        values = _field_values(columns, namespace=predicate.field.namespace, name=predicate.field.name, owner=owner)
        return _membership(values, candidates=predicate.values, negate=(predicate.op == "not in"))
    if isinstance(predicate, NotPredicate):
        return np.logical_not(evaluate_query_mask(predicate.arg, labels=labels, columns=columns, owner=owner))
    return _compose_boolean(predicate, labels=labels, columns=columns, owner=owner)


def _field_values(
    columns: Mapping[tuple[str, str], np.ndarray],
    *,
    namespace: str | None,
    name: str,
    owner: str,
) -> np.ndarray:
    if namespace is None:
        raise TypeError(f"{owner}: query field {name!r} missing namespace in evaluation path.")
    key = (namespace, name)
    if key not in columns:
        raise ValueError(f"{owner}: metadata field {namespace}.{name!r} was not projected for query evaluation.")
    return columns[key]


def _compare(values: np.ndarray, *, op: str, rhs: object, owner: str) -> np.ndarray:
    if op == "==":
        return _equal(values, rhs=rhs)
    if op == "!=":
        return np.logical_not(_equal(values, rhs=rhs))
    fn = _ORDER_OPS[op]
    out = np.zeros(values.shape, dtype=bool)
    for idx, value in enumerate(values.tolist()):
        try:
            out[idx] = bool(fn(value, rhs))
        except Exception as exc:
            raise ValueError(
                f"{owner}: ordering predicate {op!r} failed for value {value!r} and rhs {rhs!r}."
            ) from exc
    return out


def _equal(values: np.ndarray, *, rhs: object) -> np.ndarray:
    out = np.zeros(values.shape, dtype=bool)
    for idx, value in enumerate(values.tolist()):
        if _both_nan(value, rhs):
            out[idx] = True
            continue
        out[idx] = bool(value == rhs)
    return out


def _both_nan(left: object, right: object) -> bool:
    try:
        return bool(np.isnan(left)) and bool(np.isnan(right))  # type: ignore[arg-type]
    except Exception:
        return False


def _membership(values: np.ndarray, *, candidates: Sequence[object], negate: bool) -> np.ndarray:
    normalized = tuple(candidates)
    out = np.asarray([_in_candidates(value, candidates=normalized) for value in values.tolist()], dtype=bool)
    return np.logical_not(out) if negate else out


def _in_candidates(value: object, *, candidates: Sequence[object]) -> bool:
    for candidate in candidates:
        if _both_nan(value, candidate) or value == candidate:
            return True
    return False


def _compose_boolean(
    predicate: BooleanPredicate,
    *,
    labels: Sequence[object],
    columns: Mapping[tuple[str, str], np.ndarray],
    owner: str,
) -> np.ndarray:
    masks = [evaluate_query_mask(arg, labels=labels, columns=columns, owner=owner) for arg in predicate.args]
    if predicate.op == "and":
        out = masks[0]
        for mask in masks[1:]:
            out = np.logical_and(out, mask)
        return out
    out = masks[0]
    for mask in masks[1:]:
        out = np.logical_or(out, mask)
    return out


__all__ = ["apply_query_predicate", "evaluate_query_mask"]
