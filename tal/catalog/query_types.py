from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

QueryNamespace = Literal["attr", "coord", "batch"]
ComparisonOp = Literal["==", "!=", "<", "<=", ">", ">="]
MembershipOp = Literal["in", "not in"]
BooleanOp = Literal["and", "or"]

_NAMESPACES: set[str] = {"attr", "coord", "batch"}


@dataclass(frozen=True)
class FieldRef:
    namespace: QueryNamespace | None
    name: str


@dataclass(frozen=True)
class ComparisonPredicate:
    field: FieldRef
    op: ComparisonOp
    value: object


@dataclass(frozen=True)
class MembershipPredicate:
    field: FieldRef
    op: MembershipOp
    values: tuple[object, ...]


@dataclass(frozen=True)
class BooleanPredicate:
    op: BooleanOp
    args: tuple["Predicate", ...]


@dataclass(frozen=True)
class NotPredicate:
    arg: "Predicate"


Predicate = ComparisonPredicate | MembershipPredicate | BooleanPredicate | NotPredicate


def parse_field_ref(value: object, *, owner: str) -> FieldRef:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner}: field must be a non-empty string; got {value!r}.")
    if "." not in value:
        return FieldRef(namespace=None, name=value)
    prefix, suffix = value.split(".", 1)
    if prefix in _NAMESPACES:
        if not suffix:
            raise ValueError(
                f"{owner}: qualified field {value!r} must include a non-empty name after namespace."
            )
        return FieldRef(namespace=prefix, name=suffix)  # type: ignore[arg-type]
    return FieldRef(namespace=None, name=value)


def require_predicate_mapping(value: object, *, owner: str) -> Mapping[str, object]:
    if isinstance(value, str):
        raise TypeError(
            f"{owner}: string query expressions are not supported; use structured predicates."
        )
    if callable(value):
        raise TypeError(
            f"{owner}: callable predicates are not supported; use structured predicates."
        )
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{owner}: where must be a mapping predicate payload; got {type(value).__name__}."
        )
    return value


def require_membership_values(value: object, *, owner: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{owner}: membership predicates require a finite sequence value; got {value!r}.")
    out = tuple(value)
    if not out:
        raise ValueError(f"{owner}: membership predicates require at least one candidate value.")
    return out


__all__ = [
    "BooleanPredicate",
    "BooleanOp",
    "ComparisonOp",
    "ComparisonPredicate",
    "FieldRef",
    "MembershipOp",
    "MembershipPredicate",
    "NotPredicate",
    "Predicate",
    "QueryNamespace",
    "parse_field_ref",
    "require_membership_values",
    "require_predicate_mapping",
]
