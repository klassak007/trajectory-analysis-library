from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from .broadcast_intent import select_topology_policy

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject
    from .topology import TopologyPolicy


AlignmentKey = Literal["sequence", "param", "auto"]
JoinPolicy = Literal["exact", "inner", "outer", "left", "right"]
CorePolicy = Literal["strict", "numpy_named"]

_INTENT_ATTR = "_alignment_intent"
_JOIN_POLICIES = frozenset({"exact", "inner", "outer", "left", "right"})
_KEY_POLICIES = frozenset({"sequence", "param", "auto"})
_CORE_POLICIES = frozenset({"strict", "numpy_named"})


@dataclass(frozen=True)
class AlignmentIntent:
    on: AlignmentKey = "sequence"
    sequence_join: JoinPolicy | None = "exact"
    batch_join: JoinPolicy = "exact"
    core_policy: CorePolicy = "strict"


@dataclass(frozen=True)
class OperationTopologyIntent:
    policy: "TopologyPolicy"
    alignment: AlignmentIntent | None


@dataclass(frozen=True)
class OperationIntentSupport:
    semantic_default: bool = False
    alignment_intent_supported: bool = False


def _valid_join(value: str | None) -> bool:
    if value is None:
        return True
    return value in _JOIN_POLICIES


def resolve_alignment_intent(
    *,
    on: str,
    sequence_join: str | None,
    batch_join: str,
    core_policy: str,
    owner: str,
) -> AlignmentIntent:
    if on not in _KEY_POLICIES:
        raise ValueError(
            f"{owner}: unsupported alignment key {on!r}; "
            "expected 'sequence', 'param', or 'auto'."
        )
    if not _valid_join(sequence_join):
        raise ValueError(
            f"{owner}: unsupported sequence_join={sequence_join!r}; "
            f"allowed={tuple(sorted(_JOIN_POLICIES))!r} or None."
        )
    if batch_join not in _JOIN_POLICIES:
        raise ValueError(
            f"{owner}: unsupported batch_join={batch_join!r}; "
            f"allowed={tuple(sorted(_JOIN_POLICIES))!r}."
        )
    if core_policy not in _CORE_POLICIES:
        raise ValueError(
            f"{owner}: unsupported core_policy={core_policy!r}; "
            "expected 'strict' or 'numpy_named'."
        )
    if on == "sequence" and sequence_join is None:
        raise ValueError(
            f"{owner}: sequence_join=None is only valid when on='param' or on='auto'."
        )
    return AlignmentIntent(
        on=on,  # type: ignore[arg-type]
        sequence_join=sequence_join,  # type: ignore[arg-type]
        batch_join=batch_join,  # type: ignore[arg-type]
        core_policy=core_policy,  # type: ignore[arg-type]
    )


def set_alignment_intent(ao: "AnalysisObject", intent: AlignmentIntent | None) -> None:
    setattr(ao, _INTENT_ATTR, intent)


def clear_alignment_intent(ao: "AnalysisObject") -> None:
    setattr(ao, _INTENT_ATTR, None)


def read_alignment_intent(
    value: object,
    *,
    owner: str,
    label: str = "operand",
) -> AlignmentIntent | None:
    from ..analysis_object import AnalysisObject

    if not isinstance(value, AnalysisObject):
        return None
    raw = getattr(value, _INTENT_ATTR, None)
    if raw is None:
        return None
    if isinstance(raw, AlignmentIntent):
        return raw
    raise TypeError(
        f"{owner}: {label} has invalid alignment intent carrier type {type(raw).__name__}."
    )


def any_alignment_intent(values: Iterable[object], *, owner: str) -> bool:
    return any(read_alignment_intent(value, owner=owner) is not None for value in values)


def merge_alignment_intent(
    values: Iterable[object],
    *,
    owner: str,
) -> AlignmentIntent | None:
    intents = [
        intent
        for value in values
        if (intent := read_alignment_intent(value, owner=owner)) is not None
    ]
    if not intents:
        return None
    merged = intents[0]
    for idx, intent in enumerate(intents[1:], start=1):
        if intent == merged:
            continue
        raise ValueError(
            f"{owner}: conflicting alignment intents are not allowed; "
            f"intent[0]={merged!r}, intent[{idx}]={intent!r}."
        )
    return merged


def apply_alignment_intent_to_policy(
    policy: "TopologyPolicy",
    *,
    alignment: AlignmentIntent,
) -> "TopologyPolicy":
    return replace(
        policy,
        alignment_on=alignment.on,
        sequence_join=alignment.sequence_join,
        batch_join=alignment.batch_join,
        core_policy=alignment.core_policy,
    )


def select_topology_policy_with_intents(
    values: Iterable[object],
    *,
    owner: str,
    operation_family: str,
    support: OperationIntentSupport,
    strict_policy: "TopologyPolicy",
    semantic_policy: "TopologyPolicy",
) -> OperationTopologyIntent:
    policy = select_topology_policy(
        values,
        owner=owner,
        strict_policy=strict_policy,
        semantic_policy=semantic_policy,
        semantic_default=support.semantic_default,
    )
    alignment = merge_alignment_intent(values, owner=owner)
    if alignment is None:
        return OperationTopologyIntent(policy=policy, alignment=None)
    if not support.alignment_intent_supported:
        raise ValueError(
            f"{owner}: alignment intent '.a(...)' is not supported for operation family "
            f"{operation_family!r}."
        )
    return OperationTopologyIntent(
        policy=apply_alignment_intent_to_policy(policy, alignment=alignment),
        alignment=alignment,
    )


__all__ = [
    "AlignmentIntent",
    "OperationIntentSupport",
    "OperationTopologyIntent",
    "any_alignment_intent",
    "apply_alignment_intent_to_policy",
    "clear_alignment_intent",
    "merge_alignment_intent",
    "read_alignment_intent",
    "resolve_alignment_intent",
    "select_topology_policy_with_intents",
    "set_alignment_intent",
]
