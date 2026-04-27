from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject
    from .topology import TopologyPolicy


_INTENT_ATTR = "_broadcast_intent"


@dataclass(frozen=True)
class BroadcastIntent:
    mode: Literal["semantic_broadcast"]


SEMANTIC_BROADCAST_INTENT = BroadcastIntent(mode="semantic_broadcast")


def resolve_broadcast_intent(
    *,
    mode: str,
    owner: str,
) -> BroadcastIntent:
    if mode == "semantic_broadcast":
        return SEMANTIC_BROADCAST_INTENT
    raise ValueError(
        f"{owner}: unsupported broadcast mode {mode!r}; expected 'semantic_broadcast'."
    )


def set_broadcast_intent(ao: "AnalysisObject", intent: BroadcastIntent | None) -> None:
    setattr(ao, _INTENT_ATTR, intent)


def clear_broadcast_intent(ao: "AnalysisObject") -> None:
    setattr(ao, _INTENT_ATTR, None)


def read_broadcast_intent(
    value: object,
    *,
    owner: str,
    label: str = "operand",
) -> BroadcastIntent | None:
    from ..analysis_object import AnalysisObject

    if not isinstance(value, AnalysisObject):
        return None
    raw = getattr(value, _INTENT_ATTR, None)
    if raw is None:
        return None
    if isinstance(raw, BroadcastIntent):
        return raw
    raise TypeError(
        f"{owner}: {label} has invalid broadcast intent carrier type {type(raw).__name__}."
    )


def any_broadcast_intent(
    values: Iterable[object],
    *,
    owner: str,
) -> bool:
    return any(read_broadcast_intent(value, owner=owner) is not None for value in values)


def select_topology_policy(
    values: Iterable[object],
    *,
    owner: str,
    strict_policy: "TopologyPolicy",
    semantic_policy: "TopologyPolicy",
    semantic_default: bool = False,
) -> "TopologyPolicy":
    if any_broadcast_intent(values, owner=owner):
        return semantic_policy
    if semantic_default:
        return semantic_policy
    return strict_policy


__all__ = [
    "BroadcastIntent",
    "SEMANTIC_BROADCAST_INTENT",
    "any_broadcast_intent",
    "clear_broadcast_intent",
    "read_broadcast_intent",
    "resolve_broadcast_intent",
    "select_topology_policy",
    "set_broadcast_intent",
]
