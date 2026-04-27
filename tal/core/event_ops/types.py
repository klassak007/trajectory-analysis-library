from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class VarOperand:
    """Reference a context data variable by name.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    name: str


@dataclass(frozen=True)
class CoordOperand:
    """Reference a context coordinate by name.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    name: str


ConditionCompareOp = Literal["lt", "le", "gt", "ge", "eq", "ne"]


@dataclass(frozen=True)
class CompareNode:
    """Binary comparison leaf node.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    left: object
    op: ConditionCompareOp
    right: object


@dataclass(frozen=True)
class NotNode:
    """Unary negation node.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    child: "ConditionNode"


@dataclass(frozen=True)
class AndNode:
    """Boolean conjunction node.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    left: "ConditionNode"
    right: "ConditionNode"


@dataclass(frozen=True)
class OrNode:
    """Boolean disjunction node.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    left: "ConditionNode"
    right: "ConditionNode"


ConditionNode = CompareNode | NotNode | AndNode | OrNode


@dataclass(frozen=True)
class ConditionEvalOptions:
    """Condition evaluation options for context-clock masking.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    on: object | None = None
    coord_name: str = "time"
    sample_dim: str | None = None
    ao_interp: Literal["linear", "nearest"] = "linear"
    eq_atol: float = 1e-9
    eq_rtol: float = 1e-12
    validity_mode: Literal["auto", "prefix_only", "finite_gather"] = "auto"


@dataclass(frozen=True)
class EventExtractOptions:
    """Event extraction options for ``EventsAccessor.events``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    eval: ConditionEvalOptions = field(default_factory=ConditionEvalOptions)
    include_initial: bool = False
    truth_eval: Literal["exact", "before", "after", "nearest"] = "exact"
    max_events: int | None = None
    dedupe_atol: float = 1e-12


@dataclass(frozen=True)
class IntervalExtractOptions:
    """Interval extraction options for ``EventsAccessor.intervals``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    eval: ConditionEvalOptions = field(default_factory=ConditionEvalOptions)
    include_initial: bool = False
    truth_eval: Literal["exact", "before", "after", "nearest"] = "exact"
    max_segments: int | None = None


@dataclass(frozen=True)
class AtBoundariesOptions:
    """Boundary evaluation options for ``EventsAccessor.at_boundaries``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    eval: ConditionEvalOptions = field(default_factory=ConditionEvalOptions)
    edges: Literal["all", "enter", "exit"] = "all"
    mode: Literal["all", "first", "first_n", "last"] = "all"
    max_events: int | None = None
    on_empty: Literal["empty", "error"] = "empty"


@dataclass(frozen=True)
class WhenOptions:
    """Condition-driven AO selection options for ``EventsAccessor.when``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    eval: ConditionEvalOptions = field(default_factory=ConditionEvalOptions)
    layout: Literal["mask", "segments", "stream"] = "mask"
    inside: bool = True
    on_empty: Literal["empty", "error"] = "empty"
    max_segments: int | None = None


@dataclass(frozen=True)
class AroundOptions:
    """Event-locked window extraction options for ``EventsAccessor.around``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    eval: ConditionEvalOptions = field(default_factory=ConditionEvalOptions)
    edge: Literal["enter", "exit", "all"] = "enter"
    pre: float = 0.5
    post: float = 0.5
    dt: float | None = None
    grid: xr.DataArray | np.ndarray | None = None
    layout: Literal["segments", "stacked"] = "segments"


@dataclass(frozen=True)
class EvalMask:
    """Three-valued mask payload.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    truth: xr.DataArray
    determinate: xr.DataArray


def _coerce_op(op: str) -> ConditionCompareOp:
    mapping = {
        "<": "lt",
        "<=": "le",
        ">": "gt",
        ">=": "ge",
        "==": "eq",
        "!=": "ne",
        "lt": "lt",
        "le": "le",
        "gt": "gt",
        "ge": "ge",
        "eq": "eq",
        "ne": "ne",
    }
    if op not in mapping:
        raise ValueError(
            "Condition.compare: op must be one of "
            "['lt','le','gt','ge','eq','ne','<','<=','>','>=','==','!=']."
        )
    return mapping[op]


def _var_operand(name: str, *, owner: str) -> VarOperand:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{owner}: name must be a non-empty string.")
    return VarOperand(name)


def _coord_operand(name: str, *, owner: str) -> CoordOperand:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{owner}: name must be a non-empty string.")
    return CoordOperand(name)


@dataclass(frozen=True)
class Condition:
    """Immutable condition AST wrapper with boolean composition.

    Notes
    -----
    Conditions are symbolic until evaluated by an event accessor. They are not
    truth-testable in Python because the result depends on AO data.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.event_ops import Condition
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"error": ("sample", [0.1, 1.2])}, coords={"sample": [0, 1]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> condition = Condition.compare(Condition.var("error"), "gt", 1.0)
    >>> ao.events.mask(condition).values.tolist()
    [False, True]
    """

    node: ConditionNode

    @staticmethod
    def var(name: str) -> VarOperand:
        return _var_operand(name, owner="Condition.var")

    @staticmethod
    def coord(name: str) -> CoordOperand:
        return _coord_operand(name, owner="Condition.coord")

    @staticmethod
    def compare(left: object, op: str, right: object) -> "Condition":
        return Condition(CompareNode(left=left, op=_coerce_op(op), right=right))

    def __bool__(self) -> bool:
        raise TypeError(
            "Condition does not support truth-value testing; evaluate with ao.events.mask(condition)."
        )

    def __and__(self, other: object) -> "Condition":
        if not isinstance(other, Condition):
            raise TypeError("Condition.__and__: right operand must be Condition.")
        return Condition(AndNode(left=self.node, right=other.node))

    def __or__(self, other: object) -> "Condition":
        if not isinstance(other, Condition):
            raise TypeError("Condition.__or__: right operand must be Condition.")
        return Condition(OrNode(left=self.node, right=other.node))

    def __xor__(self, other: object) -> "Condition":
        if not isinstance(other, Condition):
            raise TypeError("Condition.__xor__: right operand must be Condition.")
        # XOR is represented compositionally to keep AST node ownership minimal.
        return (self | other) & ~(self & other)

    def __invert__(self) -> "Condition":
        return Condition(NotNode(child=self.node))


__all__ = [
    "AndNode",
    "AroundOptions",
    "CompareNode",
    "Condition",
    "ConditionCompareOp",
    "ConditionEvalOptions",
    "ConditionNode",
    "CoordOperand",
    "WhenOptions",
    "EventExtractOptions",
    "EvalMask",
    "IntervalExtractOptions",
    "AtBoundariesOptions",
    "NotNode",
    "OrNode",
    "VarOperand",
]
