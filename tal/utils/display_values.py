"""Bounded formatting of stored metadata, without custom representation calls."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DisplayBudget:
    depth: int
    items: int
    text: int


SUMMARY_BUDGET = DisplayBudget(3, 32, 2048)
DETAIL_BUDGET = DisplayBudget(8, 256, 16384)
_CONTAINERS = (dict, list, tuple, set, frozenset)
_TEXT_OMISSION = "… <text omitted>"


def type_label(value: object) -> str:
    """Return bounded type metadata, never the value's representation."""
    cls = type(value)
    module = cls.__module__
    name = cls.__qualname__
    if type(module) is not str or type(name) is not str:
        return "unknown type"
    return f"{module[:128]}.{name[:128]}"


@dataclass
class _Writer:
    budget: DisplayBudget
    quote_strings: bool
    parts: list[str] = field(default_factory=list)
    active: set[int] = field(default_factory=set)
    characters: int = 0
    items: int = 0
    stopped: bool = False

    def write(self, text: str) -> None:
        if self.stopped:
            return
        remaining = self.budget.text - self.characters
        self.parts.append(text[:remaining])
        self.characters += min(len(text), remaining)
        if len(text) > remaining:
            self.parts.append(_TEXT_OMISSION)
            self.stopped = True


def _scalar(value: object, writer: _Writer) -> str:
    kind = type(value)
    remaining = max(0, writer.budget.text - writer.characters)
    if kind is str:
        short = value[:remaining]
        rendered = repr(short) if writer.quote_strings else short
        return rendered + (_TEXT_OMISSION if len(value) > remaining else "")
    if kind is bytes:
        return repr(value[:remaining]) + (_TEXT_OMISSION if len(value) > remaining else "")
    if kind is int:
        if value.bit_length() > remaining * 4:
            return "<int: text omitted>"
        return hex(value) if value.bit_length() > 12000 else str(value)
    if value is None:
        return "None" if writer.quote_strings else "none"
    if kind in (bool, float, complex):
        return repr(value)
    return f"<{type_label(value)}: value omitted>"


def _container_tokens(value: object) -> tuple[str, str]:
    kind = type(value)
    if kind is dict:
        return "{", "}"
    if kind is list:
        return "[", "]"
    if kind is tuple:
        return "(", ")"
    if kind is frozenset:
        return "frozenset({", "})"
    return ("{", "}") if value else ("set(", ")")


def _entry(value: object, writer: _Writer, depth: int, *, mapping: bool) -> None:
    if not mapping:
        _visit(value, writer, depth)
        return
    key, item = value
    _visit(key, writer, depth)
    writer.write(": ")
    _visit(item, writer, depth)


def _container(value: object, writer: _Writer, depth: int) -> None:
    if id(value) in writer.active:
        writer.write("<recursive container: omitted>")
        return
    if depth >= writer.budget.depth:
        writer.write("<depth omitted>")
        return
    opening, closing = _container_tokens(value)
    writer.write(opening)
    writer.active.add(id(value))
    mapping = type(value) is dict
    entries = value.items() if mapping else value
    for index, item in enumerate(entries):
        if writer.stopped:
            break
        if index:
            writer.write(", ")
        if writer.items >= writer.budget.items:
            writer.write("<items omitted>")
            break
        _entry(item, writer, depth + 1, mapping=mapping)
    writer.active.remove(id(value))
    if type(value) is tuple and len(value) == 1 and writer.quote_strings:
        writer.write(",")
    writer.write(closing)


def _visit(value: object, writer: _Writer, depth: int) -> None:
    if writer.stopped:
        return
    if writer.items >= writer.budget.items:
        writer.write("<items omitted>")
        return
    writer.items += 1
    if type(value) in _CONTAINERS:
        _container(value, writer, depth)
    else:
        writer.write(_scalar(value, writer))


def format_stored_usage(value: object, *, budget: DisplayBudget, quote_strings: bool = True) -> tuple[str, int]:
    """Format exact built-ins while bounding traversal and string allocation."""
    writer = _Writer(budget, quote_strings)
    _visit(value, writer, 0)
    return "".join(writer.parts), writer.items


def format_stored(value: object, *, budget: DisplayBudget, quote_strings: bool = True) -> str:
    return format_stored_usage(value, budget=budget, quote_strings=quote_strings)[0]
