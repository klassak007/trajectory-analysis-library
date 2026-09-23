"""Presentation-only readers of local stored schema shapes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

from .display_values import (
    SUMMARY_BUDGET,
    DisplayBudget,
    format_stored,
    format_stored_usage,
)

MISSING = object()


@dataclass(frozen=True)
class DisplayRow:
    label: str
    value: object


def stored_path(schema: object, *path: str) -> object:
    """Read a stored path without defaults, normalization or semantic checks."""
    value = schema
    for key in path:
        if value is MISSING:
            return MISSING
        if type(value) is not dict:
            raise ValueError("unreadable stored mapping")
        value = value.get(key, MISSING)
    return value


def stored_name(value: object, *, nullable: bool = False) -> object:
    if value is None and nullable:
        return None
    if type(value) is not str:
        raise ValueError("unreadable stored name")
    return value


def stored_names(value: object) -> object:
    if type(value) not in (list, tuple):
        raise ValueError("unreadable stored names")
    names = tuple(islice(value, SUMMARY_BUDGET.items))
    for name in names:
        stored_name(name)
    return names + (("<items omitted>",) if len(value) > len(names) else ())


def optional_row(label: str, schema: object, path: tuple[str, ...], reader=stored_name) -> DisplayRow | None:
    try:
        value = stored_path(schema, *path)
        return None if value is MISSING else DisplayRow(label, reader(value))
    except (TypeError, ValueError):
        return DisplayRow(label, "unavailable")


def _roles(schema: object) -> list[DisplayRow]:
    try:
        roles = stored_path(schema, "core", "roles")
    except (TypeError, ValueError):
        return [DisplayRow("Roles", "unavailable")]
    if roles is MISSING:
        return [DisplayRow("Roles", "undeclared")]
    return [
        optional_row("Sequence", roles, ("sequence_dim",), lambda value: stored_name(value, nullable=True))
        or DisplayRow("Sequence", None),
        optional_row("Batch", roles, ("batch_dims",), stored_names) or DisplayRow("Batch", "unavailable"),
        optional_row("Core", roles, ("core_dims",), stored_names) or DisplayRow("Core", "unavailable"),
    ]


def _validity(value: object) -> str:
    name = stored_name(stored_path(value, "sequence_size_coord"))
    layout = stored_name(stored_path(value, "layout"))
    return format_stored(name, budget=SUMMARY_BUDGET, quote_strings=False) + " (" + format_stored(
        layout, budget=SUMMARY_BUDGET, quote_strings=False,
    ) + ")"


def _mapping_names(value: object) -> list[str]:
    if type(value) is not dict:
        raise ValueError("unreadable stored registry")
    names = [stored_name(key) for key in islice(value, SUMMARY_BUDGET.items)]
    if len(value) > SUMMARY_BUDGET.items:
        names.append("<items omitted>")
    return names


def core_rows(schema: object) -> tuple[DisplayRow, ...]:
    rows = _roles(schema)
    rows.extend((
        optional_row("Parameter", schema, ("core", "param_coord", "name")),
        optional_row("Validity", schema, ("core", "validity"), _validity),
        optional_row("Components", schema, ("ext", "components", "registry"), _mapping_names),
        optional_row("Extensions", schema, ("ext",), _mapping_names),
    ))
    return tuple(row for row in rows if row is not None)


def _declared_role_names(schema: object, path: tuple[str, ...], *, multiple=False) -> tuple[str, ...]:
    try:
        value = stored_path(schema, *path)
        if multiple and type(value) not in (tuple, list):
            return ()
        values = islice(value, SUMMARY_BUDGET.items) if multiple else (value,)
        return tuple(stored_name(name) for name in values)
    except (TypeError, ValueError):
        return ()


def coordinate_role_labels(schema: object, coordinates) -> dict:
    """Annotate displayed coordinates from bounded, stored declarations only."""
    roles = (
        ("sequence", _declared_role_names(schema, ("core", "roles", "sequence_dim"))),
        ("batch", _declared_role_names(schema, ("core", "roles", "batch_dims"), multiple=True)),
        ("core", _declared_role_names(schema, ("core", "roles", "core_dims"), multiple=True)),
        ("parameter", _declared_role_names(schema, ("core", "param_coord", "name"))),
    )
    return {
        name: ", ".join(role for role, names in roles
                        if name in names and (role == "parameter" or variable.dims == (name,)))
        for name, variable in coordinates.items()
    }


def summary_text(rows: tuple[DisplayRow, ...]) -> str:
    """Bound each declaration and the complete summary, without hiding later rows."""
    lines = ["TAL:"]
    remaining = SUMMARY_BUDGET.text
    items = SUMMARY_BUDGET.items
    for index, row in enumerate(rows):
        budget = DisplayBudget(
            SUMMARY_BUDGET.depth, items,
            max(0, remaining // (len(rows) - index) - len(row.label) - 8),
        )
        value, used = format_stored_usage(row.value, budget=budget, quote_strings=False)
        items -= used
        line = f"    {row.label}: {value}"
        if len(line) > remaining:
            lines.append("    <summary text omitted>")
            break
        lines.append(line)
        remaining -= len(line) + 1
    return "\n".join(lines)
