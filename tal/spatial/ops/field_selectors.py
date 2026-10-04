"""Immutable selector declarations for typed spatial field construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice

import xarray as xr

_MISSING_PREVIEW = 8


@dataclass(frozen=True)
class FieldSelectorDeclaration:
    """One validated target-label to source-name declaration."""

    slot: str
    fields: tuple[tuple[str, str], ...]

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(source for _, source in self.fields)


@dataclass(frozen=True)
class ExpandedFieldSelection:
    """One declaration expanded with a build-local literal prefix."""

    slot: str
    fields: tuple[tuple[str, str], ...]

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(source for _, source in self.fields)


def _safe_name_repr(value: object) -> str:
    try:
        return repr(value)
    except Exception:  # noqa: BLE001 - diagnostics must survive hostile names
        return f"<unrepr:{type(value).__name__}>"


def _available_preview(ds: xr.Dataset) -> str:
    shown = tuple(
        _safe_name_repr(name) for name in islice(ds.data_vars, _MISSING_PREVIEW)
    )
    omitted = len(ds.data_vars) - len(shown)
    suffix = f" (+{omitted} more)" if omitted else ""
    return f"({', '.join(shown)}){suffix}"


def require_expanded_fields(
    ds: xr.Dataset,
    selections: tuple[ExpandedFieldSelection, ...],
    *,
    owner: str,
) -> None:
    """Require every expanded selector to name an existing data variable."""
    for selection in selections:
        missing = tuple(name for name in selection.source_names if name not in ds.data_vars)
        if not missing:
            continue
        raise ValueError(
            f"{owner}: {selection.slot} fields {missing!r} are not source data variables; "
            f"available={_available_preview(ds)}."
        )


def _parse_shorthand(value: str, *, owner: str) -> tuple[tuple[str, str], ...]:
    if value.count("{") != 1 or value.count("}") != 1:
        raise ValueError(f"{owner}: selector shorthand must contain exactly one brace group.")
    start = value.index("{")
    stop = value.index("}")
    if stop < start or "{" in value[start + 1 :] or "}" in value[:stop]:
        raise ValueError(f"{owner}: selector shorthand contains malformed or nested braces.")
    prefix, suffix = value[:start], value[stop + 1 :]
    labels = tuple(value[start + 1 : stop].split(","))
    if not labels or any(not label for label in labels):
        raise ValueError(f"{owner}: selector shorthand labels must be non-empty.")
    return tuple((label, f"{prefix}{label}{suffix}") for label in labels)


def _parse_mapping(
    value: Mapping[object, object],
    *,
    owner: str,
) -> tuple[tuple[str, str], ...]:
    items = tuple(value.items())
    if not items:
        raise ValueError(f"{owner}: selector mappings must not be empty.")
    if any(type(label) is not str or type(source) is not str for label, source in items):
        raise TypeError(f"{owner}: selector mapping labels and source names must be strings.")
    if any(not label or not source for label, source in items):
        raise ValueError(f"{owner}: selector mapping labels and source names must be non-empty.")
    return tuple((label, source) for label, source in items)


def _require_target_labels(
    fields: tuple[tuple[str, str], ...],
    *,
    expected: tuple[str, ...],
    slot: str,
    owner: str,
) -> None:
    labels = tuple(label for label, _ in fields)
    if len(set(labels)) != len(labels):
        raise ValueError(f"{owner}: {slot} selector contains duplicate target labels.")
    if set(labels) != set(expected) or len(labels) != len(expected):
        raise ValueError(
            f"{owner}: {slot} selector labels must be exactly {expected!r}; got {labels!r}."
        )
    sources = tuple(source for _, source in fields)
    if len(set(sources)) != len(sources):
        raise ValueError(f"{owner}: {slot} selector repeats a source field.")


def prepare_field_selector(
    value: object,
    *,
    expected: tuple[str, ...],
    slot: str,
    owner: str,
) -> FieldSelectorDeclaration:
    """Validate and snapshot one shorthand or mapping declaration."""
    if isinstance(value, str):
        fields = _parse_shorthand(value, owner=owner)
    elif isinstance(value, Mapping):
        fields = _parse_mapping(value, owner=owner)
    else:
        raise TypeError(f"{owner}: {slot} selector must be a shorthand string or mapping.")
    _require_target_labels(fields, expected=expected, slot=slot, owner=owner)
    return FieldSelectorDeclaration(slot=slot, fields=fields)


def expand_field_selector(
    declaration: FieldSelectorDeclaration,
    *,
    prefix: object,
    owner: str,
) -> ExpandedFieldSelection:
    """Apply one build-local literal source-name prefix."""
    if type(prefix) is not str:
        raise TypeError(f"{owner}: prefix must be a string.")
    return ExpandedFieldSelection(
        slot=declaration.slot,
        fields=tuple((label, f"{prefix}{source}") for label, source in declaration.fields),
    )


__all__ = [
    "ExpandedFieldSelection",
    "FieldSelectorDeclaration",
    "expand_field_selector",
    "prepare_field_selector",
    "require_expanded_fields",
]
