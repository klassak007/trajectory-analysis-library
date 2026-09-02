from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..schema_errors import schema_error

SCHEMA_VERSION = 1
ALLOWED_CORE_KEYS = {"roles", "param_coord", "validity"}
ALLOWED_ROLE_KEYS = {"sequence_dim", "batch_dims", "core_dims"}
ALLOWED_PARAM_KEYS = {"name"}
ALLOWED_VALIDITY_KEYS = {"sequence_size_coord", "layout"}
ALLOWED_LAYOUTS = {"left_packed"}


def is_active_schema_version(value: Any) -> bool:
    """Return whether *value* is the exact supported schema version type/value."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value == SCHEMA_VERSION
    )


def fail(
    *,
    code: str,
    path: str,
    expected: Any,
    actual: Any,
    hint: str,
) -> None:
    raise schema_error(
        code=code,
        path=path,
        expected=expected,
        actual=actual,
        hint=hint,
    )


def is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def is_valid_name(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def is_valid_name_list(value: Any) -> bool:
    return isinstance(value, list) and all(is_valid_name(item) for item in value)


def first_duplicate(values: list[str]) -> str | None:
    seen: set[str] = set()
    for item in values:
        if item in seen:
            return item
        seen.add(item)
    return None


def safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<unrepr:{type(value).__name__}>"


def safe_path_key_segment(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, (int, float, bool)) or key is None:
        return str(key)
    return f"<{type(key).__name__}>"


def unknown_key_path(prefix: str, key: Any) -> str:
    return f"{prefix}.{safe_path_key_segment(key)}"


def sort_token(value: Any) -> tuple[str, str]:
    return type(value).__name__, safe_repr(value)


def sorted_mapping_keys(mapping: Mapping[Any, Any]) -> list[Any]:
    return sorted(mapping.keys(), key=sort_token)


def unknown_key_actual(key: Any) -> Any:
    if isinstance(key, str):
        return key
    return {"key": safe_repr(key), "key_type": type(key).__name__}
