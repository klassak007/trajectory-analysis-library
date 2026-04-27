from __future__ import annotations

from typing import Any


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<unrepr:{type(value).__name__}>"


def _format_schema_error(
    code: str,
    path: str,
    expected: Any,
    actual: Any,
    hint: str,
) -> str:
    expected_repr = _safe_repr(expected)
    actual_repr = _safe_repr(actual)
    return (
        f"[{code}] at {path}: expected={expected_repr}, actual={actual_repr}. "
        f"Fix: {hint}"
    )


class SchemaError(ValueError):
    """Typed schema error with stable machine-readable fields.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    __slots__ = ("code", "path", "expected", "actual", "hint")

    def __init__(
        self,
        *,
        code: str,
        path: str,
        expected: Any,
        actual: Any,
        hint: str,
    ) -> None:
        self.code = code
        self.path = path
        self.expected = expected
        self.actual = actual
        self.hint = hint
        super().__init__(_format_schema_error(code, path, expected, actual, hint))


def schema_error(
    *,
    code: str,
    path: str,
    expected: Any,
    actual: Any,
    hint: str,
) -> SchemaError:
    """Create a SchemaError instance.

    Parameters
    ----------
    code : str, optional
        Validation/diagnostic metadata used for deterministic error reporting.
    path : str, optional
        Filesystem path or logical path input.
    expected : Any, optional
        Frame/schema rewrite selector used by this operation.
    actual : Any, optional
        Frame/schema rewrite selector used by this operation.
    hint : str, optional
        Validation/diagnostic metadata used for deterministic error reporting.

    Returns
    -------
    SchemaError
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return SchemaError(
        code=code,
        path=path,
        expected=expected,
        actual=actual,
        hint=hint,
    )
