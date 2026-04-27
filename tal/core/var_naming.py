from __future__ import annotations

DEFAULT_DATAVAR_NAME = "datavar"


def default_datavar_name() -> str:
    """Return the canonical neutral payload variable name.

    Parameters
    ----------
    None
        This callable does not accept user-facing parameters.

    Returns
    -------
    str
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return DEFAULT_DATAVAR_NAME


def is_valid_user_var_name(name: object) -> bool:
    """Return True when ``name`` is a non-empty string.

    Parameters
    ----------
    name : object
        Identifier/name used for lookup or registration.

    Returns
    -------
    bool
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return isinstance(name, str) and name != ""


def fallback_var_name(name: str | None) -> str:
    """Return ``name`` when valid; otherwise return ``datavar``.

    Parameters
    ----------
    name : str | None
        Identifier/name used for lookup or registration.

    Returns
    -------
    str
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if is_valid_user_var_name(name):
        return name
    return default_datavar_name()


def preserve_or_datavar(name: str | None) -> str:
    """Preserve a semantic carrier name when available, else ``datavar``.

    Parameters
    ----------
    name : str | None
        Identifier/name used for lookup or registration.

    Returns
    -------
    str
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return fallback_var_name(name)


__all__ = [
    "default_datavar_name",
    "fallback_var_name",
    "is_valid_user_var_name",
    "preserve_or_datavar",
]
