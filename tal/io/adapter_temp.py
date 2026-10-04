from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from tempfile import TemporaryDirectory

from .adapter_cleanup import suppress_cleanup_during_active_error


@contextmanager
def owned_temporary_directory(
    *,
    prefix: str,
    owner: str,
    purpose: str,
) -> Iterator[str]:
    """Create and clean one temporary directory under a public I/O owner."""
    try:
        temporary = TemporaryDirectory(prefix=prefix)
    except Exception as exc:  # pragma: no cover - platform allocation envelope.
        raise ValueError(f"{owner}: failed creating {purpose}.") from exc
    try:
        yield temporary.name
    except BaseException:
        suppress_cleanup_during_active_error(temporary.cleanup)
        raise
    try:
        temporary.cleanup()
    except Exception as exc:  # pragma: no cover - platform cleanup envelope.
        raise ValueError(f"{owner}: failed cleaning {purpose}.") from exc


__all__ = ["owned_temporary_directory"]
