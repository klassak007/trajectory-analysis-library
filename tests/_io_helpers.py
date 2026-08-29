from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast


class _TemporaryDirectoryLike(Protocol):
    name: str

    def cleanup(self) -> None: ...


def cleanup_failing_temporary_directory(
    real_temporary_directory: Callable[..., object],
    *,
    message: str,
    error_type: type[BaseException] = OSError,
) -> type[_TemporaryDirectoryLike]:
    """Build one reusable TemporaryDirectory cleanup-failure test double."""

    class CleanupFailingTemporaryDirectory:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._temporary = cast(
                _TemporaryDirectoryLike,
                real_temporary_directory(*args, **kwargs),
            )
            self.name = self._temporary.name

        def cleanup(self) -> None:
            self._temporary.cleanup()
            raise error_type(message)

    return CleanupFailingTemporaryDirectory


__all__ = ["cleanup_failing_temporary_directory"]
