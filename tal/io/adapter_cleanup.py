from __future__ import annotations

from collections.abc import Callable


def suppress_cleanup_during_active_error(cleanup: Callable[[], None]) -> None:
    """Run best-effort cleanup without replacing an already-active failure."""
    try:
        cleanup()
    except BaseException:
        pass


__all__: list[str] = []
