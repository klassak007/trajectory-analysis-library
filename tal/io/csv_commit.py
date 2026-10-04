from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import stat as stat_module

from .adapter_paths import ExportDestinationPreflight, plan_export_parent_directories

_STAGED_FILE_PREFIX = ".tal-csv-"
_STAGED_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)


@dataclass(frozen=True)
class _CommitEntry:
    destination: str
    temporary: str


@dataclass(frozen=True)
class CsvCommitPlan:
    """Same-directory temporary paths owned by one CSV export."""

    entries: tuple[_CommitEntry, ...]

    @property
    def temporary_paths(self) -> tuple[str, ...]:
        return tuple(entry.temporary for entry in self.entries)


def _require_matching_preflight(
    destinations: Sequence[str],
    preflight: Sequence[ExportDestinationPreflight],
    *,
    owner: str,
) -> None:
    if len(destinations) != len(preflight):
        raise ValueError(f"{owner}: CSV export destination/preflight count mismatch.")
    pairs = zip(destinations, preflight, strict=True)
    if any(path != snapshot.path for path, snapshot in pairs):
        raise ValueError(f"{owner}: CSV export destination/preflight path mismatch.")


def _create_export_directories(
    preflight: Sequence[ExportDestinationPreflight], *, owner: str
) -> None:
    try:
        for directory in plan_export_parent_directories(preflight, owner=owner):
            directory.mkdir()
    except Exception as exc:
        raise ValueError(f"{owner}: failed creating CSV export directories.") from exc


def _discard_allocated_temporary(temporary: Path) -> None:
    try:
        temporary.unlink()
    except BaseException:
        pass


def _close_allocated_temporary(fd: int, temporary: Path) -> None:
    try:
        os.close(fd)
    except BaseException:
        _discard_allocated_temporary(temporary)
        raise


def _allocate_temporary(destination: str) -> str:
    parent = Path(destination).parent
    for _attempt in range(100):
        temporary = parent / f"{_STAGED_FILE_PREFIX}{secrets.token_hex(8)}.tmp"
        try:
            fd = os.open(temporary, _STAGED_FILE_FLAGS, 0o666)
        except FileExistsError:  # pragma: no cover - random collision retry.
            continue
        _close_allocated_temporary(fd, temporary)
        return str(temporary)
    raise FileExistsError("could not allocate a unique CSV temporary file")


def discard_csv_commit(plan: CsvCommitPlan) -> None:
    """Best-effort cleanup of every uncommitted temporary file."""
    for entry in reversed(plan.entries):
        try:
            Path(entry.temporary).unlink(missing_ok=True)
        except BaseException:
            pass


def prepare_csv_commit(
    destinations: Sequence[str],
    *,
    preflight: Sequence[ExportDestinationPreflight],
    owner: str,
) -> CsvCommitPlan:
    """Allocate short same-directory temporary paths after semantic preflight."""
    _require_matching_preflight(destinations, preflight, owner=owner)
    if not destinations:
        return CsvCommitPlan(())
    entries: list[_CommitEntry] = []
    try:
        _create_export_directories(preflight, owner=owner)
        for destination in destinations:
            entries.append(_CommitEntry(destination, _allocate_temporary(destination)))
    except BaseException as exc:
        discard_csv_commit(CsvCommitPlan(tuple(entries)))
        if not isinstance(exc, Exception):
            raise
        raise ValueError(f"{owner}: failed preparing CSV export destinations.") from exc
    return CsvCommitPlan(tuple(entries))


def _preserve_destination_mode(entry: _CommitEntry) -> None:
    try:
        destination_stat = os.stat(entry.destination, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat_module.S_ISREG(destination_stat.st_mode):
        raise ValueError(f"CSV export destination {entry.destination!r} is not a regular file.")
    os.chmod(entry.temporary, stat_module.S_IMODE(destination_stat.st_mode))


def _raise_commit_failure(exc: BaseException, *, owner: str, destination: str) -> None:
    if not isinstance(exc, Exception):
        raise exc
    raise ValueError(
        f"{owner}: failed writing CSV export destination {destination!r}."
    ) from exc


def commit_csv_export(plan: CsvCommitPlan, *, owner: str) -> None:
    """Replace complete destinations sequentially; the batch is nontransactional."""
    for index, entry in enumerate(plan.entries):
        try:
            _preserve_destination_mode(entry)
            os.replace(entry.temporary, entry.destination)
        except BaseException as exc:
            discard_csv_commit(CsvCommitPlan(plan.entries[index:]))
            _raise_commit_failure(exc, owner=owner, destination=entry.destination)


__all__ = [
    "CsvCommitPlan",
    "commit_csv_export",
    "discard_csv_commit",
    "prepare_csv_commit",
]
