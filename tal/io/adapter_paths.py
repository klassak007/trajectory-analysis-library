from __future__ import annotations

import glob
import stat as stat_module
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolvedIngestInput:
    label: str
    resolved_path: str


@dataclass(frozen=True)
class ExportDestinationPreflight:
    """Immutable destination plan captured before export payload execution."""

    path: str
    root: str
    ancestor: str


def _resolve_existing_path(
    path: str,
    *,
    owner: str,
    allow_directories: bool,
) -> str:
    if not isinstance(path, str) or not path:
        raise TypeError(f"{owner}: input path entries must be non-empty strings.")
    try:
        resolved_path = Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{owner}: failed resolving input path {path!r}.") from exc
    try:
        path_stat = resolved_path.stat()
    except FileNotFoundError:
        raise ValueError(f"{owner}: input path {path!r} does not exist.")
    except OSError as exc:
        raise ValueError(f"{owner}: failed inspecting input path {path!r}.") from exc
    is_file = stat_module.S_ISREG(path_stat.st_mode)
    is_allowed_directory = allow_directories and stat_module.S_ISDIR(path_stat.st_mode)
    if not is_file and not is_allowed_directory:
        expected = "a file or directory" if allow_directories else "a file"
        raise ValueError(f"{owner}: input path {path!r} must resolve to {expected}.")
    return str(resolved_path)


def existing_filesystem_identity(
    path: str,
    *,
    owner: str,
) -> tuple[int, int] | None:
    """Return a device/inode identity when an existing path supplies one."""
    try:
        path_stat = Path(path).stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:  # pragma: no cover - filesystem race envelope.
        raise ValueError(f"{owner}: failed reading filesystem identity for {path!r}.") from exc
    inode = int(path_stat.st_ino)
    if not inode:
        return None
    return int(path_stat.st_dev), inode


def _require_unique_resolved_paths(paths: list[str], *, owner: str) -> None:
    seen: set[tuple[object, ...]] = set()
    for path in paths:
        file_identity = existing_filesystem_identity(path, owner=owner)
        identity: tuple[object, ...] = ("path", filesystem_collision_key(path))
        if file_identity is not None:
            identity = ("inode", *file_identity)
        if identity in seen:
            raise ValueError(
                f"{owner}: duplicate resolved input path {path!r} is not supported; "
                "use unique inputs only."
            )
        seen.add(identity)


def _derive_stem_labels(paths: list[str], *, owner: str) -> list[str]:
    labels: list[str] = []
    for path in paths:
        stem = Path(path).stem
        if not stem:
            raise ValueError(f"{owner}: could not derive non-empty label stem from path {path!r}.")
        labels.append(stem)
    _require_unique_labels(labels, owner=owner)
    return labels


def _require_unique_labels(labels: Sequence[str], *, owner: str) -> None:
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            raise ValueError(
                f"{owner}: derived ingest label collision for {label!r}; provide mapping input "
                "with explicit unique labels."
            )
        seen.add(label)


def _mapping_records(
    value: Mapping[str, str],
    *,
    owner: str,
    allow_directories: bool,
) -> tuple[ResolvedIngestInput, ...]:
    labels: list[str] = []
    resolved_paths: list[str] = []
    for label, path in value.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"{owner}: mapping labels must be non-empty strings; got {label!r}.")
        labels.append(label)
        resolved_paths.append(
            _resolve_existing_path(
                path,
                owner=owner,
                allow_directories=allow_directories,
            )
        )
    _require_unique_labels(labels, owner=owner)
    _require_unique_resolved_paths(resolved_paths, owner=owner)
    return tuple(
        ResolvedIngestInput(
            label=label,
            resolved_path=resolved,
        )
        for label, resolved in zip(labels, resolved_paths, strict=True)
    )


def _sequence_records(
    value: Sequence[str],
    *,
    owner: str,
    allow_directories: bool,
) -> tuple[ResolvedIngestInput, ...]:
    raw_paths = list(value)
    resolved_paths = [
        _resolve_existing_path(
            path,
            owner=owner,
            allow_directories=allow_directories,
        )
        for path in raw_paths
    ]
    _require_unique_resolved_paths(resolved_paths, owner=owner)
    labels = _derive_stem_labels(raw_paths, owner=owner)
    return tuple(
        ResolvedIngestInput(
            label=label,
            resolved_path=resolved,
        )
        for label, resolved in zip(labels, resolved_paths, strict=True)
    )


def _glob_records(
    pattern: str,
    *,
    owner: str,
    allow_directories: bool,
) -> tuple[ResolvedIngestInput, ...]:
    try:
        expanded_pattern = str(Path(pattern).expanduser())
        matches = glob.glob(expanded_pattern)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{owner}: failed resolving glob pattern {pattern!r}.") from exc
    if not matches:
        raise ValueError(f"{owner}: glob pattern {pattern!r} matched no files.")
    entries: list[tuple[str, str]] = []
    for path in matches:
        resolved = _resolve_existing_path(
            path,
            owner=owner,
            allow_directories=allow_directories,
        )
        entries.append((path, resolved))
    entries.sort(key=lambda item: item[1])
    raw_paths = [item[0] for item in entries]
    resolved_paths = [item[1] for item in entries]
    _require_unique_resolved_paths(resolved_paths, owner=owner)
    labels = _derive_stem_labels(raw_paths, owner=owner)
    return tuple(
        ResolvedIngestInput(
            label=label,
            resolved_path=resolved,
        )
        for label, resolved in zip(labels, resolved_paths, strict=True)
    )


def resolve_ingest_inputs(
    value: str | Sequence[str] | Mapping[str, str],
    *,
    owner: str,
    allow_directories: bool = False,
) -> tuple[ResolvedIngestInput, ...]:
    if isinstance(value, str):
        return _glob_records(
            value,
            owner=owner,
            allow_directories=allow_directories,
        )
    if isinstance(value, Mapping):
        if len(value) == 0:
            raise ValueError(f"{owner}: mapping input must not be empty.")
        return _mapping_records(
            value,
            owner=owner,
            allow_directories=allow_directories,
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 0:
            raise ValueError(f"{owner}: sequence input must not be empty.")
        return _sequence_records(
            value,
            owner=owner,
            allow_directories=allow_directories,
        )
    raise TypeError(
        f"{owner}: inputs must be a glob string, sequence of file paths, or mapping label->path; "
        f"got {type(value).__name__}."
    )


def _resolve_export_path(path: Path, *, owner: str, description: str) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{owner}: failed resolving {description}.") from exc


def stringify_export_identity(
    value: object,
    *,
    owner: str,
    description: str,
) -> str:
    """Normalize one public export identity under the caller's error owner."""
    try:
        return str(value)
    except Exception as exc:
        raise ValueError(
            f"{owner}: failed normalizing {description} to a string."
        ) from exc


def _export_path_info(
    path: Path,
    *,
    owner: str,
) -> tuple[int, tuple[int, int]] | None:
    try:
        path_stat = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:  # pragma: no cover - filesystem race envelope.
        raise ValueError(f"{owner}: failed inspecting export path {str(path)!r}.") from exc
    return int(path_stat.st_mode), (int(path_stat.st_dev), int(path_stat.st_ino))


def _nearest_existing_path(
    path: Path,
    *,
    owner: str,
) -> tuple[Path, int, tuple[int, int]]:
    candidate = path
    while candidate != candidate.parent:
        info = _export_path_info(candidate, owner=owner)
        if info is not None:
            return candidate, *info
        candidate = candidate.parent
    info = _export_path_info(candidate, owner=owner)
    if info is None:  # pragma: no cover - filesystem invariant.
        raise ValueError(f"{owner}: could not find an existing export path ancestor.")
    return candidate, *info


def resolve_export_root(out_dir: str, *, owner: str) -> Path:
    """Resolve one export root before label-dependent path planning."""
    if not isinstance(out_dir, str) or not out_dir:
        raise TypeError(f"{owner}: out_dir must be a non-empty string.")
    try:
        root_input = Path(out_dir).expanduser()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{owner}: failed resolving export root {out_dir!r}.") from exc
    root = _resolve_export_path(
        root_input,
        owner=owner,
        description=f"export root {out_dir!r}",
    )
    ancestor, ancestor_mode, _identity = _nearest_existing_path(root, owner=owner)
    if not stat_module.S_ISDIR(ancestor_mode):
        raise ValueError(
            f"{owner}: export parent path {str(ancestor)!r} is not a directory."
        )
    return root


def _require_stable_export_resolution(
    path: Path,
    *,
    root: Path,
    owner: str,
) -> None:
    resolved_root = _resolve_export_path(root, owner=owner, description="export root")
    resolved_path = _resolve_export_path(path, owner=owner, description="export destination")
    if resolved_root != root or resolved_path != path or not path.is_relative_to(root):
        raise ValueError(
            f"{owner}: export path topology changed for destination {str(path)!r}."
        )


def snapshot_export_destinations(
    paths: Sequence[str],
    *,
    root: Path,
    owner: str,
) -> tuple[ExportDestinationPreflight, ...]:
    """Validate destinations and snapshot their pre-execution filesystem state."""
    snapshots: list[ExportDestinationPreflight] = []
    for raw_path in paths:
        path = Path(raw_path)
        _require_stable_export_resolution(path, root=root, owner=owner)
        path_info = _export_path_info(path, owner=owner)
        if path_info is not None and stat_module.S_ISDIR(path_info[0]):
            raise ValueError(
                f"{owner}: export destination {raw_path!r} is an existing directory."
            )
        if path_info is not None and not stat_module.S_ISREG(path_info[0]):
            raise ValueError(
                f"{owner}: export destination {raw_path!r} is not a regular file."
            )
        ancestor, ancestor_mode, _ancestor_identity = _nearest_existing_path(
            path.parent,
            owner=owner,
        )
        if not stat_module.S_ISDIR(ancestor_mode):
            raise ValueError(
                f"{owner}: export parent path {str(ancestor)!r} is not a directory."
            )
        snapshots.append(
            ExportDestinationPreflight(
                path=raw_path,
                root=str(root),
                ancestor=str(ancestor),
            )
        )
    return tuple(snapshots)


def _planned_parent_directories(
    snapshot: ExportDestinationPreflight,
    *,
    owner: str,
) -> tuple[Path, ...]:
    ancestor = Path(snapshot.ancestor)
    try:
        parts = Path(snapshot.path).parent.relative_to(ancestor).parts
    except ValueError as exc:  # pragma: no cover - preflight invariant.
        raise ValueError(f"{owner}: export parent escaped its verified ancestor.") from exc
    return tuple(ancestor.joinpath(*parts[:index]) for index in range(1, len(parts) + 1))


def _append_export_parent_directory(
    planned: list[Path],
    seen: dict[str, Path],
    candidate: Path,
    *,
    owner: str,
) -> None:
    key = filesystem_collision_key(str(candidate))
    previous = seen.get(key)
    if previous is None:
        seen[key] = candidate
        planned.append(candidate)
        return
    if previous != candidate:
        raise ValueError(
            f"{owner}: planned export parent directory collision between "
            f"{str(previous)!r} and {str(candidate)!r}."
        )


def plan_export_parent_directories(
    snapshots: Sequence[ExportDestinationPreflight],
    *,
    owner: str,
) -> tuple[Path, ...]:
    """Derive the ordered, unique directory-creation plan from preflight state."""
    planned: list[Path] = []
    seen: dict[str, Path] = {}
    for snapshot in snapshots:
        for candidate in _planned_parent_directories(snapshot, owner=owner):
            _append_export_parent_directory(planned, seen, candidate, owner=owner)
    return tuple(planned)


def normalize_export_label_path(
    label: object,
    *,
    root: Path,
    owner: str,
) -> str:
    raw = stringify_export_identity(
        label,
        owner=owner,
        description="export label",
    )
    if not raw:
        raise ValueError(f"{owner}: export label must not be empty.")
    normalized = raw.replace("\\", "/")
    rel = Path(normalized)
    if rel.is_absolute():
        raise ValueError(f"{owner}: export label {raw!r} resolves to an absolute path.")
    if any(part == ".." for part in rel.parts):
        raise ValueError(f"{owner}: export label {raw!r} contains '..' traversal segments.")
    try:
        destination = (root / rel).with_suffix(".csv")
    except ValueError as exc:
        raise ValueError(f"{owner}: failed resolving export label {raw!r}.") from exc
    path = _resolve_export_path(
        destination,
        owner=owner,
        description=f"export label {raw!r}",
    )
    if not path.is_relative_to(root):
        raise ValueError(f"{owner}: normalized export path {str(path)!r} escapes root {str(root)!r}.")
    return str(path)


def filesystem_collision_key(path: str) -> str:
    """Return a deterministic key for conservative filesystem collision checks."""
    normalized = unicodedata.normalize("NFC", path)
    return unicodedata.normalize("NFC", normalized.casefold())


__all__ = [
    "ExportDestinationPreflight",
    "ResolvedIngestInput",
    "existing_filesystem_identity",
    "filesystem_collision_key",
    "normalize_export_label_path",
    "plan_export_parent_directories",
    "resolve_export_root",
    "resolve_ingest_inputs",
    "snapshot_export_destinations",
    "stringify_export_identity",
]
