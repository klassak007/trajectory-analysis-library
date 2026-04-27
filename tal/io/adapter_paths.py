from __future__ import annotations

import glob
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolvedIngestInput:
    label: str
    path: str
    resolved_path: str
    explicit_label: bool


def _resolve_existing_file(path: str, *, owner: str) -> str:
    if not isinstance(path, str) or not path:
        raise TypeError(f"{owner}: input path entries must be non-empty strings.")
    resolved = str(Path(path).expanduser().resolve())
    if not Path(resolved).exists():
        raise ValueError(f"{owner}: input path {path!r} does not exist.")
    if not Path(resolved).is_file():
        raise ValueError(f"{owner}: input path {path!r} must resolve to a file.")
    return resolved


def _require_unique_resolved_paths(paths: list[str], *, owner: str) -> None:
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            raise ValueError(
                f"{owner}: duplicate resolved input path {path!r} is not supported; "
                "use unique inputs only."
            )
        seen.add(path)


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


def _mapping_records(value: Mapping[str, str], *, owner: str) -> tuple[ResolvedIngestInput, ...]:
    labels: list[str] = []
    raw_paths: list[str] = []
    resolved_paths: list[str] = []
    for label, path in value.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"{owner}: mapping labels must be non-empty strings; got {label!r}.")
        labels.append(label)
        raw_paths.append(path)
        resolved_paths.append(_resolve_existing_file(path, owner=owner))
    _require_unique_labels(labels, owner=owner)
    _require_unique_resolved_paths(resolved_paths, owner=owner)
    return tuple(
        ResolvedIngestInput(
            label=label,
            path=path,
            resolved_path=resolved,
            explicit_label=True,
        )
        for label, path, resolved in zip(labels, raw_paths, resolved_paths, strict=True)
    )


def _sequence_records(value: Sequence[str], *, owner: str) -> tuple[ResolvedIngestInput, ...]:
    raw_paths = [str(path) for path in value]
    resolved_paths = [_resolve_existing_file(path, owner=owner) for path in raw_paths]
    _require_unique_resolved_paths(resolved_paths, owner=owner)
    labels = _derive_stem_labels(raw_paths, owner=owner)
    return tuple(
        ResolvedIngestInput(
            label=label,
            path=path,
            resolved_path=resolved,
            explicit_label=False,
        )
        for label, path, resolved in zip(labels, raw_paths, resolved_paths, strict=True)
    )


def _glob_records(pattern: str, *, owner: str) -> tuple[ResolvedIngestInput, ...]:
    matches = glob.glob(pattern)
    if not matches:
        raise ValueError(f"{owner}: glob pattern {pattern!r} matched no files.")
    entries: list[tuple[str, str]] = []
    for path in matches:
        resolved = _resolve_existing_file(path, owner=owner)
        entries.append((path, resolved))
    entries.sort(key=lambda item: item[1])
    raw_paths = [item[0] for item in entries]
    resolved_paths = [item[1] for item in entries]
    _require_unique_resolved_paths(resolved_paths, owner=owner)
    labels = _derive_stem_labels(raw_paths, owner=owner)
    return tuple(
        ResolvedIngestInput(
            label=label,
            path=path,
            resolved_path=resolved,
            explicit_label=False,
        )
        for label, path, resolved in zip(labels, raw_paths, resolved_paths, strict=True)
    )


def resolve_ingest_inputs(
    value: str | Sequence[str] | Mapping[str, str],
    *,
    owner: str,
) -> tuple[ResolvedIngestInput, ...]:
    if isinstance(value, str):
        return _glob_records(value, owner=owner)
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{owner}: mapping input must not be empty.")
        return _mapping_records(value, owner=owner)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"{owner}: sequence input must not be empty.")
        return _sequence_records(value, owner=owner)
    raise TypeError(
        f"{owner}: inputs must be a glob string, sequence of file paths, or mapping label->path; "
        f"got {type(value).__name__}."
    )


def normalize_export_label_path(
    label: object,
    *,
    out_dir: str,
    owner: str,
) -> str:
    if not isinstance(out_dir, str) or not out_dir:
        raise TypeError(f"{owner}: out_dir must be a non-empty string.")
    root = Path(out_dir).expanduser().resolve()
    raw = str(label)
    if not raw:
        raise ValueError(f"{owner}: export label must not be empty.")
    normalized = raw.replace("\\", "/")
    rel = Path(normalized)
    if rel.is_absolute():
        raise ValueError(f"{owner}: export label {raw!r} resolves to an absolute path.")
    if any(part == ".." for part in rel.parts):
        raise ValueError(f"{owner}: export label {raw!r} contains '..' traversal segments.")
    path = (root / rel).with_suffix(".csv").resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{owner}: normalized export path {str(path)!r} escapes root {str(root)!r}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


__all__ = ["ResolvedIngestInput", "normalize_export_label_path", "resolve_ingest_inputs"]
