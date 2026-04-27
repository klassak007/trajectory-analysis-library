from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    return value


def resolve_sidecar_path(
    csv_path: str,
    *,
    metadata_path: str | None,
    owner: str,
) -> Path:
    if metadata_path is not None:
        return Path(metadata_path)
    csv_file = Path(csv_path)
    if csv_file.suffix:
        return csv_file.with_suffix(".tal.json")
    if csv_file.name:
        return csv_file.with_name(f"{csv_file.name}.tal.json")
    raise ValueError(f"{owner}: could not resolve metadata sidecar path from csv path {csv_path!r}.")


def write_sidecar_metadata(
    csv_path: str,
    payload: dict[str, Any],
    *,
    metadata_path: str | None,
    owner: str,
) -> str:
    sidecar = resolve_sidecar_path(csv_path, metadata_path=metadata_path, owner=owner)
    try:
        sidecar.write_text(
            json.dumps(_json_compatible(payload), sort_keys=True, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ValueError(f"{owner}: failed writing metadata sidecar {str(sidecar)!r}.") from exc
    return str(sidecar)


def read_sidecar_metadata(
    csv_path: str,
    *,
    metadata_path: str | None,
    owner: str,
) -> dict[str, Any]:
    sidecar = resolve_sidecar_path(csv_path, metadata_path=metadata_path, owner=owner)
    try:
        text = sidecar.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{owner}: metadata sidecar {str(sidecar)!r} was not found or unreadable.") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{owner}: metadata sidecar {str(sidecar)!r} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{owner}: metadata sidecar payload must be a JSON object.")
    return payload


__all__ = [
    "read_sidecar_metadata",
    "resolve_sidecar_path",
    "write_sidecar_metadata",
]
