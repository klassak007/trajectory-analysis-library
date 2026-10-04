from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AdapterArraySpool:
    """On-disk normalized arrays for one adapter record."""

    root: str
    size: int
    field_count: int


def _require_record_array(
    value: np.ndarray,
    *,
    expected_size: int | None,
    owner: str,
    label: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{owner}: {label} must be one-dimensional; got shape {array.shape!r}.")
    if expected_size is not None and array.size != expected_size:
        raise ValueError(
            f"{owner}: {label} length {array.size} does not match record length {expected_size}."
        )
    return array


def spool_adapter_arrays(
    spool_dir: str,
    *,
    row: int,
    time_values: np.ndarray,
    field_values: Sequence[np.ndarray],
    owner: str,
) -> AdapterArraySpool:
    """Persist one normalized record without retaining its arrays in a plan."""
    time_array = _require_record_array(
        time_values,
        expected_size=None,
        owner=owner,
        label="record time values",
    )
    root = Path(spool_dir) / f"record-{row:08d}"
    try:
        root.mkdir()
        np.save(root / "time.npy", time_array, allow_pickle=False)
        for index, value in enumerate(field_values):
            array = _require_record_array(
                value,
                expected_size=int(time_array.size),
                owner=owner,
                label=f"record field {index}",
            )
            np.save(root / f"field-{index:08d}.npy", array, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{owner}: failed spooling normalized adapter record {row}.") from exc
    return AdapterArraySpool(
        root=str(root),
        size=int(time_array.size),
        field_count=len(field_values),
    )


def _load_spooled_array(path: Path, *, size: int, owner: str) -> np.ndarray:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{owner}: failed reading normalized adapter spool {str(path)!r}.") from exc
    if value.ndim != 1 or value.size != size:
        raise ValueError(
            f"{owner}: normalized adapter spool {str(path)!r} has unexpected shape {value.shape!r}."
        )
    return value


def fill_spooled_adapter_row(
    spool: AdapterArraySpool,
    *,
    row: int,
    time_target: np.ndarray,
    field_targets: Sequence[np.ndarray],
    owner: str,
) -> None:
    """Copy one memory-mapped record into its final padded row."""
    if len(field_targets) != spool.field_count:
        raise ValueError(
            f"{owner}: adapter spool field count {spool.field_count} does not match "
            f"target count {len(field_targets)}."
        )
    root = Path(spool.root)
    time_target[row, : spool.size] = _load_spooled_array(
        root / "time.npy",
        size=spool.size,
        owner=owner,
    )
    for index, target in enumerate(field_targets):
        target[row, : spool.size] = _load_spooled_array(
            root / f"field-{index:08d}.npy",
            size=spool.size,
            owner=owner,
        )


__all__ = ["AdapterArraySpool", "fill_spooled_adapter_row", "spool_adapter_arrays"]
