from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_ros_text(value: object, *, field: str, owner: str, path: str) -> str:
    """Normalize arbitrary backend text while retaining the public owner."""
    try:
        return str(value)
    except Exception as exc:
        raise ValueError(
            f"{owner}: failed normalizing ROS {field} in {path!r}."
        ) from exc


@dataclass
class RosFrameMetadata:
    """Collapse constant frame metadata and retain varying values only by policy."""

    retain_nonscalar: bool
    count: int = 0
    scalar: str | None = None
    varied: bool = False
    values: list[str | None] | None = None

    def append(self, value: str | None) -> None:
        if self.count == 0:
            self.scalar = value
        elif not self.varied and value != self.scalar:
            self.varied = True
            if self.retain_nonscalar:
                self.values = [self.scalar] * self.count
        if self.values is not None:
            self.values.append(value)
        self.count += 1

    def collapse(self, order: np.ndarray) -> tuple[bool, object]:
        if self.count == 0:
            return False, None
        if not self.varied:
            return self.scalar is not None, self.scalar
        if self.values is None:
            return False, None
        return True, [self.values[int(index)] for index in order]


__all__ = ["RosFrameMetadata", "normalize_ros_text"]
