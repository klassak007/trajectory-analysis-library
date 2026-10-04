from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np

NATIVE_SIZE = 129
H0_DIRECTIONS = (1,) * 8
H1_DIRECTIONS = (1, 1, 1, 1, -1, -1, -1, -1)


@dataclass(frozen=True)
class FrozenPathFixture:
    """Schema-free representation of one frozen benchmark case."""

    name: str
    parameter: np.ndarray
    query: np.ndarray
    translation: np.ndarray
    quaternion: np.ndarray
    relations: tuple[tuple[str, str], ...]
    source: str
    destination: str
    directions: tuple[int, ...]


@dataclass(frozen=True)
class PreparedFixture:
    """Frozen fixture plus one core-owned bracket map."""

    fixture: FrozenPathFixture
    i0: np.ndarray
    i1: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class BenchmarkCaseConfig:
    """Complete configuration shared by every measurement mode."""

    case: str
    query_size: int
    edges: int


@dataclass(frozen=True)
class TimingResult:
    route: str
    effective_backend: str
    samples: tuple[float, ...]

    @property
    def median(self) -> float:
        return median(self.samples)


@dataclass(frozen=True)
class RssResult:
    route: str
    config: BenchmarkCaseConfig
    effective_backend: str
    baseline: int
    peak: int
    sample_count: int

    @property
    def increment(self) -> int:
        return max(0, self.peak - self.baseline)


@dataclass(frozen=True)
class ColdTimingResult:
    """Fresh-process timing with its worker-confirmed configuration."""

    route: str
    config: BenchmarkCaseConfig
    effective_backend: str
    elapsed: float


def _axis_for_edge(edge: int) -> np.ndarray:
    raw = np.array((1 + edge, 2 + edge % 3, 3 + (2 * edge) % 5), dtype=np.float64)
    return raw / np.linalg.norm(raw)


def _h1_values(parameter: np.ndarray, edge: int) -> tuple[np.ndarray, np.ndarray]:
    translation = np.empty((parameter.size, 3), dtype=np.float64)
    translation[:, 0] = 0.25 * (edge + 1) + (0.5 + edge / 32.0) * parameter
    translation[:, 1] = np.sin(0.3 * (edge + 1) + (0.75 + edge / 16.0) * parameter)
    translation[:, 2] = np.cos(0.2 * (edge + 1) - (0.5 + edge / 32.0) * parameter)
    half_angle = 0.5 * (0.125 * (edge + 1) + (9.0 + edge / 4.0) * parameter)
    quaternion = np.empty((parameter.size, 4), dtype=np.float64)
    quaternion[:, :3] = np.sin(half_angle)[:, None] * _axis_for_edge(edge)
    quaternion[:, 3] = np.cos(half_angle)
    return translation, quaternion


def _h0_values(parameter: np.ndarray, _edge: int) -> tuple[np.ndarray, np.ndarray]:
    translation = np.zeros((parameter.size, 3), dtype=np.float64)
    translation[:, 0] = parameter
    quaternion = np.zeros((parameter.size, 4), dtype=np.float64)
    quaternion[:, 3] = 1.0
    return translation, quaternion


def _relations(case: str, edges: int) -> tuple[tuple[str, str], ...]:
    if case == "h0":
        return tuple((f"f{edge}", f"f{edge + 1}") for edge in range(edges))
    midpoint = edges // 2
    return tuple(
        (f"f{edge + 1}", f"f{edge}") if edge < midpoint else (f"f{edge}", f"f{edge + 1}")
        for edge in range(edges)
    )


def frozen_fixture(case: str, *, query_size: int, edges: int = 8) -> FrozenPathFixture:
    """Build the frozen H0 or H1 spatial-path fixture."""
    if case not in {"h0", "h1"}:
        raise ValueError(f"unknown benchmark case {case!r}")
    parameter = np.arange(NATIVE_SIZE, dtype=np.float64) / 128.0
    query = np.arange(query_size, dtype=np.float64) / max(query_size - 1, 1)
    builder = _h0_values if case == "h0" else _h1_values
    values = tuple(builder(parameter, edge) for edge in range(edges))
    translation = np.stack(tuple(value[0] for value in values))
    quaternion = np.stack(tuple(value[1] for value in values))
    directions = (1,) * edges if case == "h0" else tuple((1 if edge < edges // 2 else -1) for edge in range(edges))
    if edges == 8:
        expected = H0_DIRECTIONS if case == "h0" else H1_DIRECTIONS
        if directions != expected:
            raise AssertionError(f"{case.upper()} direction fixture drifted: {directions!r}")
    source, destination = (f"f{edges}", "f0") if case == "h0" else ("f0", f"f{edges}")
    return FrozenPathFixture(
        case,
        parameter,
        query,
        translation,
        quaternion,
        _relations(case, edges),
        source,
        destination,
        directions,
    )


__all__ = [
    "H0_DIRECTIONS",
    "H1_DIRECTIONS",
    "NATIVE_SIZE",
    "BenchmarkCaseConfig",
    "ColdTimingResult",
    "FrozenPathFixture",
    "PreparedFixture",
    "RssResult",
    "TimingResult",
    "frozen_fixture",
]
