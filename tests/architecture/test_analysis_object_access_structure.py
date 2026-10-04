from __future__ import annotations

from pathlib import Path

from ._analysis_object_access_guard import (
    analysis_object_access_violations,
    production_access_violations,
)


def _accesses(source: str, *, path: str = "<source>") -> list[str]:
    return [
        item.access for item in analysis_object_access_violations(source, path=path)
    ]


def test_arch_ao_access_001_production_confines_raw_access() -> None:
    """Protect the durable direct backing-state owner; refactoring is allowed."""
    assert production_access_violations() == ()


def test_arch_ao_access_002_direct_raw_syntax_rejected() -> None:
    """Check only documented direct syntax, never inferred receiver identity."""
    source = """
array_payload = array.data
ordinary = value.as_dataset()
dynamic = value.as_dataset(copy=mode)
private = cache._data
removed = value.unsafe_data
public_raw = value.as_dataset(copy="none")
"""

    assert _accesses(source) == [
        "_data",
        "unsafe_data",
        'as_dataset(copy="none")',
    ]


def test_arch_ao_access_003_owner_files_have_narrow_roles() -> None:
    """Allow owner refactors while preserving direct bind/read authority."""
    analysis_object_source = """
class AnalysisObject:
    def _bind_dataset(self, ds):
        self._data = ds

    def inspect(self, other):
        current = self._data
        other_value = other._data
        self._data = current
        return current, other_value

    def remove(self):
        del self._data

class Unrelated:
    def inspect(self):
        return self._data
"""
    assert _accesses(
        analysis_object_source,
        path="tal/core/analysis_object.py",
    ) == ["_data", "_data"]

    dataset_owner_source = """
def inspect(source):
    return source._data

def mutate(source, replacement):
    source._data = replacement
"""
    assert _accesses(
        dataset_owner_source,
        path="tal/core/dataset_ownership.py",
    ) == ["_data"]


def test_arch_ao_access_004_public_sources_drop_removed_properties() -> None:
    """Keep removed public spellings out of docs and executable examples."""
    paths = [
        *Path("docs").rglob("*.md"),
        *Path("examples").rglob("*.py"),
        *Path("examples").rglob("*.ipynb"),
    ]
    removed = ("unsafe_data", "AnalysisObject.data", "ao.data")
    offenders = {
        path.as_posix(): token
        for path in paths
        for token in removed
        if token in path.read_text(encoding="utf-8")
    }
    assert offenders == {}
