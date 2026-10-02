"""Durable Contract 135 budget boundary for query and first-party extensions.

Runtime tests cannot measure source complexity. Shared tooling recognizes local
definitions, declared signatures and nested control statements only; it does not
infer receiver types or runtime values. Arbitrary private refactoring and module
splits remain permitted within the AGENTS.md budgets. Import relationships belong
to Import Linter, rather than a duplicate pytest scan.
"""

from pathlib import Path

import pytest

from tools.architecture_budget import (
    file_loc,
    function_lengths,
    function_parameter_counts,
    function_control_depths,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ROOT / "tal/core/param_ops/query.py",
    *sorted((ROOT / "extensions/src/tal_extensions").rglob("*.py")),
)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_distribution_complexity_budgets(path):
    assert file_loc(path=path) <= 600
    assert all(size <= 50 for size in function_lengths(path).values())
    assert all(count <= 10 for count in function_parameter_counts(path).values())
    assert all(depth <= 2 for depth in function_control_depths(path).values())
