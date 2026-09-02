from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import tal
import tal.core
from tal.core import dataset_ownership as dataset_owner

from ._budget import file_loc, function_lengths

_OWNER_MODULE = "tal.core.dataset_ownership"
_PRODUCTION_PATHS = (
    Path("tal/core/analysis_object.py"),
    Path("tal/core/dataset_ownership.py"),
)


def _owner_objects() -> tuple[object, ...]:
    exported = tuple(
        getattr(dataset_owner, name)
        for name in getattr(dataset_owner, "__all__", ())
    )
    local = tuple(
        value
        for value in vars(dataset_owner).values()
        if getattr(value, "__module__", None) == _OWNER_MODULE
    )
    candidates = (dataset_owner, *exported, *local)
    return tuple(
        value
        for index, value in enumerate(candidates)
        if not any(value is prior for prior in candidates[:index])
    )


def _public_owner_aliases(module: object) -> set[str]:
    names = getattr(module, "__all__", None)
    if names is None:
        names = tuple(name for name in vars(module) if not name.startswith("_"))
    owners = _owner_objects()
    return {
        name
        for name in names
        if (value := getattr(module, name, None)) is not None
        and any(value is owner for owner in owners)
    }


def test_arch_dataset_ownership_001_owner_is_not_public_core_api() -> None:
    """Protect the public boundary without freezing private call mechanics."""
    assert _public_owner_aliases(tal) == set()
    assert _public_owner_aliases(tal.core) == set()
    leaked = SimpleNamespace(
        __all__=("CopyPolicy",),
        CopyPolicy=dataset_owner.DatasetCopyMode,
    )
    assert _public_owner_aliases(leaked) == {"CopyPolicy"}


def test_arch_dataset_ownership_002_changed_production_stays_within_budgets() -> None:
    """Apply shared AGENTS.md budgets to the two 127C ownership modules."""
    for path in _PRODUCTION_PATHS:
        assert file_loc(path=path) <= 600, f"{path} exceeds file budget"
        for name, length in function_lengths(path).items():
            assert length <= 50, f"{path}:{name} exceeds function budget"
