from __future__ import annotations

import pytest

from tests.docs_examples._user_guide_examples import USER_GUIDE_EXECUTABLE_EXAMPLES
from tests.docs_examples._user_guide_registry import (
    USER_GUIDE_EXAMPLES_BY_CHAPTER,
    example_ids_in_chapter,
    iter_user_guide_example_ids,
    validate_user_guide_registry,
)


def test_user_guide_registry_is_resolved() -> None:
    missing = validate_user_guide_registry()
    assert not missing, f"Missing user-guide example handlers: {missing!r}"


@pytest.mark.parametrize("example_id", tuple(sorted(iter_user_guide_example_ids())))
def test_user_guide_example_executes(example_id: str) -> None:
    if example_id == "UG-ASTRO-DIRECTION":
        pytest.importorskip("astropy")
    USER_GUIDE_EXECUTABLE_EXAMPLES[example_id]()


@pytest.mark.parametrize("chapter", tuple(sorted(USER_GUIDE_EXAMPLES_BY_CHAPTER)))
def test_each_chapter_has_registered_and_annotated_examples(chapter: str) -> None:
    registered = USER_GUIDE_EXAMPLES_BY_CHAPTER[chapter]
    annotated = example_ids_in_chapter(chapter)
    assert registered, f"Chapter {chapter!r} must define at least one user-guide example"
    assert tuple(registered) == annotated


def test_user_guide_registry_has_no_orphaned_examples() -> None:
    registered = set(iter_user_guide_example_ids())
    handlers = set(USER_GUIDE_EXECUTABLE_EXAMPLES)
    assert handlers == registered
