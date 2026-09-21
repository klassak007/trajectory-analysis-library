from __future__ import annotations

import pytest

from tests.docs_examples._examples import EXECUTABLE_EXAMPLES
from tests.docs_examples._manifest import required_example_ids
from tests.docs_examples._user_guide_examples import USER_GUIDE_EXECUTABLE_EXAMPLES

SPATIAL_EXAMPLE_IDS = tuple(sorted(example_id for example_id in required_example_ids() if example_id.startswith("SPATIAL-")))


@pytest.mark.parametrize("example_id", SPATIAL_EXAMPLE_IDS)
def test_spatial_doc_example_executes(example_id: str) -> None:
    EXECUTABLE_EXAMPLES[example_id]()


def test_spatial_doc_field_build_001() -> None:
    """ID: SPATIAL_DOC_FIELD_BUILD_001."""
    EXECUTABLE_EXAMPLES["SPATIAL-FIELD-BUILD"]()


def test_spatial_doc_field_reader_001() -> None:
    """ID: SPATIAL_DOC_FIELD_READER_001."""
    USER_GUIDE_EXECUTABLE_EXAMPLES[
        "UG-CREATING-SPATIAL-FIELDS-FROM-READERS"
    ]()
