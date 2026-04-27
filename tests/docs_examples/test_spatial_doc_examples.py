from __future__ import annotations

import pytest

from tests.docs_examples._examples import EXECUTABLE_EXAMPLES
from tests.docs_examples._manifest import required_example_ids


SPATIAL_EXAMPLE_IDS = tuple(sorted(example_id for example_id in required_example_ids() if example_id.startswith("SPATIAL-")))


@pytest.mark.parametrize("example_id", SPATIAL_EXAMPLE_IDS)
def test_spatial_doc_example_executes(example_id: str) -> None:
    EXECUTABLE_EXAMPLES[example_id]()

