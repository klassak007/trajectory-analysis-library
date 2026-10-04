import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[2] / "docs/api/reducers.md"
EXAMPLES = re.findall(r"^```python\n(.*?)^```$", PAGE.read_text(), flags=re.MULTILINE | re.DOTALL)


@pytest.mark.parametrize("code", EXAMPLES, ids=[f"block-{i}" for i in range(len(EXAMPLES))])
def test_reducer_page_examples_execute_independently(code: str) -> None:
    """ID: GROUP_DOC_071_reducer_page_examples_are_self_contained."""
    # Execute only this repository-owned documentation, with fresh local state.
    exec(compile(code, str(PAGE), "exec"), {})  # noqa: S102
