import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "docs/api/types/path_solve.md"


def test_bound_pose_page_example_executes():
    """ID: BIND_127H_015_published_bound_pose_example_executes."""
    section = PAGE.read_text().split("## Minimal Example\n", 1)[1]
    code = re.search(r"```python\n(.*?)\n```", section, re.DOTALL).group(1)
    exec(compile(code, str(PAGE), "exec"), {})  # noqa: S102
