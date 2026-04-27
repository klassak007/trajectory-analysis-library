from __future__ import annotations

from pathlib import Path
import re

from tests.docs_examples._user_guide_registry import (
    USER_GUIDE_EXAMPLES_BY_CHAPTER,
    example_ids_in_chapter,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
USER_GUIDE_DIR = REPO_ROOT / "docs" / "user-guide"
DOCS_INDEX = REPO_ROOT / "docs" / "index.md"
PYTHON_BLOCK_RE = re.compile(r"```python\b")

LEGACY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\.set_group\(", "Use `AnalysisObject.from_data(..., batch_dims=...)` or `set_roles(...)` instead of `.set_group(...)`."),
    (r"\.sample\(", "Use `sequence_dim` role declaration and accessor APIs instead of `.sample(...)`."),
    (r"\.time_coord\(", "Use `param_coord` semantics (`from_data(..., param_coord=...)` or `set_param_coord(...)`) instead of `.time_coord(...)`."),
    (r"\bao\.at\(", "Use `ao.param.at(...)` instead of `ao.at(...)`."),
    (r"\bao\.resample_to\(", "Use `ao.param.resample_to(...)` instead of `ao.resample_to(...)`."),
    (r"\bao\.interp_like\(", "Use `ao.param.interp_like(...)` instead of `ao.interp_like(...)`."),
    (r"\bao\.index_coord\(", "Use `ao.param.index(...)` instead of `ao.index_coord(...)`."),
    (r"\bao\.groupby\(", "Use `ao.group.groupby(...)` instead of direct `ao.groupby(...)`."),
    (r"examples/0[3-9]_\*", "Use `examples/release/` paths instead of legacy notebook globs."),
    (r"examples/1[0-9]_\*", "Use `examples/release/` paths instead of legacy notebook globs."),
)


def _load_user_guide_text() -> str:
    parts: list[str] = []
    for path in sorted(USER_GUIDE_DIR.glob("*.md")):
        parts.append(f"\n# FILE: {path}\n")
        parts.append(path.read_text(encoding="utf-8"))
    parts.append(f"\n# FILE: {DOCS_INDEX}\n")
    parts.append(DOCS_INDEX.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_user_guide_has_no_legacy_api_patterns() -> None:
    corpus = _load_user_guide_text()
    failures: list[str] = []
    for pattern, message in LEGACY_PATTERNS:
        if re.search(pattern, corpus):
            failures.append(f"{pattern!r}: {message}")
    assert not failures, "\n".join(failures)

def test_user_guide_avoids_version_transition_language() -> None:
    corpus = _load_user_guide_text().lower()
    assert "changes for v3" not in corpus
    assert "migration" not in corpus


def test_user_guide_python_blocks_are_annotated_for_example_coverage() -> None:
    failures: list[str] = []
    for chapter in sorted(USER_GUIDE_EXAMPLES_BY_CHAPTER):
        content = (USER_GUIDE_DIR / f"{chapter}.md").read_text(encoding="utf-8")
        python_blocks = len(PYTHON_BLOCK_RE.findall(content))
        annotated = len(example_ids_in_chapter(chapter))
        if python_blocks != annotated:
            failures.append(
                f"{chapter}: python block count {python_blocks} does not match annotated example count {annotated}"
            )
    assert not failures, "\n".join(failures)
