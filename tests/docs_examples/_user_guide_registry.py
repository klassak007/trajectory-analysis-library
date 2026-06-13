from __future__ import annotations

from pathlib import Path
import re

from tests.docs_examples._user_guide_examples import USER_GUIDE_EXECUTABLE_EXAMPLES


REPO_ROOT = Path(__file__).resolve().parents[2]
USER_GUIDE_DIR = REPO_ROOT / "docs" / "user-guide"
EXAMPLE_ID_RE = re.compile(r"<!--\s*example-id:\s*([A-Z0-9-]+)\s*-->")

USER_GUIDE_EXAMPLES_BY_CHAPTER: dict[str, tuple[str, ...]] = {
    "overview": ("UG-OVERVIEW-BASIC-WORKFLOW",),
    "core_concepts": ("UG-CORE-CONCEPTS-ROLES",),
    "creating_trajectory_objects": ("UG-CREATING-SEQUENCE-AO",),
    "indexing": ("UG-INDEXING-PARAM-QUERY",),
    "time": ("UG-TIME-SYNCHRONIZE",),
    "events": ("UG-EVENTS-WINDOWS",),
    "linalg": ("UG-LINALG-BASIC",),
    "numpy": ("UG-NUMPY-UFUNCS",),
    "spatial": ("UG-SPATIAL-POSE",),
    "geo": ("UG-GEO-LLA", "UG-GEO-CONVERSION", "UG-GEO-ENU", "UG-GEO-DISTANCE", "UG-GEO-INTERPOLATION"),
    "frames": ("UG-FRAMES-BASIC",),
    "viewing": ("UG-VIEWING-SCHEMA",),
}


def iter_user_guide_example_ids() -> tuple[str, ...]:
    seen: list[str] = []
    for ids in USER_GUIDE_EXAMPLES_BY_CHAPTER.values():
        for example_id in ids:
            if example_id not in seen:
                seen.append(example_id)
    return tuple(seen)


def chapter_path(chapter: str) -> Path:
    return USER_GUIDE_DIR / f"{chapter}.md"


def example_ids_in_chapter(chapter: str) -> tuple[str, ...]:
    content = chapter_path(chapter).read_text(encoding="utf-8")
    return tuple(EXAMPLE_ID_RE.findall(content))


def validate_user_guide_registry() -> list[str]:
    missing: list[str] = []
    for example_id in iter_user_guide_example_ids():
        if example_id not in USER_GUIDE_EXECUTABLE_EXAMPLES:
            missing.append(example_id)
    return missing
