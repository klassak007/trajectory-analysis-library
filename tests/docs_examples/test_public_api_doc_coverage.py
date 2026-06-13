from __future__ import annotations

import inspect
import re

from tests.docs_examples._examples import EXECUTABLE_EXAMPLES
from tests.docs_examples._manifest import (
    CURATED_SCOPE_COUNTS,
    DOCSTRING_SECTION_REQUIREMENTS,
    DUUNDER_FAMILY_DOC_OWNER,
    EXAMPLE_REQUIRED_SYMBOLS,
    _resolve_symbol,
    curated_scope_counts,
    inventory_required_example_ids,
    iter_inventory_example_symbols,
    iter_curated_public_symbols,
    iter_scoped_public_symbols,
    required_example_ids,
)


NUMBA_AUTOSUMMARY_RAISES_SYMBOLS = {
    "centered_window_bounds",
    "clipped_window_bounds",
    "forward_window_bounds",
    "backward_window_bounds",
    "prepare_block_rows",
    "prepare_scan_rows",
    "prepare_topology_rows",
    "prepare_window_rows",
    "require_numba",
    "warm_median",
    "cold_subprocess",
}


def _docstring(obj: object) -> str:
    doc = getattr(obj, "__doc__", None)
    return doc if isinstance(doc, str) else ""


def _summary_line(doc: str) -> str:
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _normalized_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _extract_examples_block(doc: str) -> str:
    match = re.search(
        r"(?ms)^\s*Examples\s*\n\s*-{3,}\s*\n(?P<body>.*?)(?:\n\s*[A-Z][A-Za-z ]*\s*\n\s*-{3,}\s*\n|\Z)",
        doc,
    )
    if match is None:
        return ""
    return match.group("body")


def _has_section(doc: str, section: str) -> bool:
    return re.search(rf"(?m)^\s*{re.escape(section)}\s*\n\s*-{{3,}}\s*$", doc) is not None


def _numba_autosummary_symbols() -> tuple[str, ...]:
    text = open("docs/api/numba.md", encoding="utf-8").read()
    return tuple(re.findall(r"(?m)^\s+(tal\.utils\.numba\.[A-Za-z_][A-Za-z0-9_]*)\s*$", text))


def test_curated_scope_counts_match_plan() -> None:
    expected_total = 242
    observed = curated_scope_counts()
    assert observed == CURATED_SCOPE_COUNTS
    assert sum(observed.values()) == expected_total


def test_scoped_public_symbols_have_docstrings() -> None:
    missing: list[str] = []
    for record in iter_scoped_public_symbols():
        if not _docstring(record.obj).strip():
            missing.append(record.symbol)
    assert not missing, f"Missing docstrings for public scoped symbols: {missing!r}"


def test_scoped_docstrings_have_informative_summaries() -> None:
    failures: list[str] = []
    for record in iter_scoped_public_symbols():
        doc = _docstring(record.obj).strip()
        if not doc:
            continue
        summary = _summary_line(doc)
        if not summary:
            failures.append(f"{record.symbol}: missing summary line")
            continue
        if not summary.endswith("."):
            failures.append(f"{record.symbol}: summary should end with a period")
        if len(summary.split()) < 2:
            failures.append(f"{record.symbol}: summary is too short")
        symbol_leaf = record.symbol.rsplit(".", 1)[-1]
        if _normalized_token(summary.rstrip(".")) == _normalized_token(symbol_leaf):
            failures.append(f"{record.symbol}: summary repeats symbol name without semantics")
    assert not failures, "\n".join(failures)


def test_scoped_docstrings_use_numpy_style_sections() -> None:
    failures: list[str] = []
    forbidden = (
        r"(?m)^\s*Args:\s*$",
        r"(?m)^\s*Arguments:\s*$",
        r"(?m)^\s*Keyword Args:\s*$",
        r"(?m)^\s*Example:\s*$",
    )
    for record in iter_scoped_public_symbols():
        doc = _docstring(record.obj)
        for pattern in forbidden:
            if re.search(pattern, doc):
                failures.append(f"{record.symbol}: contains non-NumPy section header matching {pattern!r}")
    assert not failures, "\n".join(failures)


def test_required_docstring_sections_present() -> None:
    index = {record.symbol: record for record in iter_scoped_public_symbols()}
    failures: list[str] = []
    for symbol, sections in DOCSTRING_SECTION_REQUIREMENTS.items():
        record = index.get(symbol)
        if record is None:
            failures.append(f"{symbol}: symbol not found in manifest scope")
            continue
        doc = _docstring(record.obj)
        for section in sections:
            pattern = rf"(?m)^\s*{re.escape(section)}\s*\n\s*-{{3,}}\s*$"
            if re.search(pattern, doc) is None:
                failures.append(f"{symbol}: missing section '{section}'")
    assert not failures, "\n".join(failures)


def test_numba_public_autosummary_symbols_have_numpy_docstrings() -> None:
    failures: list[str] = []
    symbols = _numba_autosummary_symbols()
    assert symbols, "docs/api/numba.md must list tal.utils.numba autosummary symbols"
    for symbol in symbols:
        obj = _resolve_symbol(symbol)
        doc = _docstring(obj)
        if not doc.strip():
            failures.append(f"{symbol}: missing docstring")
            continue
        required = ["Parameters"]
        if inspect.isfunction(obj):
            required.extend(["Returns", "Examples"])
        if symbol.rsplit(".", 1)[-1] in NUMBA_AUTOSUMMARY_RAISES_SYMBOLS:
            required.append("Raises")
        for section in required:
            if not _has_section(doc, section):
                failures.append(f"{symbol}: missing section '{section}'")
        if inspect.isfunction(obj) and ">>>" not in _extract_examples_block(doc):
            failures.append(f"{symbol}: Examples block must include runnable '>>>' snippet(s)")
    assert not failures, "\n".join(failures)


def test_required_example_symbols_contain_runnable_snippets() -> None:
    index = {record.symbol: record for record in iter_scoped_public_symbols()}
    failures: list[str] = []
    for symbol in EXAMPLE_REQUIRED_SYMBOLS:
        record = index.get(symbol)
        if record is None:
            failures.append(f"{symbol}: symbol not found in manifest scope")
            continue
        block = _extract_examples_block(_docstring(record.obj))
        if ">>>" not in block:
            failures.append(f"{symbol}: Examples block must include runnable '>>>' snippet(s)")
    assert not failures, "\n".join(failures)


def test_inventory_example_symbols_contain_runnable_snippets() -> None:
    failures: list[str] = []
    for record in iter_inventory_example_symbols():
        block = _extract_examples_block(_docstring(record.obj))
        if ">>>" not in block:
            failures.append(f"{record.symbol}: Examples block must include runnable '>>>' snippet(s)")
    assert not failures, "\n".join(failures)


def test_curated_public_callables_include_examples() -> None:
    failures: list[str] = []
    for record in iter_curated_public_symbols():
        if record.kind == "property":
            continue
        block = _extract_examples_block(_docstring(record.obj))
        if ">>>" not in block:
            failures.append(record.symbol)
    assert not failures, "Public callable docstrings missing Examples snippets:\n" + "\n".join(failures)


def test_docstrings_do_not_expose_internal_example_ids() -> None:
    failures: list[str] = []
    for record in iter_scoped_public_symbols():
        doc = _docstring(record.obj)
        if re.search(r"Example ID:\s*[A-Z0-9-]+", doc):
            failures.append(record.symbol)
    assert not failures, f"Docstrings expose internal Example IDs: {failures!r}"


def test_required_example_ids_have_executable_tests() -> None:
    required = required_example_ids()
    available = set(EXECUTABLE_EXAMPLES)
    missing = sorted(required - available)
    assert not missing, f"Missing executable example handlers: {missing!r}"


def test_inventory_example_ids_have_executable_tests() -> None:
    required = inventory_required_example_ids()
    available = set(EXECUTABLE_EXAMPLES)
    missing = sorted(required - available)
    assert not missing, f"Missing executable example handlers: {missing!r}"


def test_operator_family_docs_present_on_class_owners() -> None:
    index = {record.symbol: record for record in iter_scoped_public_symbols()}
    failures: list[str] = []
    for symbol, required_snippets in DUUNDER_FAMILY_DOC_OWNER.items():
        record = index.get(symbol)
        if record is None:
            failures.append(f"{symbol}: class not found in manifest scope")
            continue
        doc_lower = _docstring(record.obj).lower()
        for snippet in required_snippets:
            if snippet.lower() not in doc_lower:
                failures.append(f"{symbol}: missing operator-family snippet '{snippet}'")
    assert not failures, "\n".join(failures)
