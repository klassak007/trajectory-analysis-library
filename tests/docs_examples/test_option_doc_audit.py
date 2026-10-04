from __future__ import annotations

import inspect
import re

from tests.docs_examples._manifest import iter_curated_public_symbols
from tests.docs_examples._options_audit_checklist import (
    CURATED_OPTION_AUDIT_CHECKLIST,
    CURATED_OPTION_AUDIT_COUNT,
)

_BATCH_GROUPED_REDUCER_SYMBOLS = tuple(
    f"tal.core.group_ops.batch_view.BatchGroupedView.{name}"
    for name in (
        "all",
        "any",
        "count",
        "max",
        "mean",
        "median",
        "min",
        "std",
        "sum",
        "var",
    )
)
_SHARED_OPTION_AUDIT_SYMBOLS = frozenset(
    {
        "tal.core.group_ops.batch_view.BatchGroupedView",
        *_BATCH_GROUPED_REDUCER_SYMBOLS,
    }
)


def _docstring(obj: object) -> str:
    doc = getattr(obj, "__doc__", None)
    return doc if isinstance(doc, str) else ""


def _extract_section(doc: str, section: str) -> str:
    match = re.search(
        rf"(?ms)^\s*{re.escape(section)}\s*\n\s*-{{3,}}\s*\n(?P<body>.*?)(?:\n\s*[A-Z][A-Za-z ]*\s*\n\s*-{{3,}}\s*\n|\Z)",
        doc,
    )
    return "" if match is None else match.group("body")


def _has_section(doc: str, section: str) -> bool:
    return bool(_extract_section(doc, section).strip())


def _extract_examples_block(doc: str) -> str:
    return _extract_section(doc, "Examples")


def _extract_param_description(doc: str, param_name: str) -> str:
    lines = doc.splitlines()
    for idx, line in enumerate(lines):
        if re.match(rf"^\s*{re.escape(param_name)}\s*:\s*.+$", line):
            body: list[str] = []
            for next_line in lines[idx + 1 :]:
                if not next_line.strip() and body:
                    break
                if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*.+$", next_line):
                    break
                if re.match(r"^\s*[A-Z][A-Za-z ]+$", next_line):
                    break
                if next_line.startswith((" ", "\t")):
                    body.append(next_line.strip())
            return " ".join(body)
    return ""


def _option_symbol_rows_from_manifest() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for record in iter_curated_public_symbols():
        if record.symbol in _SHARED_OPTION_AUDIT_SYMBOLS:
            continue
        obj = record.obj
        if isinstance(obj, property):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        for name in signature.parameters:
            if name in {"opts", "options"}:
                rows.append((record.symbol, name))
                break
    return sorted(rows)




def test_option_checklist_scope_is_frozen() -> None:
    assert CURATED_OPTION_AUDIT_COUNT == len(CURATED_OPTION_AUDIT_CHECKLIST)
    derived = _option_symbol_rows_from_manifest()
    expected = sorted((item.symbol, item.option_param) for item in CURATED_OPTION_AUDIT_CHECKLIST)
    assert derived == expected


def test_batch_grouped_reducers_share_one_verified_option_doc_family() -> None:
    """ID: GROUP_DOC_061_batch_grouped_options_share_verified_family."""
    index = {record.symbol: record for record in iter_curated_public_symbols()}
    failures: list[str] = []
    for symbol in _BATCH_GROUPED_REDUCER_SYMBOLS:
        obj = index[symbol].obj
        signature = inspect.signature(obj)
        if "opts" not in signature.parameters:
            failures.append(f"{symbol}: missing opts parameter")
        doc = _docstring(obj)
        for section in ("Parameters", "Returns", "Raises", "Notes", "Examples"):
            if not _has_section(doc, section):
                failures.append(f"{symbol}: missing {section} section")
        description = _extract_param_description(doc, "opts")
        required = ("GroupMaterializeOptions", "BatchGroupReduceOptions")
        if not all(name in description for name in required):
            failures.append(f"{symbol}: opts description omits a topology option type")
    assert not failures, "\n".join(failures)


def test_batch_grouped_view_documents_return_only_construction() -> None:
    """ID: GROUP_DOC_062_batch_grouped_view_is_return_only."""
    index = {record.symbol: record for record in iter_curated_public_symbols()}
    doc = _docstring(index["tal.core.group_ops.batch_view.BatchGroupedView"].obj)
    assert "return-only result type" in doc
    assert not _has_section(doc, "Parameters")


def test_option_checklist_rows_are_closed() -> None:
    failures: list[str] = []
    for item in CURATED_OPTION_AUDIT_CHECKLIST:
        flags = (
            item.has_parameters,
            item.has_opts_entry,
            item.has_returns,
            item.has_raises,
            item.has_notes,
            item.has_examples,
            item.has_nondefault_opts_example,
        )
        if not all(flags):
            failures.append(item.symbol)
    assert not failures, f"Option checklist rows not fully closed: {failures!r}"


def test_option_docstrings_have_non_placeholder_inline_option_summaries() -> None:
    index = {record.symbol: record for record in iter_curated_public_symbols()}
    failures: list[str] = []
    banned_phrases = (
        "controls operation behavior.",
        "optional options controlling policy and numeric behavior for this operation.",
    )
    for item in CURATED_OPTION_AUDIT_CHECKLIST:
        record = index[item.symbol]
        doc = _docstring(record.obj)
        description = _extract_param_description(doc, item.option_param).strip()
        if not description:
            failures.append(f"{item.symbol}: missing description for `{item.option_param}`")
            continue
        lowered = description.lower()
        for phrase in banned_phrases:
            if phrase in lowered:
                failures.append(f"{item.symbol}: placeholder option summary text")
                break
        field_tokens = re.findall(r"``[^`]+``", description)
        if len(field_tokens) < 2:
            failures.append(f"{item.symbol}: option summary lacks concrete option fields")
    assert not failures, "\n".join(failures)
