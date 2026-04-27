from __future__ import annotations

import inspect
import re

from tests.docs_examples._manifest import iter_curated_public_symbols
from tests.docs_examples._options_audit_checklist import (
    CURATED_OPTION_AUDIT_CHECKLIST,
    CURATED_OPTION_AUDIT_COUNT,
)


def _docstring(obj: object) -> str:
    doc = getattr(obj, "__doc__", None)
    return doc if isinstance(doc, str) else ""


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
                if next_line.startswith(" ") or next_line.startswith("\t"):
                    body.append(next_line.strip())
            return " ".join(body)
    return ""


def _option_symbol_rows_from_manifest() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for record in iter_curated_public_symbols():
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
    assert CURATED_OPTION_AUDIT_COUNT == 104
    derived = _option_symbol_rows_from_manifest()
    expected = sorted((item.symbol, item.option_param) for item in CURATED_OPTION_AUDIT_CHECKLIST)
    assert derived == expected


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
