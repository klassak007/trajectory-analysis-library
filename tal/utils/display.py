"""Private presentation entrypoints shared by AnalysisObject subclasses."""

from __future__ import annotations

from itertools import islice

from tal.core.dataset_ownership import analysis_object_dataset

from .display_metadata import MISSING, DisplayRow, core_rows, summary_text
from .display_projection import presentation_view
from .display_values import DETAIL_BUDGET, SUMMARY_BUDGET, format_stored
from .display_xarray import html_display, text_display


def _identity(value: object) -> str:
    cls = type(value)
    module = cls.__module__
    name = cls.__qualname__
    module_label = module[:128] + ("…" if len(module) > 128 else "")
    name_label = name[:128] + ("…" if len(name) > 128 else "")
    if module.startswith("tal."):
        return f"tal.{cls.__name__[:128]}"
    return f"{module_label}.{name_label}"


def _domain_rows(value, schema):
    try:
        contribute = getattr(value, "_display_rows", None)
        domain = () if contribute is None else tuple(islice(contribute(schema), SUMMARY_BUDGET.items + 1))
        if any(type(row) is not DisplayRow or type(row.label) is not str for row in domain):
            raise TypeError("unreadable presentation rows")
        rows = tuple(DisplayRow(row.label[:128], row.value) for row in domain[:SUMMARY_BUDGET.items])
        return rows + ((DisplayRow("Domain", "<items omitted>"),) if len(domain) > len(rows) else ())
    except Exception:  # noqa: BLE001 -- optional presentation hook is fault-isolated
        return (DisplayRow("Domain", "unavailable"),)


def _presentation(value):
    source = analysis_object_dataset(value)
    schema = source.attrs.get("tal", MISSING)
    rows = core_rows(schema) + _domain_rows(value, schema)
    return presentation_view(source), _identity(value), summary_text(rows), schema


def analysis_object_repr(value: object) -> str:
    view, label, summary, _ = _presentation(value)
    return text_display(view, label, summary)


def analysis_object_html(value: object) -> str:
    view, label, summary, schema = _presentation(value)
    text = text_display(view, label, summary)
    detail = "undeclared" if schema is MISSING else format_stored(schema, budget=DETAIL_BUDGET)
    return html_display(view, label, summary, detail, text)
