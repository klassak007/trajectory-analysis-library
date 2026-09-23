"""Isolated integration with xarray's native text and HTML section formatters."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from textwrap import fill

import xarray as xr
from xarray.core import formatting

from .display_coordinates import COORDINATE_STYLE, html_coordinates, text_coordinates
from .display_projection import DatasetDisplayView


def _indexes(view):
    indexes = formatting._get_indexes_dict(view.xindexes)
    include_default = xr.get_options()["display_default_indexes"]
    return formatting.filter_nondefault_indexes(indexes, include_default is not True)


def text_display(view: DatasetDisplayView, label: str, summary: str) -> str:
    """Delegate native sections; only identity and TAL rows are added here."""
    width = formatting._calculate_col_width(view.variables)
    max_rows = xr.get_options()["display_max_rows"]
    dimensions = formatting.dim_summary_limited(view.sizes, col_width=width + 1, max_rows=max_rows)
    lines = [
        f"<{label}> Size: {formatting.render_human_readable_nbytes(view.nbytes)}",
        f"{formatting.pretty_print('Dimensions:', width)}({dimensions})",
        "\n".join(fill(line, width=xr.get_options()["display_width"], subsequent_indent="        ")
                  for line in summary.splitlines()),
    ]
    if view.coords:
        lines.append(text_coordinates(view, width, max_rows))
    if view.omitted:
        lines.append("Transform-backed coordinates:\n    " + "\n    ".join(view.omitted))
    unindexed = formatting.unindexed_dims_repr(view.dims, view.source.coords, max_rows=max_rows)
    if unindexed:
        lines.append(unindexed)
    lines.append(formatting.data_vars_repr(view.data_vars, col_width=width, max_rows=max_rows))
    indexes = _indexes(view)
    if indexes:
        lines.append(formatting.indexes_repr(indexes, max_rows=max_rows))
    if view.attrs:
        lines.append(formatting.attrs_repr(view.attrs, max_rows=max_rows))
    return "\n".join(lines)


@dataclass(frozen=True)
class _TextFallback:
    text: str

    def __repr__(self) -> str:
        return self.text


def _html_sections(native, view, summary, detail):
    sections = [native.dim_section(view), f"<pre class='tal-summary'>{escape(summary)}</pre>"]
    if view.coords:
        sections.append(html_coordinates(native, view))
    if view.omitted:
        sections.append("<pre class='tal-summary'>" + escape("\n".join(view.omitted)) + "</pre>")
    sections.append(native.datavar_section(view.data_vars))
    indexes = _indexes(view)
    if indexes:
        sections.append(native.index_section(indexes))
    if view.attrs:
        sections.append(native.attr_section(view.attrs))
    sections.append(native.collapsible_section(
        "TAL schema", details=f"<pre class='tal-schema'>{escape(detail)}</pre>",
        n_items=1, collapsed=True,
    ))
    return sections


def html_display(view: DatasetDisplayView, label: str, summary: str, detail: str, text: str) -> str:
    fallback = f"<pre>{escape(text)}</pre>"
    if xr.get_options()["display_style"] == "text":
        return fallback
    try:
        from xarray.core import formatting_html as native

        required = ("dim_section", "summarize_variable", "_mapping_section", "datavar_section", "index_section",
                    "attr_section", "collapsible_section", "_obj_repr")
        if not all(callable(getattr(native, name, None)) for name in required):
            return fallback
        header = [f"<div class='xr-obj-type'>{escape(label)}</div>",
                  f"<span class='tal-size'>Size: {formatting.render_human_readable_nbytes(view.nbytes)}</span>"]
        sections = _html_sections(native, view, summary, detail)
        result = native._obj_repr(_TextFallback(text), header, sections)
    except Exception:  # noqa: BLE001 -- required fallback for the optional upstream seam
        # Upstream private formatting is an optional rich-display seam. Native
        # third-party repr failures remain outside the supported boundary.
        return fallback
    style = (
        "<style>" + COORDINATE_STYLE + ".tal-display .tal-summary,.tal-display .tal-schema{"
        "white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:.4em;"
        "color:inherit;background:transparent;grid-column:1/-1}"
        ".tal-display .tal-size{margin-left:auto}</style>"
    )
    return f"<div class='tal-display'>{style}{result}</div>"
