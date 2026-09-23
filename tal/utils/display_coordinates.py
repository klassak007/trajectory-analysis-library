"""Coordinate Role column around xarray's native row and section formatters."""

from __future__ import annotations

from functools import partial
from html import escape

import xarray as xr
from xarray.core import formatting


def _ordered_coordinates(coords):
    dimensions = {name: index for index, name in enumerate(coords.sizes)}
    return dict(sorted(coords.items(), key=lambda item: dimensions.get(item[0], len(dimensions))))


def _text_row(name, variable, col_width, *, roles, role_width, is_index=False):
    # The native formatter's fixed-width name column is the isolated text seam.
    # Reserve space before asking xarray to select its native value preview.
    row = formatting.summarize_variable(
        name, variable, col_width, is_index=is_index,
        max_width=xr.get_options()["display_width"] - role_width,
    )
    return row[:col_width] + roles[name].ljust(role_width) + row[col_width:]


def text_coordinates(view, col_width, max_rows):
    role_width = max(len("Role"), *(len(role) for role in view.roles.values())) + 2
    section = formatting._mapping_repr(
        _ordered_coordinates(view.coords), title="Coordinates",
        summarizer=partial(_text_row, roles=view.roles, role_width=role_width),
        expand_option_name="display_expand_coords", col_width=col_width,
        indexes=view.xindexes, max_rows=max_rows,
    )
    heading, separator, rows = section.partition("\n")
    if not separator:
        return heading
    return f"{heading}\n{' ' * col_width}Role\n{rows}"


def _html_rows(coords, *, native, roles):
    rows = ["<li class='tal-coordinate-heading'><span class='tal-coordinate-role'>Role</span></li>"]
    for name, variable in _ordered_coordinates(coords).items():
        content = native.summarize_variable(name, variable, is_index=name in coords.xindexes)
        role = f"<div class='tal-coordinate-role'>{escape(roles[name])}</div>"
        rows.append(f"<li class='xr-var-item tal-coordinate-row'>{content}{role}</li>")
    return "<ul class='xr-var-list tal-coordinates'>" + "".join(rows) + "</ul>"


def html_coordinates(native, view):
    return native._mapping_section(
        view.coords, name="Coordinates",
        details_func=partial(_html_rows, native=native, roles=view.roles),
        max_items_collapse=25, expand_option_name="display_expand_coords",
    )


# A coordinate-local grid adds one column without changing native Dataset,
# data-variable, index, or expanded attribute/data sections. Row markup and
# independent controls still come from xarray's native variable formatter.
COORDINATE_STYLE = """
.tal-display .tal-coordinates {
  display:grid; grid-column:1/-1; list-style:none; margin:0; padding:0!important;
  grid-template-columns:minmax(65px,150px) minmax(65px,max-content) auto auto minmax(0,1fr) 0 20px 0 20px;
}
.tal-display .tal-coordinate-row,.tal-display .tal-coordinate-heading {
  display:grid; grid-column:1/-1; grid-template-columns:subgrid;
}
.tal-display .tal-coordinate-role {
  grid-column:2; grid-row:1; padding-right:10px; overflow-wrap:anywhere; font-size:.9em;
}
.tal-display .tal-coordinate-heading {color:var(--xr-font-color2);}
.tal-display .tal-coordinate-row > .xr-var-name {grid-column:1; grid-row:1;}
.tal-display .tal-coordinate-row > .xr-var-dims {grid-column:3; grid-row:1;}
.tal-display .tal-coordinate-row > .xr-var-dtype {grid-column:4; grid-row:1;}
.tal-display .tal-coordinate-row > .xr-var-preview {grid-column:5; grid-row:1;}
.tal-display .tal-coordinate-row > .xr-var-attrs-in {grid-column:6; grid-row:1;}
.tal-display .tal-coordinate-row > .xr-var-attrs-in + label {grid-column:7; grid-row:1;}
.tal-display .tal-coordinate-row > .xr-var-data-in {grid-column:8; grid-row:1;}
.tal-display .tal-coordinate-row > .xr-var-data-in + label {grid-column:9; grid-row:1;}
"""
