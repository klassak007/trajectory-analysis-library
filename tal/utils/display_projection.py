"""Read-only xarray presentation views with transform-backed values omitted."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import xarray as xr

from .display_metadata import MISSING, coordinate_role_labels
from .display_values import SUMMARY_BUDGET, format_stored, type_label


@dataclass(frozen=True)
class DatasetDisplayView:
    source: xr.Dataset
    preview: xr.Dataset
    omitted: tuple[str, ...]
    roles: dict

    @property
    def sizes(self):
        return self.source.sizes

    @property
    def dims(self):
        return self.source.dims

    @property
    def nbytes(self):
        return self.source.nbytes

    @property
    def variables(self):
        return self.preview.variables

    @property
    def coords(self):
        return self.preview.coords

    @property
    def data_vars(self):
        return self.preview.data_vars

    @property
    def xindexes(self):
        return self.preview.xindexes

    @property
    def attrs(self):
        return {name: value for name, value in self.source.attrs.items() if name != "tal"}


def presentation_view(source: xr.Dataset) -> DatasetDisplayView:
    names = []
    omitted = []
    for index, coordinates in source.xindexes.group_by_index():
        if not isinstance(index, xr.indexes.CoordinateTransformIndex):
            continue
        names.extend(coordinates)
        displayed = tuple(islice(coordinates, SUMMARY_BUDGET.items))
        if len(coordinates) > len(displayed):
            displayed += ("<items omitted>",)
        description = format_stored(displayed, budget=SUMMARY_BUDGET, quote_strings=False)
        omitted.append(f"{description}: {type_label(index)}; values omitted: transform-backed")
    # Structural xarray projection shares buffers/graphs, without the public
    # ownership-copy or schema-update paths. Original sizes include orphan dims.
    preview = source.drop_vars(names) if names else source
    roles = coordinate_role_labels(source.attrs.get("tal", MISSING), preview.coords)
    return DatasetDisplayView(source, preview, tuple(omitted), roles)
