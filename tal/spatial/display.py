"""Stored-only spatial presentation contributions; no graph or semantic reads."""

from tal.utils.display_metadata import (
    DisplayRow,
    optional_row,
    stored_name,
    stored_path,
)


def _frames(block):
    if type(block) is not dict:
        raise ValueError("unreadable stored frames")
    return {name: stored_name(block[name], nullable=True) for name in ("parent", "child") if name in block}


def _representation(block):
    return {"representation": stored_name(stored_path(block, "rep"))}


def spatial_display_rows(schema: object) -> tuple[DisplayRow, ...]:
    rows = (
        optional_row("Frames", schema, ("ext", "frames"), _frames),
        optional_row("Spatial", schema, ("ext", "spatial", "representation"), _representation),
    )
    return tuple(row for row in rows if row is not None)
