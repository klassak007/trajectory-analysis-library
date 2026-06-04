"""Experimental facade for TAL Numba utility mechanics."""

from __future__ import annotations

from tal.utils.block_rows import BlockInputSpec, BlockRows, prepare_block_rows, row_count
from tal.utils.numba_scan import ScanAxisSpec, ScanInputSpec, ScanRows, prepare_scan_rows
from tal.utils.numba_stencil import (
    WindowBounds,
    backward_window_bounds,
    centered_window_bounds,
    clipped_window_bounds,
    forward_window_bounds,
)
from tal.utils.numba_support import njit_kernel, require_numba


__all__ = [
    "BlockInputSpec",
    "BlockRows",
    "ScanAxisSpec",
    "ScanInputSpec",
    "ScanRows",
    "WindowBounds",
    "backward_window_bounds",
    "centered_window_bounds",
    "clipped_window_bounds",
    "forward_window_bounds",
    "njit_kernel",
    "prepare_block_rows",
    "prepare_scan_rows",
    "require_numba",
    "row_count",
]
