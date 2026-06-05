"""Experimental facade for TAL Numba utility mechanics."""

from __future__ import annotations

from tal.utils.block_rows import BlockInputSpec, BlockRows, prepare_block_rows, row_count
from tal.utils.numba_bench import break_even_calls, cold_subprocess, time_once, warm_median
from tal.utils.numba_scan import ScanAxisSpec, ScanInputSpec, ScanRows, prepare_scan_rows, prepare_topology_rows
from tal.utils.numba_stencil import (
    WindowBounds,
    WindowRows,
    backward_window_bounds,
    centered_window_bounds,
    clipped_window_bounds,
    forward_window_bounds,
    prepare_window_rows,
)
from tal.utils.numba_support import njit_kernel, require_numba


__all__ = [
    "BlockInputSpec",
    "BlockRows",
    "ScanAxisSpec",
    "ScanInputSpec",
    "ScanRows",
    "WindowBounds",
    "WindowRows",
    "backward_window_bounds",
    "break_even_calls",
    "centered_window_bounds",
    "clipped_window_bounds",
    "cold_subprocess",
    "forward_window_bounds",
    "njit_kernel",
    "prepare_block_rows",
    "prepare_scan_rows",
    "prepare_topology_rows",
    "prepare_window_rows",
    "require_numba",
    "row_count",
    "time_once",
    "warm_median",
]
